import optuna
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
from VGG_16_custom import DeepFakeDetection
from data_methods import create_tensors_from_csv, calculate_metrics, load_csv_data
from constants import *

# Early stopping implementation
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def objective(trial):
    """
    Optuna objective function for hyperparameter tuning using training and validation sets.
    """

    # Hyperparameter search space
    learning_rate = trial.suggest_loguniform("learning_rate", 0.00001, 0.001)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout = trial.suggest_uniform("dropout", 0.2, 0.8)

    # Print the current trial parameters
    print(f"Current trial parameters: {trial.params}")

    # Model initialization
    model = DeepFakeDetection(
        epochs=10, batch_size=batch_size, learning_rate=learning_rate
    ).to(DEVICE)

    # Apply dynamic dropout to the model
    for layer in model.children():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = dropout

    # Loss, optimizer, and scheduler
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=5)

    # Load training and validation data
    x_train_paths, X_train_features, train_labels = load_csv_data(TRAIN_CSV)
    x_train_paths = [os.path.join(WAV2VEC_FOLDER, p) for p in x_train_paths]
    x_validation_paths, X_validation_features, validation_labels = load_csv_data(VALIDATION_CSV)
    x_validation_paths = [os.path.join(WAV2VEC_FOLDER, p) for p in x_validation_paths]

    # Normalize data
    mean = np.mean(X_train_features, axis=0)
    std = np.std(X_train_features, axis=0)
    X_train_features_normalized = (X_train_features - mean) / std
    X_validation_features_normalized = (X_validation_features - mean) / std

    train_loss, val_loss = 0, 0
    for epoch in range(10):  # Limited epochs for optimization
        model.train()
        train_loss = 0
        for i in tqdm(range(0, len(x_train_paths), batch_size)):
            x_wav2vec_batch, x_features_batch, y_batch = create_tensors_from_csv(
                x_train_paths, X_train_features_normalized, train_labels, i, batch_size
            )
            x_wav2vec_batch, x_features_batch, y_batch = (
                x_wav2vec_batch.to(DEVICE),
                x_features_batch.to(DEVICE),
                y_batch.to(DEVICE),
            )

            optimizer.zero_grad()
            y_pred = model(x_wav2vec_batch, x_features_batch).squeeze()
            y_batch = y_batch.view(-1)
            y_pred = y_pred.squeeze(-1)
            loss = criterion(y_pred, y_batch.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        all_y_true, all_y_pred = [], []
        with torch.no_grad():
            for i in tqdm(range(0, len(x_validation_paths), batch_size)):
                x_wav2vec_batch, x_features_batch, y_batch = create_tensors_from_csv(
                    x_validation_paths,
                    X_validation_features_normalized,
                    validation_labels,
                    i,
                    batch_size,
                )
                x_wav2vec_batch, x_features_batch, y_batch = (
                    x_wav2vec_batch.to(DEVICE),
                    x_features_batch.to(DEVICE),
                    y_batch.to(DEVICE),
                )
                y_pred = model(x_wav2vec_batch, x_features_batch).squeeze()
                val_loss += criterion(y_pred.squeeze(), y_batch.float()).item()
                all_y_true.extend(y_batch.cpu().numpy())
                all_y_pred.extend(y_pred.squeeze().cpu())

        # Compute validation metrics
        val_loss /= len(x_validation_paths) // batch_size
        binary_y_pred = (np.array(all_y_pred) > 0.5).astype(int)
        accuracy, recall, f1 = calculate_metrics(np.array(all_y_true), binary_y_pred)

        print(f'Train Loss = {train_loss}, Validation Loss = {val_loss}, Accuracy = {accuracy}, Recall = {recall}, F1 = {f1}')

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    return val_loss


def evaluate_on_test(model, test_csv, mean, std, batch_size):
    """
    Evaluate the model on the test set after tuning.
    """
    x_test_paths, X_test_features, test_labels = load_csv_data(test_csv)
    X_test_features_normalized = (X_test_features - mean) / std

    model.eval()
    test_loss = 0
    all_y_true, all_y_pred = [], []

    criterion = torch.nn.BCELoss()

    with torch.no_grad():
        for i in range(0, len(x_test_paths), batch_size):
            x_wav2vec_batch, x_features_batch, y_batch = create_tensors_from_csv(
                x_test_paths, X_test_features_normalized, test_labels, i, batch_size
            )
            x_wav2vec_batch, x_features_batch, y_batch = (
                x_wav2vec_batch.to(DEVICE),
                x_features_batch.to(DEVICE),
                y_batch.to(DEVICE),
            )
            y_pred = model(x_wav2vec_batch, x_features_batch).squeeze()
            test_loss += criterion(y_pred.squeeze(), y_batch.float()).item()
            all_y_true.extend(y_batch.cpu().numpy())
            all_y_pred.extend(y_pred.squeeze().cpu())

    test_loss /= len(x_test_paths) // batch_size
    binary_y_pred = (np.array(all_y_pred) > 0.5).astype(int)
    accuracy, recall, f1 = calculate_metrics(np.array(all_y_true), binary_y_pred)

    print(f"Test Loss = {test_loss:.4f}, Accuracy = {accuracy:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")


# Run Optuna optimization
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    # Retrain the model with the best parameters on train and validation data
    best_model = DeepFakeDetection(
        epochs=10,
        batch_size=best_params["batch_size"],
        learning_rate=best_params["learning_rate"],
    ).to(DEVICE)

    x_train_paths, X_train_features, train_labels = load_csv_data(TRAIN_CSV)
    x_train_paths = [os.path.join(WAV2VEC_FOLDER, p) for p in x_train_paths]

    # Normalize data
    mean = np.mean(X_train_features, axis=0)
    std = np.std(X_train_features, axis=0)

    # Evaluate on test data
    evaluate_on_test(best_model, TEST_CSV, mean, std, best_params["batch_size"])
