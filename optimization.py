import copy
import csv
import shutil
from datetime import datetime
import optuna
import os
import numpy as np
from tqdm import tqdm
from VGG_16_custom import DeepFakeDetection
from data_methods import create_tensors_from_csv, calculate_metrics, load_csv_data
from constants import *
import signal
import sys

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

    best_val_f1 = 0.0
    best_model_path = None
    best_val_loss = float('inf')

    # Hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.2, 0.65)
    layers = trial.suggest_categorical("dense_layers", [i for i in range(2, 7)])

    # Print the current trial parameters
    print(f"Current trial parameters: {trial.params}")

    # Model initialization
    model = DeepFakeDetection(
        batch_size=batch_size,
        learning_rate=learning_rate,
        dense_layers= layers
    ).to(DEVICE)

    # Apply dynamic dropout to the model
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout

    # Loss, optimizer, and scheduler
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=10)

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

    for epoch in tqdm(range(EPOCHS)):  # Limited epochs for optimization
        model.train()
        train_loss = 0
        count_train = 0
        for i in range(0, len(x_train_paths), batch_size):
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

            count_train += 1

        train_loss = train_loss / count_train  # Calculate average training loss

        # Validation phase
        model.eval()
        val_loss = 0
        all_y_true, all_y_pred = [], []
        with torch.no_grad():
            for i in range(0, len(x_validation_paths), batch_size):
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
        accuracy, recall, f1 = calculate_metrics(np.array(all_y_true), np.array(all_y_pred))

        print(f'\nEpoch {epoch} : Train Loss = {train_loss}, Validation Loss = {val_loss}, Accuracy = {accuracy}, Recall = {recall}, F1 = {f1}')

        # check for best model
        if val_loss < best_val_loss:

            # saving the model in case of an error
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            with open(BEST_PARAMS_PATH, "w") as f:
                f.write(str(trial.params))
            print(f"New best model saved with val_loss = {val_loss:.4f}")

            best_val_loss = val_loss
            best_val_f1 = f1  # Save the F1 from this epoch
            best_model_path = BEST_MODEL_PATH


        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    # Store your final best metrics in user_attrs
    trial.set_user_attr("best_val_loss", best_val_loss)
    trial.set_user_attr("best_val_f1", best_val_f1)
    trial.set_user_attr("best_model_path", best_model_path)

    return best_val_loss


def evaluate_on_test(model, test_csv, mean, std, batch_size):
    """
    Evaluate the model on the test set after tuning.
    """
    x_test_paths, X_test_features, test_labels = load_csv_data(test_csv)
    x_test_paths = [os.path.join(WAV2VEC_FOLDER, p) for p in x_test_paths]
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


def save_best_model(study, prefix="DeepFakeModel", extension="pth"):
    best_trial = study.best_trial
    best_model_path = best_trial.user_attrs["best_model_path"]
    best_val_loss = best_trial.user_attrs["best_val_loss"]
    best_params = best_trial.params

    # Construct a new filename
    model_filename = (
        f"{prefix}_"
        f"lr={best_params.get('learning_rate', 0.001)}_"
        f"bs={best_params.get('batch_size', 32)}_"
        f"drop={best_params.get('dropout', 0.5):.2f}_"
        f"layers={best_params.get('dense_layers', 3)}_"
        f"valloss={best_val_loss:.4f}.{extension}"
    )

    # Load the best model's state dict
    checkpoint = torch.load(best_model_path)

    # Save final checkpoint with hyperparams + state_dict
    torch.save({
        "state_dict": checkpoint["state_dict"],
        "hyperparams": best_params
    }, model_filename)

    print(f"Best model saved to {model_filename}")
    return model_filename


def load_best_model(model_class, save_path, device="cpu"):
    """
    :param model_class: The class definition for DeepFakeDetection or similar.
    :param save_path: Path to the saved .pth file.
    :param device: "cpu" or "cuda"
    :return: Instantiated model loaded with the best weights.
    """
    checkpoint = torch.load(save_path, map_location=device)
    best_model_path = checkpoint["state_dict"]
    best_params = checkpoint["hyperparams"]

    # Instantiate the model with the best hyperparameters:
    model = model_class(
        batch_size=best_params["batch_size"],
        learning_rate=best_params["learning_rate"],
        dense_layers=best_params["dense_layers"]  # or default
    ).to(device)

    # Load the saved state_dict
    model.load_state_dict(best_model_path)

    return model


def save_all_trials_csv(study, filename_prefix="optuna_results"):
    """
    Save the hyperparameters and metrics of each trial to a CSV file.
    """

    # Generate the timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{filename_prefix}_{timestamp}.csv"

    # Define the header with all the columns we want
    # Adapt column names for your hyperparameters (e.g., dropout vs. drop, etc.)
    header = [
        "trial_number",
        "learning_rate",
        "batch_size",
        "dropout",
        "dense_layers",
        "best_val_loss",
        "best_val_f1",
        "state"
    ]

    # Open the CSV file for writing
    with open(filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for trial in study.trials:  # iterate over all trials
            # If you only want completed trials, do:
            # if trial.state == optuna.trial.TrialState.COMPLETE:

            # Extract hyperparameters from trial.params
            lr = trial.params.get("learning_rate", None)
            bs = trial.params.get("batch_size", None)
            drop = trial.params.get("dropout", None)
            layers = trial.params.get("dense_layers", None)

            # Extract user_attrs from the objective
            val_loss = trial.user_attrs.get("best_val_loss", None)
            val_f1 = trial.user_attrs.get("best_val_f1", None)

            # Write a row to the CSV
            writer.writerow([
                trial.number,     # Unique trial index
                lr,
                bs,
                drop,
                layers,
                val_loss,
                val_f1,
                trial.state.name  # e.g., COMPLETE, PRUNED, FAIL, etc.
            ])

    print(f"All trial results have been saved to '{filename}'.")


def save_checkpoint_on_exit(signum, frame):
    print("Signal received. Saving study and exiting gracefully...")
    if study and study.best_trial:
        torch.save(study.best_trial.user_attrs["best_model_path"], BEST_MODEL_PATH)
        with open(BEST_PARAMS_PATH, "w") as f:
            f.write(str(study.best_trial.params))
    sys.exit(0)

# Run Optuna optimization
if __name__ == "__main__":
    # Directories and paths
    os.makedirs("checkpoints", exist_ok=True)
    BEST_MODEL_PATH = "checkpoints/best_model.pth"
    BEST_PARAMS_PATH = "checkpoints/best_params.json"
    STUDY_DB_PATH = "sqlite:///checkpoints/optuna_study.db"


    #setting up signals for error recovery
    signal.signal(signal.SIGINT, save_checkpoint_on_exit)
    signal.signal(signal.SIGTERM, save_checkpoint_on_exit)

    # run the optuna study
    study = optuna.create_study(storage=STUDY_DB_PATH,
                                study_name="speech_classification",
                                direction="minimize",
                                load_if_exists=True)
    study.optimize(objective, n_trials=TRIALS, show_progress_bar=True)

    #save the results
    save_all_trials_csv(study, filename_prefix="data/results/optuna_results")
    final_checkpoint_path = save_best_model(study)

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    # get the best model
    best_trial = study.best_trial
    best_val_loss, best_model_path = best_trial.user_attrs["best_val_loss"], best_trial.user_attrs["best_model_path"]

    # load the best model with the best parameters
    best_model = DeepFakeDetection(
        batch_size=best_params["batch_size"],
        learning_rate=best_params["learning_rate"],
        dense_layers=best_params["dense_layers"]
    ).to(DEVICE)

    # Load the state dictionary from the final checkpoint
    checkpoint = torch.load(final_checkpoint_path)
    best_model.load_state_dict(checkpoint["state_dict"])  # Load the weights


    # computing norm and std on the training data to use on the test data
    x_train_paths, X_train_features, train_labels = load_csv_data(TRAIN_CSV)
    x_train_paths = [os.path.join(WAV2VEC_FOLDER, p) for p in x_train_paths]
    mean = np.mean(X_train_features, axis=0)
    std = np.std(X_train_features, axis=0)

    # Evaluate on test data
    evaluate_on_test(best_model, TEST_CSV, mean, std, best_params["batch_size"])
