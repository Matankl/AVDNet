import csv
from datetime import datetime
import optuna
import os
import numpy as np
from data.Architectures.FullArchitecture import DeepFakeDetector
from data.Architectures.VGG16 import DeepFakeDetection
from data.Architectures.VGG16_FeaturesOnly import FeaturesOnly
from data_methods import create_tensors_from_csv, calculate_metrics, get_dataloader
from constants import *
from train_methods import train_model, save_model, load_model


# Early stopping implementation
class EarlyStopping:
    def __init__(self, patience=5, delta=0.000001, exp_threshold = 10000):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.high_threshold = exp_threshold

    def __call__(self, val_loss):
        if self.best_loss is None: # if the loss is not yet has been instantiated
            self.best_loss = val_loss

        if torch.isnan(val_loss) or torch.isinf(val_loss):  # if the loss exploded/have an issue
            self.early_stop = True

        elif val_loss >= self.high_threshold:
            self.early_stop = True


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
    best_trial_loss = float('inf')

    # Hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size",[8, 16, 32])
    dropout = trial.suggest_float("dropout", 0.1, 0.70)
    dense_layers = trial.suggest_int("dense_layers", 2, 7)  # total number of dense layers in classifier
    dense_initial_dim = trial.suggest_int("dense_initial_dim", 128, 2048, step=64)

    # Transformer fusion parameters
    transformer_layers = trial.suggest_int("transformer_layers", 1, 4)
    transformer_nhead = trial.suggest_int("transformer_nhead", 4, 8)
    head_dim = trial.suggest_int("head_dim", 16, 64, step=16)  # or choose an appropriate range
    d_model = head_dim * transformer_nhead

    # Pretrained module freezing parameters
    freeze_cnn_layers = trial.suggest_int("freeze_cnn_layers", 5, 15)
    freeze_encoder_layers = trial.suggest_int("freeze_encoder_layers", 0, 8)

    # Backbone selection: choose between 'vgg' and 'resnet'
    # backbone = trial.suggest_categorical("backbone", ["vgg", "resnet"])

    # Optimizer weight decay
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)

    # Print the current trial parameters
    print(f"Current trial parameters: {trial.params}")

    # Loading the data
    fraction_to_test = PARTIAL_TRAINING
    train_loader = get_dataloader("Train", DATASET_FOLDER, batch_size=batch_size, num_workers=2, fraction=fraction_to_test)
    val_loader = get_dataloader("Validation", DATASET_FOLDER, batch_size=batch_size, num_workers=2, fraction = fraction_to_test)

    # Build dense classifier hidden dimensions based on a linear decrease.
    # For instance, if dense_layers=3 and dense_initial_dim=512, you might have dimensions: [256, 128]
    dense_hidden_dims = []
    current_dim = dense_initial_dim
    for _ in range(dense_layers - 1):
        next_dim = current_dim // 2
        dense_hidden_dims.append(next_dim)
        current_dim = next_dim

    # Model initialization with tunable parameters.
    model = DeepFakeDetector(
        backbone="vgg",
        freeze_cnn=True,
        freeze_cnn_layers=freeze_cnn_layers,
        freeze_wav2vec=True,
        freeze_feature_extractor=True,
        freeze_encoder_layers=freeze_encoder_layers,
        d_model=d_model,
        nhead=transformer_nhead,
        num_layers=transformer_layers,
        dense_hidden_dims=dense_hidden_dims
    ).to(DEVICE)

    # Apply dynamic dropout to all dropout variants in the model.
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.p = dropout

    # Loss, optimizer, and early stopping
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = setup_optimizer(model, learning_rate, weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=PATIENCE)

    print("Starting to train:")
    # Train the model
    best_trial_loss, val_loss, f1 = train_model(
        best_trial_loss,
        criterion,
        early_stopping,
        model,
        optimizer,
        train_loader,
        trial,
        val_loader
    )

    # Store the best validation loss for the trial
    trial.set_user_attr("best_val_loss", best_trial_loss)

    return best_trial_loss, val_loss, f1


def setup_optimizer(model, learning_rate, weight_decay):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:  # Ignore frozen layers
            if "bn" in name or "bias" in name:  # Exclude BatchNorm & bias terms
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    # Define optimizer with separate parameter groups
    optimizer = torch.optim.Adam([
        {'params': decay_params, 'weight_decay': weight_decay},  # Apply weight decay
        {'params': no_decay_params, 'weight_decay': 0.0}  # No weight decay for BatchNorm & biases
    ], lr=learning_rate)

    return optimizer


def evaluate_on_test(model, test_csv, batch_size=None):
    """
    Evaluate the model on the test set after tuning.
    """

    # Create Test DataLoader
    if type(model) == DeepFakeDetection:
        test_loader = get_dataloader(test_csv, WAV2VEC_FOLDER, batch_size=batch_size, num_workers=2)
    elif type(model) == DeepFakeDetector:
        test_loader = get_dataloader("Validation", DATASET_FOLDER, batch_size=batch_size, num_workers=2)

    # Testing Loop with DataLoader
    model.eval()
    test_loss = 0
    all_y_true, all_y_pred = [], []
    criterion = torch.nn.BCELoss()

    with torch.no_grad():
        for x_paths_batch, x_features_batch, y_batch in test_loader:
            x_features_batch, y_batch = x_features_batch.to(DEVICE), y_batch.to(DEVICE)

            # Choose model type
            if isinstance(model, DeepFakeDetection):
                y_pred = model(x_paths_batch, x_features_batch).squeeze()
            elif isinstance(model, FeaturesOnly):
                y_pred = model(x_features_batch).squeeze()
            elif isinstance(model, DeepFakeDetector):
                y_pred = model(x_paths_batch, x_features_batch).squeeze()

            # Compute loss
            try:
                test_loss += criterion(y_pred, y_batch).item()
            except ValueError:
                y_pred = y_pred.view_as(y_batch)  # Reshape y_pred to match y_batch
                test_loss += criterion(y_pred, y_batch).item()

            # Store predictions
            all_y_true.extend(y_batch.cpu().numpy())
            all_y_pred.extend(y_pred.cpu().numpy())

    # Average test loss per batch
    test_loss /= len(test_loader)

    # Convert probabilities to binary predictions
    binary_y_pred = (np.array(all_y_pred) > 0.5).astype(int)

    # Compute metrics
    accuracy, recall, f1 = calculate_metrics(np.array(all_y_true), binary_y_pred)

    print(f"Test Loss = {test_loss:.4f}, Accuracy = {accuracy:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")
    return accuracy, recall, f1


def save_best_model(study, prefix="DeepFakeModel", extension="pth"):

    if study._is_multi_objective():
        best_trial = study.best_trials[0]
    else:
        best_trial = study.best_trial
    best_model_pth = best_trial.user_attrs["best_model_path"]
    best_val_loss = best_trial.user_attrs["best_val_loss"]
    params = best_trial.params

    saved_model = load_model(best_model_pth)

    # Construct a new filename
    model_filename = (
        f"{prefix}_"
        f"lr={params.get('learning_rate', 0.001):.5f}_"
        f"bs={params.get('batch_size', 32)}_"
        f"drop={params.get('dropout', 0.5):.2f}_"
        f"layers={params.get('dense_layers', 3)}_"
        f"valloss={best_val_loss:.4f}.{extension}"
    )

    # Save final checkpoint
    save_model(saved_model, model_filename)

    print(f"Best model saved to {model_filename}")
    return model_filename


def log_result(trial, filename="optuna_trials.csv"):
    """Logs all trial results into a CSV file for easy tracking."""

    # Get the trial results
    trial_dict = trial.params  # Hyperparameters
    trial_dict["trial_number"] = trial.number  # Trial number

    # Check if the trial is multi-objective
    if hasattr(trial, "values") and isinstance(trial.values, tuple):
        # Multi-objective: Store multiple objective values
        for i, value in enumerate(trial.values):
            trial_dict[f"value_{i}"] = value
    else:
        # Single-objective: Store a single value
        trial_dict["value"] = trial.value

            # Check if file exists to write headers
    file_exists = os.path.isfile(filename)

    # Append trial results to the CSV file
    with open(filename, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trial_dict.keys())

        # Write headers only if the file is new
        if not file_exists:
            writer.writeheader()

        # Write the trial data
        writer.writerow(trial_dict)


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
    if not os.path.exists("data/results"):
        os.mkdir("data/results")
    with open(filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for trial in study.trials:  # iterate over all trials

            log_result(trial)

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


def save_best_model_callback(study, trial):
    global best_model_path, best_validation_loss
    this_trial_loss = trial.user_attrs["best_val_loss"]
    this_trial_model_path = trial.user_attrs["best_model_path"]

    if this_trial_loss < best_validation_loss:
        best_validation_loss = this_trial_loss
        best_model_path = this_trial_model_path
        study.set_user_attr("best_model_path", this_trial_model_path)

        print(f"New best model (Trial {trial.number}) saved with val_loss = {best_validation_loss:.4f}")


# Run Optuna optimization
if __name__ == "__main__":
    best_model = None
    best_model_path = None
    best_validation_loss = 1000000

    # Directories and paths
    os.makedirs("checkpoints", exist_ok=True)
    BEST_MODEL_PATH = "checkpoints/best_model.pth"
    BEST_PARAMS_PATH = "checkpoints/best_params.json"
    STUDY_DB_PATH = "sqlite:///checkpoints/optuna_study.db"
    if not LOAD_TRAINING:
        STUDY_DB_PATH = None

    # run the optuna study
    study = optuna.create_study(storage=STUDY_DB_PATH,
                                study_name="speech_classification",
                                directions=["minimize", "minimize", "maximize"],
                                load_if_exists=LOAD_TRAINING)

    study.optimize(objective, n_trials=TRIALS, show_progress_bar=True, callbacks=[save_best_model_callback])

    #save the results
    save_all_trials_csv(study, filename_prefix="data/results/optuna_results")
    path_to_best_model = save_best_model(study)

    # Get the best hyperparameters
    # Print best trials (Pareto front) along with their hyperparameters
    print("\nBest Trials (Pareto front) with Hyperparameters:")
    for trial in study.best_trials:
        print(f"Trial {trial.number}:")
        print(f"  Best Loss       = {trial.values[0]:.6f}")
        print(f"  Last Epoch Loss = {trial.values[1]:.6f}")
        print(f"  F1-score        = {trial.values[2]:.6f}")
        print("  Hyperparameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print("-" * 50)  # Separator for better readability

    # best_params = study.best_params
    # print("Best hyperparameters:", best_params)

    # load the best model with the best parameters
    loaded_model = load_model(path_to_best_model)

    # Evaluate on test data
    evaluate_on_test(loaded_model, TEST_CSV)

