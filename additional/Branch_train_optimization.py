import random

from Architectures.branches import Wav2Vec2DeepfakeClassifier
from constants import *
import csv
from datetime import datetime
import optuna
import os
import numpy as np

from data_methods import get_dataloader
from train_methods import train_model
import math


# Early stopping implementation
class EarlyStopping:
    def __init__(self, patience=5, delta=0.001, exp_threshold=10000):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.prev_loss = None
        self.counter = 0
        self.prev_counter = 0
        self.early_stop = False
        self.high_threshold = exp_threshold

    def __call__(self, val_loss):
        if self.best_loss is None: # if the loss is not yet has been instantiated
            self.best_loss = val_loss
            self.prev_loss = val_loss

        if math.isnan(val_loss) or math.isinf(val_loss):  # if the loss exploded/have an issue
            self.early_stop = True

        elif val_loss >= self.high_threshold:
            self.early_stop = True

        if self.prev_loss <= val_loss:
            self.prev_counter +=1
        else:
            self.prev_counter = 0


        if val_loss > self.best_loss * (1 - self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

            if self.prev_counter >= self.patience - 2:
                self.early_stop = True

        else:
            self.best_loss = val_loss
            self.counter = 0


def objective(trial):
    """
    Optuna objective function for hyperparameter tuning using training and validation sets.
    """
    best_trial_loss = float('inf')
    set_global_seed(SEED + trial.number) # to have different reproducible trials but not identical.
    # Hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    batch_size = 32
    dropout = trial.suggest_float("dropout", 0.1, 0.90)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)



    # Print the current trial parameters
    print(f"Trial {trial.number} with parameters: {trial.params}")

    # Loading the data
    train_loader = get_dataloader("Train", DATASET_FOLDER, batch_size=batch_size, num_workers=8,
                                  fraction=PARTIAL_TRAINING, data_aug=0)
    val_loader = get_dataloader("Validation", DATASET_FOLDER, batch_size=batch_size, num_workers=8,
                                fraction=PARTIAL_TESTING)

    # ---- Model ----
    model = Wav2Vec2DeepfakeClassifier().to(DEVICE).to(DEVICE)


    # Loss, optimizer, and early stopping
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = setup_optimizer(model, learning_rate, weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=min(50, EPOCHS), eta_min=1e-8)
    early_stopping = EarlyStopping(patience=PATIENCE)

    # Train the model
    best_trial_loss, val_loss, f1 = train_model(
        best_trial_loss,
        criterion,
        early_stopping,
        model,
        optimizer,
        scheduler,
        train_loader,
        trial,
        val_loader,
        r"run_path.txt"
    )

    # Store the best validation loss for the trial
    trial.set_user_attr("best_val_loss", best_trial_loss)

    return best_trial_loss, f1

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


def log_result(trial, filename="optuna_trials.csv"):
    """Logs all trial results into a CSV file for easy tracking."""

    value_dict = {}
    # Check if the trial is multi-objective
    if hasattr(trial, "values") and trial.values is not None:
        # Multi-objective: Store multiple objective values
        for i, val in enumerate(trial.values):
            value_dict[f"value_{i}"] = val
    elif hasattr(trial, "value") and  trial.value is not None:
        # Single-objective: Store a single value
        value_dict["value"] = trial.value

    else:
        return

    # Merge dictionaries, ensuring order: trial_number -> values -> hyperparams
    ordered_trial_dict = {
        "trial_number": trial.number,  # First column
        **value_dict,  # Multi-objective values (value_0, value_1, ...)
        **trial.params  # Hyperparameters (remaining values)
    }


    # Check if file exists to write headers
    file_exists = os.path.isfile(filename)

    # Append trial results to the CSV file
    with open(filename, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_trial_dict.keys())

        # Write headers only if the file is new
        if not file_exists:
            writer.writeheader()

        # Write the trial data
        writer.writerow(ordered_trial_dict)

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

    for trial in study.trials:  # iterate over all trial
        log_result(trial, filename=f"Final Models/study_results.csv")


def save_best_model_callback(study, trial):
    global best_model_path, best_validation_loss
    this_trial_loss = trial.user_attrs["best_val_loss"]
    this_trial_model_path = trial.user_attrs["best_model_path"]

    if this_trial_loss < best_validation_loss:
        best_validation_loss = this_trial_loss
        best_model_path = this_trial_model_path
        study.set_user_attr("best_model_path", this_trial_model_path)

        print(f"New best model (Trial {trial.number}) saved with val_loss = {best_validation_loss:.4f}")

def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Run Optuna optimization
if __name__ == "__main__":
    best_validation_loss = 1000000

    # Directories and paths
    os.makedirs("wav2vec_branch_checkpoints", exist_ok=True)

    STUDY_DB_PATH = "sqlite:///checkpoints/wav2vec_branch.db"
    LOAD_TRAINING = False
    if not LOAD_TRAINING:
        STUDY_DB_PATH = None

    #set random seed
    set_global_seed(SEED)

    # run the optuna study
    study = optuna.create_study(storage=STUDY_DB_PATH,
                                study_name="model training 1 with fixed loss",
                                directions=["minimize", "maximize"],
                                load_if_exists=LOAD_TRAINING)

    if hasattr(study, "num_trials"):
        nb_trials = TRIALS - study.num_trials if TRIALS > study.num_trials else 0
    else:
        nb_trials = TRIALS

    study.optimize(objective, n_trials=nb_trials, show_progress_bar=True, callbacks=[save_best_model_callback])

    #save the results
    save_all_trials_csv(study, filename_prefix="data/results/optuna_results")
    print("csv saved...")