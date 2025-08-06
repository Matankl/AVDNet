import random
from constants import *
import csv
from datetime import datetime
import optuna
import os
import numpy as np
from additional.Archive.AVDNet import DeepFakeDetector
from Architectures.AVDNetV2 import AVDNet
from additional.Archive.VGG16 import DeepFakeDetection
from additional.Archive.VGG16_FeaturesOnly import FeaturesOnly
from data_methods import calculate_metrics, get_dataloader
from train_methods import train_model, save_model, load_model
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
    # learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-4, log=True)
    learning_rate = 1.334e-6
    learning_rate = trial.suggest_float("learning_rate", learning_rate*0.95, learning_rate*1.05, log=True)
    # batch_size = trial.suggest_categorical("batch_size",[8, 16])
    batch_size = 16
    # dropout = trial.suggest_float("dropout", 0.1, 0.90)
    dropout = 0.8
    dropout = trial.suggest_float("dropout", dropout*0.95, dropout*1.05)
    # weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    weight_decay = 3.797e-06
    weight_decay = trial.suggest_float("weight_decay", weight_decay*0.95, weight_decay*1.05, log=True)
    # aug_prob = trial.suggest_float("aug_prob", 0.1, 0.6)
    aug_prob = DEFAULT_AUG_PROB


    # this is the model parameters of the best model with data aug and partial info

    # Transformer fusion parameters
    # dense_layers = trial.suggest_int("dense_layers", 2, 5)  # total number of dense layers in classifier
    dense_layers = 4
    # dense_initial_dim = trial.suggest_int("dense_initial_dim", 256, 1600, step=64)
    dense_initial_dim = 384
    # transformer_layers = 4 #trial.suggest_int("transformer_layers", 1, 4)
    transformer_layers = 4
    # transformer_nhead = trial.suggest_int("transformer_nhead", 8, 24)
    transformer_nhead = 24
    # head_dim = trial.suggest_int("head_dim", 80, 172, step=16)  # or choose an appropriate range
    head_dim = 128
    d_model = head_dim * transformer_nhead

    # Pretrained module freezing parameters
    # freeze_cnn_layers = trial.suggest_int("freeze_cnn_layers", 5, 13)
    freeze_cnn_layers = 7
    # freeze_encoder_layers = trial.suggest_int("freeze_encoder_layers", 6, 24)
    freeze_encoder_layers = 7

    # Print the current trial parameters
    print(f"Trial {trial.number} with parameters: {trial.params}")

    # Loading the data
    train_loader = get_dataloader("Train", DATASET_FOLDER, batch_size=batch_size, num_workers=8,
                                  fraction=PARTIAL_TRAINING, data_aug=aug_prob)
    val_loader = get_dataloader("Validation", DATASET_FOLDER, batch_size=batch_size, num_workers=8,
                                fraction=PARTIAL_TESTING)


    # Build dense classifier hidden dimensions based on a linear decrease.
    # For instance, if dense_layers=3 and dense_initial_dim=512, you might have dimensions: [256, 128]
    dense_hidden_dims = []
    current_dim = dense_initial_dim
    for _ in range(dense_layers - 1):
        next_dim = current_dim // 2
        dense_hidden_dims.append(next_dim)
        current_dim = next_dim

    # Model initialization with tunable parameters.
    model = AVDNet(
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

def evaluate_on_test(model, test_csv, batch_size=None):
    """
    Evaluate the model on the test set after tuning.
    """


    if type(model) == DeepFakeDetector:
        test_loader = get_dataloader("Validation", CUSTOM_DATASET_FOLDER, batch_size=batch_size, num_workers=2)
    else: raise ValueError("Model not yet supported")

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

    # Open the CSV file for writing
    # if not os.path.exists("data/results"):
    #     os.mkdir("data/results")
    for trial in study.trials:  # iterate over all trial
        log_result(trial, filename=f"Final Models/study_results.csv")
    # with open(filename, mode="w", newline="") as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow(header)
    #
    #     for trial in study.trials:  # iterate over all trials
    #
    #         log_result(trial, filename=f"study_results.csv")
    #
    #         # If you only want completed trials, do:
    #         # if trial.state == optuna.trial.TrialState.COMPLETE:
    #
    #         # Extract hyperparameters from trial.params
    #         lr = trial.params.get("learning_rate", None)
    #         bs = trial.params.get("batch_size", None)
    #         drop = trial.params.get("dropout", None)
    #         layers = trial.params.get("dense_layers", None)
    #
    #         # Extract user_attrs from the objective
    #         val_loss = trial.user_attrs.get("best_val_loss", None)
    #         val_f1 = trial.user_attrs.get("best_val_f1", None)
    #
    #         # Write a row to the CSV
    #         writer.writerow([
    #             trial.number,     # Unique trial index
    #             lr,
    #             bs,
    #             drop,
    #             layers,
    #             val_loss,
    #             val_f1,
    #             trial.state.name  # e.g., COMPLETE, PRUNED, FAIL, etc.
    #         ])

    # print(f"All trial results have been saved to '{filename}'.")

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
    best_model = None
    best_model_path = None
    best_validation_loss = 1000000

    # Directories and paths
    os.makedirs("checkpoints", exist_ok=True)
    BEST_MODEL_PATH = "checkpoints/best_model.pth"
    BEST_PARAMS_PATH = "checkpoints/best_params.json"
    # STUDY_DB_PATH = "sqlite:///checkpoints/optuna_study.db"
    # STUDY_DB_PATH = "sqlite:///checkpoints/Wav2Vec_ResNet.db"
    # STUDY_DB_PATH = "sqlite:///checkpoints/Wav2Vec_ResNet34.db"
    # STUDY_DB_PATH = "sqlite:///checkpoints/Wav2Vec_VGG300M.db"
    # STUDY_DB_PATH = "sqlite:///checkpoints/Wav2Vec_VGG.db"
    # STUDY_DB_PATH = "sqlite:///checkpoints/Wav2Vec_VGG_spatial_info.db"
    # STUDY_DB_PATH = "sqlite:///checkpoints/Wav2Vec_VGG_spatial_data_aug.db"
    # STUDY_DB_PATH = "sqlite:///checkpoints/AVDNET.db"
    STUDY_DB_PATH = "sqlite:///checkpoints/AVDNET_architecture_from_custom_best.db"
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