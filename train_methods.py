import importlib
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from data_methods import calculate_metrics
matplotlib.use('Agg')
from constants import *


def plot_loss(data, dir_path = ""):
    plt.figure(figsize=(10, 10))
    plt.plot(data['train_loss'], label='Train Loss')
    plt.plot(data['val_loss'], label='Val Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim(left=0)  # Ensure x-axis starts at 0
    plt.ylim(bottom=0)  # Ensure y-axis starts at 0
    plt.legend()
    plt.savefig(dir_path + '/loss_plot.jpeg')


def save_model(model, path):
    """This function saves the model as a .pth file
    and keep tracks of:
    1. the parameters of the model
    2. the hyperparameters of the model
    3. the class name of the model to easier later one loading
    """

    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': model.config,
        'model_class': model.__class__.__name__},
        path)

    return path


def load_model(save_path, model_class = None):
    """
    :param model_class: The class definition for DeepFakeDetection or similar (optional).
    :param save_path: Path to the saved .pth file.
    :return: Instantiated model loaded with the best weights.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(save_path, map_location=device)

    # Retrieve hyperparameters
    hyperparameters = checkpoint.get('hyperparameters', {})


    # Retrieve the saved class name
    model_class_name = checkpoint['model_class']
    if model_class is None:
        # Ensure the class is defined in the current environment
        if model_class_name in globals():
            model_class = globals()[model_class_name]  # Get the class reference
        else:
            try:
                module_name = f"data.Architectures.{model_class_name}"  # Replace with the correct module path
                module = importlib.import_module(module_name)
                model_class = getattr(module, model_class_name)
            except (ModuleNotFoundError, AttributeError):
                raise ValueError(f"⚠️ Class `{model_class_name}` not found in the current script or module.\n"
                                 f"👉 Ensure `{model_class_name}` is correctly defined in `{module_name}`.\n"
                                 f"👉 Alternatively, pass `model_class` explicitly to `load_model()`.")

    # Load the saved weights into the new model
    model = model_class(**hyperparameters)  # Instantiate the model
    model.load_state_dict(checkpoint['model_state_dict'])

    return model.to(device)


def train_one_epoch(model, train_loader, optimizer, criterion):
    """
    Performs one epoch of training. Returns the average training loss
    and a flag indicating if early termination is needed due to
    numerical instability (NaN/Inf in loss).
    """
    model.train()
    train_loss = 0.0
    count_train = 0
    exploding_batch_count = 0

    for input_1, input_2, y_batch in train_loader:
        input_1, input_2, y_batch = (
            input_1.to(DEVICE),
            input_2.to(DEVICE),
            y_batch.to(DEVICE)
        )
        optimizer.zero_grad()

        y_pred = model(input_1, input_2).squeeze()
        y_batch = y_batch.view(-1)  # ensure the shapes match
        y_pred = y_pred.squeeze(-1)  # handle extra dimension if present
        loss = criterion(y_pred, y_batch.float())

        # Check for numerical instability
        if torch.isnan(loss) or torch.isinf(loss):
            exploding_batch_count += 1
            del loss  # Free the loss tensor
            torch.cuda.empty_cache()  # Manually clear GPU cache

            if exploding_batch_count >= len(train_loader) * 0.1:
                print("Warning: NaN/Inf detected in loss. Skipping training.")
                return float('inf'), True  # Return infinite loss, signal early termination

            continue

        # Backpropagation step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        train_loss += loss.detach().item()
        count_train += 1

    # Avoid division by zero in case all batches got skipped
    avg_train_loss = train_loss / (count_train + 1e-10)
    return avg_train_loss, False


def validate_model(model, val_loader, criterion):
    """
    Performs validation on the given model and returns the validation
    loss and calculated metrics (accuracy, recall, f1).
    """
    model.eval()
    val_loss = 0.0
    all_y_true, all_y_pred = [], []

    with torch.no_grad():
        for input_1, input_2, y_batch in val_loader:
            input_1, input_2, y_batch = (
                input_1.to(DEVICE),
                input_2.to(DEVICE),
                y_batch.to(DEVICE)
            )
            y_pred = model(input_1, input_2).squeeze()

            batch_loss = criterion(y_pred.squeeze(), y_batch.float()).item()
            val_loss += batch_loss

            all_y_true.extend(y_batch.detach().cpu().numpy())
            all_y_pred.extend(y_pred.detach().squeeze().cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    accuracy, recall, f1 = calculate_metrics(np.array(all_y_true), np.array(all_y_pred))
    return avg_val_loss, accuracy, recall, f1


def train_model(best_trial_loss, criterion, early_stopping, model, optimizer, train_loader, trial, val_loader):
    """
    Main training method that loops over EPOCHS, calling the
    separate train and validation methods.
    """
    for epoch in tqdm(range(EPOCHS)):
        # --- TRAINING PHASE ---
        train_loss, early_termination = train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        # If we detect NaN/Inf too often, stop and return worst values
        if early_termination:
            return float('inf'), float('inf'), 0

        # --- VALIDATION PHASE ---
        val_loss, accuracy, recall, f1 = validate_model(model, val_loader, criterion)

        print(
            f"\nEpoch {epoch} : "
            f"Train Loss = {train_loss:.4f}, "
            f"Validation Loss = {val_loss:.4f}, "
            f"Accuracy = {accuracy:.4f}, "
            f"Recall = {recall:.4f}, "
            f"F1 = {f1:.4f}"
        )

        # Track the best validation loss
        if val_loss < best_trial_loss:
            best_trial_loss = val_loss
            temp_model_path = f"checkpoints/tmp_model_trial_{trial.number}.pth"
            trial.set_user_attr("best_model_path", temp_model_path)
            save_model(model, temp_model_path)

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            return best_trial_loss, val_loss, f1

    return best_trial_loss, val_loss, f1

