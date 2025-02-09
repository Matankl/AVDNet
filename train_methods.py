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

def train_model(best_trial_loss, criterion, early_stopping, model, optimizer, train_loader, trial, val_loader):
    for epoch in tqdm(range(EPOCHS)):  # Limited epochs for optimization
        model.train()
        train_loss = -1
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
            y_batch = y_batch.view(-1)
            y_pred = y_pred.squeeze(-1)
            loss = criterion(y_pred, y_batch.float())

            # skipping a specific batch if numerical instability
            if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
                exploding_batch_count += 1
                if exploding_batch_count >= len(train_loader) * 0.1:
                    print("Warning: NaN/Inf detected in loss. Skipping training.")
                    return float('inf'), float('inf'), 0  # Return worst values

                continue  # Skip this batch

            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

            count_train += 1

        train_loss = train_loss / (count_train + 1e-10)  # Calculate average training loss

        # Validation phase
        model.eval()
        val_loss = 0
        all_y_true, all_y_pred = [], []
        with torch.no_grad():
            for input_1, input_2, y_batch in val_loader:
                input_1, input_2, y_batch = (
                    input_1.to(DEVICE),
                    input_2.to(DEVICE),
                    y_batch.to(DEVICE)
                )
                y_pred = model(input_1, input_2).squeeze()
                val_loss += criterion(y_pred.squeeze(), y_batch.float()).item()
                all_y_true.extend(y_batch.cpu().numpy())
                all_y_pred.extend(y_pred.squeeze().cpu())

        # Compute validation metrics
        val_loss /= len(val_loader)
        accuracy, recall, f1 = calculate_metrics(np.array(all_y_true), np.array(all_y_pred))
        print(
            f'\nEpoch {epoch} : Train Loss = {train_loss}, Validation Loss = {val_loss}, Accuracy = {accuracy}, Recall = {recall}, F1 = {f1}')

        if val_loss < best_trial_loss:
            best_trial_loss = val_loss
            # Copy current model’s state_dict
            temp_model_path = f"checkpoints/tmp_model_trial_{trial.number}.pth"
            trial.set_user_attr("best_model_path", temp_model_path)
            save_model(model, temp_model_path)


            # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            return best_trial_loss, val_loss, f1

    return best_trial_loss, val_loss, f1
