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


def train_model(best_trial_loss, criterion, early_stopping, model, optimizer, train_loader, trial, val_loader):
    for epoch in tqdm(range(EPOCHS)):  # Limited epochs for optimization
        model.train()
        train_loss = 0
        count_train = 0
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
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

            count_train += 1

        train_loss = train_loss / count_train  # Calculate average training loss

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
            torch.save({'model_state_dict': model.state_dict()}, temp_model_path)

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    return best_trial_loss, val_loss
