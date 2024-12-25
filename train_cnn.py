from VGGM_16_custom import DeepFakeDetection
import numpy as np
import os
import pandas as pd
from datetime import datetime
import matplotlib
from sklearn.metrics import accuracy_score, recall_score, f1_score
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import openpyxl
from constants import *
from tqdm import tqdm
# Function to create tensors for training/validation batches from CSV data
import torch


# Function to load data paths and labels from a CSV file
def load_csv_data(csv_path):
    data = pd.read_csv(csv_path)
    x_paths = data.iloc[:, 1].values  # Extract the paths to wav2vec matrices
    labels = data['label'].values.astype(int)  # Extract and cast labels to integers
    return x_paths, labels



def create_tensors_from_csv(x_paths, labels, start_idx, block_num, target_shape=None):
    """
    Creates tensors from wav2vec matrices and labels.

    Parameters:
    - x_paths (list): List of paths to .npy files containing wav2vec matrices.
    - labels (list): Corresponding labels for the wav2vec matrices.
    - start_idx (int): Starting index in the dataset.
    - block_num (int): Number of samples to process in one block.
    - target_shape (tuple): Desired shape for the wav2vec matrices (e.g., (T, D)).

    Returns:
    - x (torch.Tensor): Tensor of wav2vec matrices (with added channel dimension).
    - y (torch.Tensor): Tensor of labels.
    """
    x, y = [], []

    for i in range(start_idx, min(start_idx + block_num, len(x_paths))):
        # Load wav2vec matrix for a sample
        wav2vec_matrix = np.load(x_paths[i], allow_pickle=True)

        # Convert the matrix to a tensor and add channel dimension
        wav2vec_matrix = wav2vec_matrix.clone().detach().to(DEVICE)
        x.append(wav2vec_matrix)  # Directly append tensor (not wrapped in a list)
        y.append(labels[i])

    # Stack tensors into a single batch tensor
    x = torch.stack(x)  # Shape: (batch_size, T, D) or (batch_size, channels, T, D)
    y = torch.tensor(y, dtype=torch.float)  # Convert labels to tensor
    if(DEBUGMODE):
        print(x.shape)
        print(y.shape)
    return x, y


# Function to calculate evaluation metrics
def calculate_metrics(y_true, y_pred):
    y_pred_labels = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
    acc = accuracy_score(y_true, y_pred_labels)  # Calculate accuracy
    recall = recall_score(y_true, y_pred_labels)  # Calculate recall
    f1 = f1_score(y_true, y_pred_labels)  # Calculate F1-score
    return acc, recall, f1

# Paths for data and results
data_path = 'data/wav2vac2.csv'  # Path where the data CSV is stored
training_results_path = 'data/results/'  # Directory for saving training results

# Load training data
csv_file = INPUT_CSV
x_paths, labels = load_csv_data(csv_file)

# Setting the model and his parameters
model = DeepFakeDetection(EPOCHS, BATCH_SIZE, LEARNING_RATE).to(DEVICE)
learning_rate = model.get_learning_rate()  # Get learning rate from the model
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Initialize Adam optimizer
criterion = torch.nn.BCELoss()  # Binary cross-entropy loss function
epoch, batch_size = model.get_epochs(), model.get_batch_size()  # Get number of epochs and batch size from the model

print('Start training:')
print('Device:', DEVICE)
print("Optimizer: ", optimizer)


# Dataframe to store training results for each epoch
results_df = pd.DataFrame([], columns=['train_loss', 'val_loss', 'accuracy', 'recall', 'f1_score'])

# Training loop
for Epoch in range(epoch):
    print("Epoch :", Epoch)
    model.train()  # Set model to training mode
    train_loss = 0
    count_train = 0

    # Iterating over training data in batches
    for i in tqdm(range(0, len(x_paths), batch_size)):
        x_batch, y_batch = create_tensors_from_csv([os.path.join(WAV2VEC_FOLDER, p) for p in x_paths], labels, i, batch_size)  # Create batch tensors
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)  # Move tensors to the device

        if x_batch.size(0) != batch_size:
            print("smaller batch", x_batch.size(0))
            continue

        optimizer.zero_grad()  # Zero out gradients from the previous step
        y_pred = model(x_batch).squeeze()  # Forward pass
        y_batch = y_batch.view(-1)
        loss = criterion(y_pred, y_batch.float())  # Calculate loss
        train_loss += loss.item()  # Accumulate training loss
        loss.backward()  # Backward pass (gradient computation)
        optimizer.step()  # Update model parameters
        count_train += 1

    train_loss = train_loss / count_train  # Calculate average training loss

    # Validation phase
    with torch.no_grad():
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        all_y_true, all_y_pred = [], []  # Lists to store true and predicted labels

        for i in tqdm(range(0, len(x_paths), batch_size)):
            x_batch, y_batch = create_tensors_from_csv([os.path.join(WAV2VEC_FOLDER, p) for p in x_paths], labels, i, batch_size)  # Create batch tensors
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)  # Move tensors to the device

            if x_batch.size(0) != batch_size:
                print("smaller batch", x_batch.size(0))
                continue

            y_pred = model(x_batch)  # Forward pass
            val_loss += criterion(y_pred.squeeze(), y_batch.float()).item()  # Accumulate validation loss
            all_y_true.extend(y_batch.cpu().numpy())  # Collect true labels
            all_y_pred.extend(y_pred.squeeze().cpu().numpy())  # Collect predicted probabilities

        val_loss = val_loss / count_train  # Calculate average validation loss
        accuracy, recall, f1 = calculate_metrics(np.array(all_y_true), np.array(all_y_pred))  # Compute metrics

    # Log results of the current epoch
    results_df.loc[len(results_df)] = [train_loss, val_loss, accuracy, recall, f1]


if DEBUGMODE:
    print(f'EpochS {EPOCHS}: Train Loss = {train_loss}, Val Loss = {val_loss}, Accuracy = {accuracy}, Recall = {recall}, F1 = {f1}')

# Preparing results folder
now_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")  # Timestamp for unique result folder
dir_path = training_results_path + now_time

#creating the directories if needed
os.makedirs(dir_path, exist_ok=True)  # Create directory if it doesn't exist
os.makedirs(training_results_path, exist_ok=True)  # Create directory if it doesn't exist

# Save results to an Excel file
results_df.to_excel(dir_path + '/final_report.xlsx')

# Plot loss curves and save as an image
plt.figure(figsize=(10, 10))
plt.plot(results_df['train_loss'], label='Train Loss')
plt.plot(results_df['val_loss'], label='Val Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim(left=0)  # Ensure x-axis starts at 0
plt.ylim(bottom=0)  # Ensure y-axis starts at 0
plt.legend()
plt.savefig(dir_path + '/loss_plot.jpeg')