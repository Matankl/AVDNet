import numpy as np
import os
import pandas as pd
from datetime import datetime
import matplotlib
from sklearn.metrics import accuracy_score, recall_score, f1_score
from VGGM_16_custom import DeepFakeDetection
import matplotlib.pyplot as plt
import openpyxl
from constants import *
from tqdm import tqdm
import torch
matplotlib.use('Agg')

# Function to load data paths and labels from a CSV file
def load_csv_data(csv_path):
    data = pd.read_csv(csv_path)
    data = data.sample(frac=1).reset_index(drop=True)
    x_paths = data.iloc[:, 1].values  # Extract the paths to wav2vec matrices
    labels = data['label'].values.astype(int)  # Extract and cast labels to integers
    Xfeatures = data.iloc[:, 2:-1].values  # Corrected slicing for Xfeatures
    return x_paths, Xfeatures, labels

# Function to create tensors for training/validation batches from CSV data
def create_tensors_from_csv(x_paths, Xfeatures, labels, start_idx, block_num, target_shape=None):
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
    x_wav2vec, x_vectors, y = [], [], []
    for i in range(start_idx, min(start_idx + block_num, len(x_paths))):

        # Load wav2vec matrix for a sample
        wav2vec_matrix = np.load(x_paths[i], allow_pickle=True)

        # Convert the matrix to a tensor and add channel dimension
        wav2vec_matrix = wav2vec_matrix.clone().detach()
        x_wav2vec.append(torch.tensor(wav2vec_matrix))  # Directly append tensor (not wrapped in a list)

        tensor_vector = torch.tensor(Xfeatures[i], dtype=torch.float).detach()
        x_vectors.append(tensor_vector)

        y.append(labels[i])

    # Stack tensors into a single batch tensor
    x_wav2vec = torch.stack(x_wav2vec)  # Shape: (batch_size, T, D) or (batch_size, channels, T, D)
    x_vectors = torch.stack(x_vectors)
    y = torch.tensor(y, dtype=torch.float)  # Convert labels to tensor
    if(DEBUGMODE):
        print(x_wav2vec.shape)
        print(y.shape)
    return x_wav2vec, x_vectors, y

# Function to calculate evaluation metrics
def calculate_metrics(y_true, y_pred):
    y_pred_labels = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
    acc = accuracy_score(y_true, y_pred_labels)  # Calculate accuracy
    recall = recall_score(y_true, y_pred_labels)  # Calculate recall
    f1 = f1_score(y_true, y_pred_labels)  # Calculate F1-score
    return acc, recall, f1

# Load training data
csv_file = INPUT_CSV
x_paths, Xfeatures, labels = load_csv_data(csv_file)

print('Start training:')

# Set the device to GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the custom model and move it to the selected device
model = DeepFakeDetection(EPOCHS, batch_size = BATCH_SIZE, learning_rate= 0.0001).to(DEVICE)

# Setting model parameters
learning_rate = model.get_learning_rate()  # Get learning rate from the model
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Initialize Adam optimizer
criterion = torch.nn.BCELoss()  # Binary cross-entropy loss function
epoch, batch_size = model.get_epochs(), model.get_batch_size()  # Get number of epochs and batch size from the model

# Dataframe to store training results for each epoch
results_df = pd.DataFrame([], columns=['train_loss', 'val_loss', 'accuracy', 'recall', 'f1_score'])

# Training loop
for Epoch in range(epoch):
    print("Epoch :", Epoch)
    model.train()  # Set model to training mode
    train_loss = 0
    count_train = 0
    val_loss = 0

    # Iterating over training data in batches
    for i in tqdm(range(0, len(x_paths), batch_size)):
        x_wav2vec_batch, x_features_batch, y_batch = create_tensors_from_csv([os.path.join(WAV2VEC_FOLDER, p) for p in x_paths], Xfeatures, labels, i, batch_size)  # Create batch tensors
        x_wav2vec_batch, x_features_batch, y_batch = x_wav2vec_batch.detach().to(DEVICE), x_features_batch.detach().to(DEVICE), y_batch.detach().to(DEVICE)  # Move tensors to the device

        # if x_wav2vec_batch.size(0) != batch_size:
        #     print("smaller batch", x_wav2vec_batch.size(0))
        #     continue

        optimizer.zero_grad()  # Zero out gradients from the previous step
        y_pred = model(x_wav2vec_batch, x_features_batch)  # Forward pass
        # Reshape target to match predictions
        y_batch = y_batch.view(-1)  # Ensure y_batch is 1D
        y_pred = y_pred.squeeze(-1)  # Ensure y_pred is also 1D
        # Calculate loss
        loss = criterion(y_pred, y_batch.float())
        loss.backward()
        optimizer.step()  # Update model parameters

        count_train += 1

        train_loss += loss.item()  # Accumulate training loss

    train_loss = train_loss / count_train  # Calculate average training loss

    # Validation phase
    with torch.no_grad():
        model.eval()  # Set model to evaluation mode
        all_y_true, all_y_pred = [], []  # Lists to store true and predicted labels

        for i in tqdm(range(0, len(x_paths), batch_size)):
            x_wav2vec_batch, x_features_batch, y_batch = create_tensors_from_csv([os.path.join(WAV2VEC_FOLDER, p) for p in x_paths], Xfeatures, labels, i, batch_size)  # Create batch tensors
            x_wav2vec_batch, x_features_batch, y_batch = x_wav2vec_batch.detach().to(DEVICE), x_features_batch.detach().to(DEVICE), y_batch.detach().to(DEVICE)  # Move tensors to the device

            # if x_wav2vec_batch.size(0) != batch_size:
            #     print("smaller batch", x_wav2vec_batch.size(0))
            #     continue

            y_pred = model(x_wav2vec_batch, x_features_batch).squeeze()  # Forward pass
            val_loss += criterion(y_pred.squeeze(), y_batch.float()).item()  # Accumulate validation loss
            all_y_true.extend(y_batch.cpu().numpy())  # Collect true labels
            all_y_pred.extend(y_pred.squeeze().cpu())  # Collect predicted probabilities

    val_loss = val_loss / count_train  # Calculate average validation loss
    accuracy, recall, f1 = calculate_metrics(np.array(all_y_true), np.array(all_y_pred))  # Compute metrics

    # Log results of the current epoch
    results_df.loc[len(results_df)] = [train_loss, val_loss, accuracy, recall, f1]

    print(f'Epochs {Epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}, Accuracy = {accuracy}, Recall = {recall}, F1 = {f1}')

# Preparing results folder
now_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")  # Timestamp for unique result folder
dir_path = TRAINING_DATA_PATH + now_time
#creating the directories if needed
os.makedirs(dir_path, exist_ok=True)  # Create directory if it doesn't exist
os.makedirs(TRAINING_DATA_PATH, exist_ok=True)  # Create directory if it doesn't exist

# Save results to an Excel file
results_df.to_excel(dir_path + '/final_report.xlsx')

def plot_loss(data):
    plt.figure(figsize=(10, 10))
    plt.plot(data['train_loss'], label='Train Loss')
    plt.plot(data['val_loss'], label='Val Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim(left=0)  # Ensure x-axis starts at 0
    plt.ylim(bottom=0)  # Ensure y-axis starts at 0
    plt.legend()
    plt.savefig(dir_path + '/loss_plot.jpeg')

# Plot loss curves and save as an image
plot_loss(results_df)