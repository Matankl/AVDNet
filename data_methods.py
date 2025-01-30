import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score
from constants import *
from torch.utils.data import Dataset, DataLoader



# Define Dataset for Training & Validation
class Wav2VecDataset(Dataset):
    def __init__(self, csv_path, wav2vec_folder):
        self.data = pd.read_csv(csv_path)
        self.wav2vec_folder = wav2vec_folder

        # Extract paths, features, and labels
        self.x_paths = self.data.iloc[:, 1].values  # wav2vec2 matrix paths
        self.Xfeatures = self.data.iloc[:, 2:-1].values.astype(np.float32)  # Numeric features
        self.labels = self.data['label'].values.astype(int)  # Labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_path = os.path.join(self.wav2vec_folder, self.x_paths[idx])

        # Load wav2vec matrix
        wav2vec_matrix = np.load(x_path, allow_pickle=True)
        wav2vec_tensor = wav2vec_matrix.clone().detach()

        # Load additional features
        x_features = torch.tensor(self.Xfeatures[idx], dtype=torch.float32)

        # Load label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # BCELoss needs float labels

        return wav2vec_tensor, x_features, label

# Function to create DataLoader
def get_dataloader(csv_path, wav2vec_folder, pin_memory=False, batch_size=32, shuffle=True, num_workers=4):
    dataset = Wav2VecDataset(csv_path, wav2vec_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return dataloader

# Function to create tensors for training/validation batches from CSV data
def create_tensors_from_csv(x_paths, Xfeatures, labels, start_idx, block_num, target_shape=None):
    """
    Creates tensors from wav2vec2 matrices and labels.

    Parameters:
    - x_paths (list): List of paths to .npy files containing wav2vec2 matrices.
    - labels (list): Corresponding labels for the wav2vec2 matrices.
    - start_idx (int): Starting index in the dataset.
    - block_num (int): Number of samples to process in one block.
    - target_shape (tuple): Desired shape for the wav2vec2 matrices (e.g., (T, D)).

    Returns:
    - x (torch.Tensor): Tensor of wav2vec2 matrices (with added channel dimension).
    - y (torch.Tensor): Tensor of labels.
    """
    x_wav2vec, x_vectors, y = [], [], []
    for i in range(start_idx, min(start_idx + block_num, len(x_paths))):

        # Load wav2vec matrix for a sample
        wav2vec_matrix = np.load(x_paths[i], allow_pickle=True)

        # Convert the matrix to a tensor and add channel dimension
        wav2vec_matrix = wav2vec_matrix.clone().detach()
        x_wav2vec.append(wav2vec_matrix)  # Directly append tensor (not wrapped in a list)

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
