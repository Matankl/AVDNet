# Directories and paths
from optimization import evaluate_on_test
import os
import numpy as np
from tqdm import tqdm
from VGG_16_custom import DeepFakeDetection
from data_methods import create_tensors_from_csv, calculate_metrics, load_csv_data
from constants import *


best_model_path = r"C:\Users\parde\PycharmProjects\The-model\DeepFakeModel_lr=0.00052_bs=32_drop=0.47_layers=3_valloss=0.0736.pth"

# load the best model with the best parameters
best_model = DeepFakeDetection(
    batch_size=32,
    learning_rate=32,
    dense_layers=3
).to(DEVICE)

# Load the state dictionary from the final checkpoint
checkpoint = torch.load(best_model_path)
best_model.load_state_dict(checkpoint["state_dict"])  # Load the weights


# computing norm and std on the training data to use on the test data
x_train_paths, X_train_features, train_labels = load_csv_data(r"C:\Users\parde\PycharmProjects\The-model\additional\train_30h.csv")
x_train_paths = [os.path.join(WAV2VEC_FOLDER, p) for p in x_train_paths]
mean = np.mean(X_train_features, axis=0)
std = np.std(X_train_features, axis=0)

# Evaluate on test data
evaluate_on_test(best_model, r"C:\Users\parde\PycharmProjects\The-model\additional\test_30h.csv", mean, std, 32)