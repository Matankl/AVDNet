# Directories and paths
import os

from tqdm import tqdm

from Architectures.AVDNetV2 import AVDNet
from data_methods import get_dataloader, calculate_metrics
import numpy as np
from Architectures.AVDNet import DeepFakeDetector
from constants import *
from train_methods import load_model

def evaluate_on_test(model_path):
    best_model_path = model_path

    # load the best model with the best parameters
    model = load_model(best_model_path, AVDNet)
    # model = torch.load(best_model_path, weights_only=False)

    val_loader = get_dataloader("Test", DATASET_FOLDER, batch_size=4, num_workers=1)
    criterion = torch.nn.BCEWithLogitsLoss()


    # Validation phase
    model.eval()
    val_loss = 0
    all_y_true, all_y_pred = [], []
    with torch.no_grad():
        for input_1, input_2, y_batch in tqdm(val_loader):
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
    print(f'Test Loss = {val_loss}, Accuracy = {accuracy}, Recall = {recall}, F1 = {f1}')


for i in range(10):
    model_path = rf"/home/hp4ran/PycharmProjects/The-model/checkpoints/tmp_model_trial_{i}.pth"
    if os.path.exists(model_path):
        print(f"Results from {model_path}:")
        evaluate_on_test(model_path)
        continue

    break