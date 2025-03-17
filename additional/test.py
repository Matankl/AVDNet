# Directories and paths
from tqdm import tqdm

from data_methods import get_dataloader, calculate_metrics
import numpy as np
from Architectures.FullArchitecture import DeepFakeDetector
from constants import *
from train_methods import load_model

best_model_path = r"/home/hp4ran/PycharmProjects/The-model/checkpoints/tmp_model_trial_0.pth"

# load the best model with the best parameters
model = load_model(best_model_path, DeepFakeDetector)

val_loader = get_dataloader("Test", DATASET_FOLDER, batch_size=8, num_workers=1)
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
