import os
from datetime import datetime
import matplotlib
from VGGM_16_custom import DeepFakeDetection
import matplotlib.pyplot as plt
import openpyxl
from tqdm import tqdm
matplotlib.use('Agg')
from constants import *
from data_methods import *

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