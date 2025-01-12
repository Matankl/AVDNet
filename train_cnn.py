import os
from datetime import datetime
import matplotlib
from VGG_16_custom import DeepFakeDetection
import matplotlib.pyplot as plt
import openpyxl
from tqdm import tqdm
matplotlib.use('Agg')
# import onnx # to save the model
# folders import
from data_methods import *
from constants import *

# Load training data
x_paths, Xfeatures, labels = load_csv_data(TRAIN_CSV)
x_paths = [os.path.join(WAV2VEC_FOLDER, p) for p in x_paths]
x_test_paths, X_test_features, test_labels = load_csv_data(TEST_CSV)
x_test_paths = [os.path.join(WAV2VEC_FOLDER, p) for p in x_test_paths]
x_validation_paths, X_validation_features, validation_labels = load_csv_data(VALIDATION_CSV)
x_validation_paths = [os.path.join(WAV2VEC_FOLDER, p) for p in x_validation_paths]

# Compute the mean and standard deviation from the training data
mean = np.mean(Xfeatures, axis=0)
std = np.std(Xfeatures, axis=0)

# Normalize the training, test, and validation data
Xfeatures_normalized = (Xfeatures - mean) / std
X_test_features_normalized = (X_test_features - mean) / std
X_validation_features_normalized = (X_validation_features - mean) / std

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
        x_wav2vec_batch, x_features_batch, y_batch = create_tensors_from_csv(x_paths, Xfeatures_normalized, labels, i, batch_size)  # Create batch tensors
        x_wav2vec_batch, x_features_batch, y_batch = x_wav2vec_batch.to(DEVICE), x_features_batch.to(DEVICE), y_batch.to(DEVICE)  # Move tensors to the device

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

    # Test phase
    with torch.no_grad():
        model.eval()  # Set model to evaluation mode
        all_y_true, all_y_pred = [], []  # Lists to store true and predicted labels

        for i in tqdm(range(0, len(x_test_paths), batch_size), desc="Training Progress"):
            x_wav2vec_batch, x_features_batch, y_batch = create_tensors_from_csv(x_test_paths, X_test_features_normalized, test_labels, i, batch_size)  # Create batch tensors
            x_wav2vec_batch, x_features_batch, y_batch = x_wav2vec_batch.to(DEVICE), x_features_batch.to(DEVICE), y_batch.to(DEVICE)  # Move tensors to the device
            y_pred = model(x_wav2vec_batch, x_features_batch).squeeze()  # Forward pass
            val_loss += criterion(y_pred.squeeze(), y_batch.float()).item()  # Accumulate validation loss
            all_y_true.extend(y_batch.cpu().numpy())  # Collect true labels
            all_y_pred.extend(y_pred.squeeze().cpu())  # Collect predicted probabilities

    val_loss = val_loss / (len(x_test_paths) // batch_size)
    accuracy, recall, f1 = calculate_metrics(np.array(all_y_true), np.array(all_y_pred))

    # Log results of the current epoch
    results_df.loc[len(results_df)] = [train_loss, val_loss, accuracy, recall, f1]

    print(f'Epochs {Epoch}: Train Loss = {train_loss}, Validation Loss = {val_loss}, Accuracy = {accuracy}, Recall = {recall}, F1 = {f1}')

# Preparing results folder
now_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")  # Timestamp for unique result folder
dir_path = TRAINING_DATA_PATH + now_time + learning_rate + DROP_OUT + batch_size

#creating the directories if needed
os.makedirs(dir_path, exist_ok=True)  # Create directory if it doesn't exist

# Save results to an Excel file
results_df.to_excel(f'{dir_path}/final_report.xlsx', index=False)

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

if SAVE_MODEL:
    # Create an instance of your model
    model = DeepFakeDetection(epochs=10, batch_size=32, learning_rate=0.001)

    # Set the model to evaluation mode
    model.eval()

    # Example input for the model (adjust the dimensions according to your input shape)
    dummy_input = (torch.randn(1, 1, 128, 128), torch.randn(1, 10))  # Adjust based on your input sizes

    # Save model to ONNX format
    onnx_model_path = "deepfake_detection.onnx"
    torch.onnx.export(
        model,  # Model
        dummy_input,  # Example input tuple
        onnx_model_path,  # Output file
        export_params=True,  # Store trained parameter weights
        opset_version=11,  # ONNX version
        input_names=["X_Wav2Vec", "X_Features"],  # Input names
        output_names=["output"],  # Output name
        dynamic_axes={"X_Wav2Vec": {0: "batch_size"}, "X_Features": {0: "batch_size"}}  # Dynamic axes
    )
    print(f"Model saved to {onnx_model_path}")

""" 
instuctions for trancfering the model:
pip onnx
    pip install onnx onnx-tf tensorflow


then convert from onnx model to tensorflow model:

from onnx_tf.backend import prepare
import onnx

# Load the ONNX model
onnx_model_path = "deepfake_detection.onnx"
onnx_model = onnx.load(onnx_model_path)

# Convert to TensorFlow
tf_rep = prepare(onnx_model)

# Export the TensorFlow model
tf_model_path = "deepfake_detection_tf"
tf_rep.export_graph(tf_model_path)
print(f"Model converted to TensorFlow and saved to {tf_model_path}")




this is the load and use:

import tensorflow as tf

# Load the TensorFlow model
tf_model = tf.saved_model.load("deepfake_detection_tf")

# Example input (ensure the input format matches the saved model)
import numpy as np
X_Wav2Vec = np.random.randn(1, 1, 128, 128).astype(np.float32)  # Adjust shape
X_Features = np.random.randn(1, 10).astype(np.float32)

# Use the model
output = tf_model(X_Wav2Vec=X_Wav2Vec, X_Features=X_Features)
print("Model output:", output)

"""