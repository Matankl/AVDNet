import os
import torch
from data_methods import *
from constants import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, classification_report
import seaborn as sns

# Error analysis function
def error_analysis(csv_path, model, mean, std, batch_size, device, save_failed_path="failed_predictions.txt", conf_matrix_path="confusion_matrix.png"):
    """
    Perform error analysis on a dataset by evaluating a model.

    Args:
        csv_path (str): Path to the CSV file containing features and labels.
        model (torch.nn.Module): Trained model to evaluate.
        mean (np.array): Mean of the features for normalization.
        std (np.array): Standard deviation of the features for normalization.
        batch_size (int): Batch size for evaluation.
        device (str): Device to use ("cpu" or "cuda").
        save_failed_path (str): Path to save the failed predictions.
        conf_matrix_path (str): Path to save the confusion matrix plot.

    Returns:
        None
    """
    # Load data
    x_paths, X_features, labels = load_csv_data(csv_path)
    x_paths_for_wav2vec = [os.path.join(WAV2VEC_FOLDER, p) for p in x_paths]
    X_features_normalized = (X_features - mean) / std

    model.eval()
    all_y_true, all_y_pred = [], []
    failed_files = []

    with torch.no_grad():
        for i in range(0, len(x_paths), batch_size):
            # Prepare batch
            x_wav2vec_batch, x_features_batch, y_batch = create_tensors_from_csv(
                x_paths_for_wav2vec, X_features_normalized, labels, i, batch_size
            )
            x_wav2vec_batch, x_features_batch, y_batch = (
                x_wav2vec_batch.to(device),
                x_features_batch.to(device),
                y_batch.to(device),
            )

            # Model prediction
            print(x_wav2vec_batch.shape, x_features_batch.shape)
            print(x_wav2vec_batch, x_features_batch)
            y_pred = model(x_wav2vec_batch, x_features_batch).squeeze()
            all_y_true.extend(y_batch.cpu().numpy())
            all_y_pred.extend(y_pred.cpu().numpy())

            # Track failed files
            binary_preds = (y_pred.cpu().numpy() > 0.5).astype(int)
            mismatched = np.where(binary_preds != y_batch.cpu().numpy())[0]
            for idx in mismatched:
                failed_files.append(x_paths[i + idx])

    # Compute metrics
    binary_y_pred = (np.array(all_y_pred) > 0.5).astype(int)
    conf_matrix = confusion_matrix(all_y_true, binary_y_pred)
    f1 = f1_score(all_y_true, binary_y_pred)
    accuracy = accuracy_score(all_y_true, binary_y_pred)
    recall = recall_score(all_y_true, binary_y_pred)

    # Save failed predictions
    with open(save_failed_path, "w") as f:
        for path in failed_files:
            f.write(path + "\n")

    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(conf_matrix_path)
    plt.close()

    # Print results
    print("Confusion Matrix:\n", conf_matrix)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Classification Report:\n", classification_report(all_y_true, binary_y_pred))

    print(f"Failed predictions saved to: {save_failed_path}")
    print(f"Confusion matrix saved to: {conf_matrix_path}")

# Example usage
if __name__ == "__main__":
    trained_model = r"C:\Users\parde\PycharmProjects\The-model\checkpoints\tmp_model_trial_4.pth" # Load your trained model
    model = torch.load(trained_model, map_location=DEVICE)


    csv_path = r"/data/Inputs/test_30h.csv"
    batch_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load mean and std from training data
    mean = 0.5
    std = 0.5

    # Run error analysis
    error_analysis(csv_path, model, mean, std, batch_size, device)
