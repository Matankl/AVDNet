import pandas as pd
import numpy as np
import os

# Path to the CSV file
CSV_FILE = "../data/Inputs/validation_30h.csv"
matrices_folder = "D:\Database\Audio\DeepFakeProject\Wav2vecMatrices"

# Define the correct shape
EXPECTED_SHAPE = (1, 199, 29)

# Step 1: Load the CSV file
df = pd.read_csv(CSV_FILE)

# Step 2: Filter rows with the correct matrix shape
valid_rows = []

for index, row in df.iterrows():
    matrix_path = row['Wav2VecPath']  # Assuming the column containing paths is named 'path'

    # Load the wav2vec matrix (assuming it's a NumPy array saved in .npy format)
    matrix = np.load(os.path.join(matrices_folder, matrix_path), allow_pickle=True)

    # Check if the shape matches the expected shape
    if matrix.shape == EXPECTED_SHAPE:
        valid_rows.append(row)


# Step 3: Create a new DataFrame with valid rows
filtered_df = pd.DataFrame(valid_rows)

# Step 4: Save the filtered DataFrame back to the CSV
filtered_df.to_csv(CSV_FILE, index=False)
