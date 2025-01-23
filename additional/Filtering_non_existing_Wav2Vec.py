import numpy as np
import pandas as pd
import os
import torch
from constants import WAV2VEC_FOLDER

def filtering_wrong_wav2vec(csv_path, clmn, expected_shape):
    # Read CSV into a DataFrame
    df = pd.read_csv(csv_path)
    # Assuming the column containing paths is named "path", adjust if necessary
    initial_rows = len(df)

    # Function to check if file exists and has correct dimensions
    def is_valid_tensor(file_path):
        full_path = os.path.join(WAV2VEC_FOLDER, file_path)

        if not os.path.exists(full_path):
            print("file non existing")
            return False  # File does not exist

        try:
            matrix = np.load(full_path, allow_pickle=True)  # Load the matrix
            # Ensure it's a matrix and check dimensions
            if matrix.shape == expected_shape:
                return True
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

        print("Wrong sized")
        return False  # If it doesn't match the expected shape or loading fails

    # filtering wrong sized wav2vec or inexistans
    df_filtered = df[df[clmn].apply(is_valid_tensor)]

    # Count deleted rows
    deleted_rows = initial_rows - len(df_filtered)

    # Save the filtered DataFrame
    df_filtered.to_csv(csv_path, index=False)

    # Output result
    print(f"Deleted rows: {deleted_rows}")

if __name__ == '__main__':

    clmn = 'Wav2VecPath'
    expected_shape = (1, 199, 29)  # Adjust to your expected shape
    csv_paths = [f"../data/Inputs/{name}_70h.csv" for name in ["train", "test", "validation"]]

    for csv_path in csv_paths:
        print(f"filtering {csv_path}")
        filtering_wrong_wav2vec(csv_path, clmn, expected_shape)
