import pickle
import numpy as np
import torch

# Path to the pickle file
pickle_file_path = r"C:\Users\pardes\Downloads\0000a0f45a2a9ca26455c76d7abfe5992806f8ad0f014a18616fb7dda86c508753765e61697993e5d2a0d9e2fab52a822b31ed5c3f7f3e5bc37495453f6b335f_1.pkl"

# Load the pickled object
with open(pickle_file_path, "rb") as file:
    obj = pickle.load(file)

# Check if the object is a PyTorch tensor
if isinstance(obj, torch.Tensor):
    # Print the shape of the tensor
    print("Shape of the PyTorch tensor:", obj.shape)
else:
    print("The unpickled object is not a PyTorch tensor.")
