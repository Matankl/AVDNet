import torch

# Model
WEIGHTS_DIRECTORY = "data/model"
INPUT_CSV = "/home/or/Desktop/DataSets/DeepFakeProject/Inputs.csv"
WAV2VEC_FOLDER = '/home/or/Desktop/DataSets/DeepFakeProject/Wav2vecMatrices' #where
INPUT_SHAPE=None # to modify
NUM_OF_CLASSES = 2

EPOCHS = 25
DEBUGMODE = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
DENSE_LAYERS = 3
DROP_OUT = 0.5

TRAINING_DATA_PATH = 'data/results/'  # Directory for saving training results
