import torch


# Model
WEIGHTS_DIRECTORY = "data/model"
INPUT_CSV = "D:\Database\Audio\inputs.csv"
WAV2VEC_FOLDER = 'D:\Database\Audio\Wav2VecRepresentation'
INPUT_SHAPE=None # to modify
NUM_OF_CLASSES = 2

EPOCHS = 20
DEBUGMODE = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

