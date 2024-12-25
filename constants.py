import torch

EPOCHS = 20
BATCH_SIZE = 30
LEARNING_RATE = 0.0001

DEBUGMODE = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
WEIGHTS_DIRECTORY = "data/model"
INPUT_CSV = "D:\Database\Audio\inputs.csv"
WAV2VEC_FOLDER = 'D:\Database\Audio\Wav2VecRepresentation'
INPUT_SHAPE=None # to modify
