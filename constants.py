import torch

# Model
WEIGHTS_DIRECTORY = "data/model"
INPUT_CSV = "D:\Database\Audio\DeepFakeProject\inputs.csv"
WAV2VEC_FOLDER = 'D:\Database\Audio\DeepFakeProject\Wav2vecMatrices'

EPOCHS = 30
DEBUGMODE = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
