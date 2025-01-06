import torch

# Signal processing
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NUM_FFT = 512
BUCKET_STEP = 1
MAX_SEC = 10

# Model
WEIGHTS_DIRECTORY = "data/model"
INPUT_CSV = "/home/or/Desktop/DataSets/DeepFakeProject/Inputs.csv"
WAV2VEC_FOLDER = '/home/or/Desktop/DataSets/DeepFakeProject/Wav2vecMatrices'
INPUT_SHAPE=None # to modify
NUM_OF_CLASSES = 2

EPOCHS = 1
DEBUGMODE = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

