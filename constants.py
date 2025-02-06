import torch

# Model
WEIGHTS_DIRECTORY = "data/model"

#CSV PATHS
HOURS = 70
INPUTS_PATH = "data/Inputs"
TRAIN_CSV = f"{INPUTS_PATH}/train_{HOURS}h.csv"
TEST_CSV = f"{INPUTS_PATH}/test_{HOURS}h.csv"
VALIDATION_CSV = f"{INPUTS_PATH}/validation_{HOURS}h.csv"
WAV2VEC_FOLDER = 'D:\Database\Audio\DeepFakeProject\Wav2vecMatrices' # The folder containing the Wav2Vec matrices
DATASET_FOLDER = "/media/hp4ran/TOSHIBA EXT/Database/Audio/DeepFakeProject"
#OPTUNA PARAMETERS
LOAD_TRAINING = False
EPOCHS = 50
TRIALS = 60
PATIENCE = 2
PARTIAL_TRAINING = 0.001

DEBUGMODE = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Runnig on device:", DEVICE)
SAVE_MODEL = False


# The model parameters
BATCH_SIZE = 16
DROP_OUT = 0.3

# logs path
TRAINING_DATA_PATH = 'data/results/'  # Directory for saving training results
