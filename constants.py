import warnings
warnings.filterwarnings("ignore")  # Suppress user warnings
import torch


#DATA
DATASET = "ASVSPOOF" # "CUSTOM",
# DATASET = "CUSTOM"
# HOURS = 70
INPUTS_PATH = "data/Inputs"
ASVSPOOF_FOLDER = '/home/hp4ran/Desktop/ASVspoof'
TRACK = 1
CUSTOM_DATASET_FOLDER = "/home/hp4ran/DeepFakeProject"


if DATASET == "ASVSPOOF":
    DATASET_FOLDER = ASVSPOOF_FOLDER
elif DATASET == "CUSTOM":
    DATASET_FOLDER = CUSTOM_DATASET_FOLDER

#OPTUNA PARAMETERS
SEED = 42
LOAD_TRAINING = True
DATA_AUGMENTATION = True # to use the previous data augmentation script, set to False
EPOCHS = 100
TRIALS = 20
PATIENCE = 4
PARTIAL_TRAINING = 1# between 0-1 how much of the data to use
PARTIAL_TESTING = 1
DEBUGMODE = False


# logs path
TRAINING_DATA_PATH = 'data/results/'  # Directory for saving training results

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", DEVICE)
#a file for dynamic loading ?