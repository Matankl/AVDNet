import torch

# Model
WEIGHTS_DIRECTORY = "data/model"

#CSV PATHS
TRAIN_CSV = "additional/train_30h.csv"
TEST_CSV = "additional/test_30h.csv"
VALIDATION_CSV = "additional/validation_30h.csv"

WAV2VEC_FOLDER = 'D:\Database\Audio\DeepFakeProject\Wav2vecMatrices' # The folder containing the Wav2Vec matrices


EPOCHS = 25
DEBUGMODE = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The model parameters
BATCH_SIZE = 16
DROP_OUT = 0.5

# logs path
TRAINING_DATA_PATH = 'data/results/'  # Directory for saving training results
