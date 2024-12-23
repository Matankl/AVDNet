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
INPUT_CSV = "data/inputs.csv"
WAV2VEC_FOLDER = '/home/or/Desktop/DataSets/pklGeneratedFolder'
INPUT_SHAPE=None # to modify
NUM_OF_CLASSES = 2

EPOCHS = 20
DEBUGMODE = False