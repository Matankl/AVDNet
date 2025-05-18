from data_methods import get_dataloader
from constants import DATASET_FOLDER
import sounddevice as sd
train_data = get_dataloader("Train", DATASET_FOLDER, fraction=0.01, batch_size=1, num_workers=1, data_aug=1)

for sample in train_data:

    lfcc, waveform, label = sample


    waveform_np = waveform.squeeze().cpu().numpy()
    sd.play(waveform_np, 16000)
    sd.wait()