from playsound3 import playsound

from constants import *
import os
import random
import numpy as np
import pandas as pd
import torchaudio
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_curve
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Define Dataset for Training & Validation
class Wav2VecDataset(Dataset):
    def __init__(self, csv_path, wav2vec_folder):
        self.data = pd.read_csv(csv_path)
        self.wav2vec_folder = wav2vec_folder

        # Extract paths, features, and labels
        self.x_paths = self.data.iloc[:, 1].values  # wav2vec2 matrix paths
        self.Xfeatures = self.data.iloc[:, 2:-1].values.astype(np.float32)  # Numeric features
        self.labels = self.data['label'].values.astype(int)  # Labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_path = os.path.join(self.wav2vec_folder, self.x_paths[idx])

        # Load wav2vec matrix
        wav2vec_matrix = np.load(x_path, allow_pickle=True)
        wav2vec_tensor = wav2vec_matrix.clone().detach()

        # Load additional features
        x_features = torch.tensor(self.Xfeatures[idx], dtype=torch.float32)

        # Load label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # BCELoss needs float labels

        return wav2vec_tensor, x_features, label

lfcc_transform = T.LFCC(
            sample_rate=16000, n_lfcc=80, n_filter=128, log_lf=False)

class ASVspoof5Dataset(Dataset):
    def __init__(self, data_dir, split, aug_prob=0, track = 1, fraction = 1):
        """
        Args:
            data_dir: Path to the ASVspoof5 dataset
            protocol_file: Path to the protocol file
            is_train: Whether this is for training or evaluation
            split: One of "train", "dev", "eval"
            max_frames: Maximum number of frames to use
            transform: Optional transforms to apply
        """
        self.data_dir = data_dir
        self.subset = split
        unwanted_attacks = ["A18", "A20", "A23", "A27", "A30", "A31", "A32"]
        self.sample_rate = 16000
        self.augment_prob = aug_prob
        self.expected_length = self.sample_rate * 4  # 4 seconds

        protocol_file = self.get_protocol_file(data_dir, split, track)

        # Read protocol file (TSV format in ASVspoof5)
        self.metadata = pd.read_csv(protocol_file, header=None, delim_whitespace=True)

        count = [0, 0]
        # Handle different protocol file formats based on track
        if track == 1:
            # For track 1 files, we have columns:
            # SPEAKER_ID FLAC_FILE_NAME SPEAKER_GENDER CODEC CODEC_Q CODEC_SEED ATTACK_TAG ATTACK_LABEL KEY TMP
            # Map labels: bonafide -> 0, spoof -> 1
            self.metadata.iloc[:, 8] = self.metadata.iloc[:, 8].map({'bonafide': 0, 'spoof': 1})
            # Keep relevant columns
            self.metadata = self.metadata.iloc[:, [1, 7, 8]]  # Example index selection
        elif track == 2:

            # For track 2 files, we have columns:
            # TARGET_SPEAKER_ID FLAC_FILE_NAME TARGET_GENDER ATTACK_LABEL ASV_KEY
            # Map labels: bonafide -> 0, spoof -> 1
            self.metadata['label'] = self.metadata.iloc[:, 3].apply(
                lambda x: 0 if x == 'bonafide' else 1
            )

            # Keep relevant columns
            self.metadata = self.metadata.iloc[:, [1, 3, 4]]  # Example index selection

        else: raise ValueError ("Wrong track number")

        #filtering out the non generated attacks.
        self.metadata = self.metadata[~self.metadata.iloc[:, 1].isin(unwanted_attacks)]

        for i in range(len(self.metadata)):
            count[self.metadata.iloc[i, 2]] += 1

        print(f"fake to real {count}, ratio: {count[1]/count[0]}")

        # Create audio paths based on subset
        if split == "Train":
            self.wavs_dir = os.path.join(data_dir, 'flac_T')
        elif split == "Validation":
            self.wavs_dir = os.path.join(data_dir, 'flac_D')
        else:  # eval
            self.wavs_dir = os.path.join(data_dir, 'flac_E_eval')

        def non_empty(row):
            filename = row[1]
            audio_path = os.path.join(self.wavs_dir, f"{filename}.flac")
            return os.path.exists(audio_path) and os.path.getsize(audio_path) > 1

        self.metadata = self.metadata[self.metadata.apply(non_empty, axis=1)]

        if fraction:
            self.metadata = self.metadata.sample(frac = fraction)

        is_train = (self.subset == "Train")
        print(f"Loaded {len(self.metadata)} {'training' if is_train else 'evaluation'} samples from {split} set")



    def get_protocol_file(self, data_dir, subset, track):
        # Define the paths to the dataset files based on the track
        if track == 1:
            if subset == "Train":
                return os.path.join(data_dir, "ASVspoof5_protocols/ASVspoof5.train.tsv")
            elif subset == "Validation":
                return os.path.join(data_dir, "ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv")
            elif subset == "Test":
                return os.path.join(data_dir, "ASVspoof5_protocols/ASVspoof5.eval.track_1.tsv")

        elif track == 2:
            if subset == "Train":
                return os.path.join(data_dir, "ASVspoof5_protocols/ASVspoof5.train.tsv")
            elif subset == "Validation":
                return os.path.join(data_dir, "ASVspoof5_protocols/ASVspoof5.dev.track_2.trial.tsv")
            elif subset == "Test":
                return os.path.join(data_dir, "ASVspoof5_protocols/ASVspoof5.eval.track_2.trial.tsv")

        raise ValueError(f"Wrong subset: {subset}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = row[1]
        label = row[8]

        # Full path to the audio file.
        audio_path = os.path.join(self.wavs_dir, f"{filename}.flac")


        try:
            waveform, sr = torchaudio.load(audio_path, format="flac")
        except Exception as e:
            print(f"[ERROR] Failed to load: {audio_path} - {e}")
            raise e  # or return None or dummy data

        # Ensure exact length using padding or truncation
        if waveform.shape[1] < self.expected_length:
            pad_size = self.expected_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_size))  # Pad with zeros
        elif waveform.shape[1] > self.expected_length:
            waveform = waveform[:, :self.expected_length]  # Truncate

        if self.augment_prob > random.random() and self.subset == "Train":
            waveform, _ = augment_audio_fixed(waveform, self.sample_rate)

        # Extract LFCC features from the waveform.
        lfcc_input = lfcc_transform(waveform)

        return lfcc_input, waveform, label

    # def extract_lfcc_torchaudio(self, waveform, sample_rate=16000, n_lfcc=80, n_filter=128, log_lf=False):
    #     """
    #     Extract LFCC features from waveform using torchaudio.
    #
    #     Args:
    #         waveform (torch.Tensor): Audio tensor of shape (1, samples)
    #         sample_rate (int): Sample rate of audio (default: 16kHz)
    #         n_lfcc (int): Number of LFCC coefficients (default: 40)
    #         n_filter (int): Number of linear filters (default: 128)
    #         log_lf (bool): Whether to apply log scale on LFCC (default: False)
    #
    #     Returns:
    #         torch.Tensor: LFCC features of shape (1, n_lfcc, time_steps)
    #     """
    #     lfcc_transform = T.LFCC(
    #         sample_rate=sample_rate,
    #         n_lfcc=n_lfcc,
    #         n_filter=n_filter,
    #         log_lf=log_lf
    #     )
    #
    #     lfcc_features = lfcc_transform(waveform)  # (1, n_lfcc, time_steps)
    #
    #     return lfcc_features


class RawAudioDatasetLoader(Dataset):
    def __init__(self, root_dir, split="Train", fraction = False, data_aug = 0):
        """
        Args:
            root_dir (str): Path to the 'database' directory containing 'Real' and 'Fake' subfolders.
            split (str): One of 'Train', 'Test', or 'Validation' (determines which CSVs to load).
        """
        self.data = []
        self.augment_prob = data_aug
        self.sample_rate = 16000
        self.expected_length = self.sample_rate * 4 # 4 seconds
        self.dataset_type = split

        # Recursively search for dataset_type.csv in all subdirectories
        for class_name in ["Real", "Fake"]:  # Labels inferred from folder names
            class_label = 0 if class_name == "Real" else 1
            class_path = os.path.join(root_dir, class_name)

            if not os.path.exists(class_path):
                continue  # Skip if folder doesn't exist

            for source_folder in os.listdir(class_path):
                source_path = os.path.join(class_path, source_folder)
                if os.path.isdir(source_path):  # Ensure it's a directory
                    csv_path = os.path.join(source_path, f"{split}.csv")
                    if os.path.exists(csv_path):
                        # Read CSV and extract filenames and labels
                        df = pd.read_csv(csv_path)
                        # Assumes first column is the filename (with .wav extension)
                        # and the last column is the label.
                        filenames = df.iloc[:, 0].tolist()
                        labels = df.iloc[:, -1].tolist() if len(df.columns) > 1 else [class_label] * len(df)
                        for i, filename in enumerate(filenames):
                            # Since the audio is in the same directory as the CSV, use source_path directly.

                            if os.path.exists(os.path.join(source_path, split, f"{filename}")):
                                self.data.append((os.path.join(source_path, split), filename, labels[i]))

        # Shuffle all (path, filename, label) entries together
        random.shuffle(self.data)
        if fraction:
            self.data = self.data[:int(len(self.data) * fraction)]

        # Unpack shuffled data into separate lists
        self.file_list = [(entry[0], entry[1]) for entry in self.data]  # (source_path, filename)
        self.labels = [entry[2] for entry in self.data]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_dir, filename = self.file_list[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)


        # Full path to the audio file.
        audio_path = os.path.join(audio_dir, f"{filename}")
        waveform, sr = torchaudio.load(audio_path, format="wav")

        # Decide whether to apply augmentation.
        use_augmented = random.random() < self.augment_prob
        if use_augmented and self.dataset_type == "Train":
            if DATA_AUGMENTATION:
                waveform, _ = augment_audio_fixed(waveform, self.sample_rate)
            else:
                waveform, _ = augment_audio(waveform, sr)

        # Ensure exact length using padding or truncation
        if waveform.shape[1] < self.expected_length:
            pad_size = self.expected_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_size))  # Pad with zeros
        elif waveform.shape[1] > self.expected_length:
            waveform = waveform[:, :self.expected_length]  # Truncate

        # Extract LFCC features from the waveform.
        lfcc_input = lfcc_transform(waveform)
        # For fine-tuning Wav2Vec, use the raw waveform.
        wav2vec_input = waveform

        return lfcc_input, wav2vec_input, label


def augment_audio(waveform, sample_rate):
    """
    Apply audio augmentation return the augmented waveform and a list of the applied augmentations
    """

    raise NotImplementedError("old function shouldn't be used")
    # Randomly apply augmentations
    augmentations = []
    if torch.rand(1) > 0.8:
        rand = (torch.rand(1).item())
        waveform = T.Vol(rand)(waveform)  # Random Volume Change
    if torch.rand(1) > 0.8:
        rand = 35 + int(torch.rand(1).item() * 10)
        waveform = T.TimeMasking(time_mask_param=rand)(waveform)  # SpecAugment Time Masking
    if torch.rand(1) > 0.8:
        rand = 2 + int(torch.rand(1).item() * 5)
        waveform = T.FrequencyMasking(freq_mask_param=rand)(waveform)  # SpecAugment Frequency Masking
    if torch.rand(1) > 0.8:
        waveform = waveform + 0.005 * torch.randn_like(waveform)  # Mild Noise Injection

    return waveform, augmentations

def time_mask_waveform(waveform, sample_rate, mask_duration_ms=100):
    """
    Apply time masking to an audio waveform by zeroing out a random segment.

    :param waveform: Tensor of shape (channels, time)
    :param sample_rate: Sampling rate of the waveform
    :param mask_duration_ms: Duration of the mask in milliseconds
    :return: Time-masked waveform
    """
    num_samples = waveform.shape[1]
    mask_duration = int((mask_duration_ms / 1000.0) * sample_rate)

    if mask_duration >= num_samples:
        return waveform  # Avoid masking the entire signal

    start_idx = torch.randint(0, num_samples - mask_duration, (1,)).item()
    waveform[:, start_idx:start_idx + mask_duration] = 0  # Zero out the segment

    return waveform

def frequency_mask_waveform(waveform, sample_rate, n_fft=2048, mask_size=2000):
    """Apply frequency masking using STFT and ISTFT with a randomly chosen mask range."""

    # Convert to frequency domain
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=n_fft//4, return_complex=True)

    # Generate frequency bins
    freqs = torch.linspace(0, sample_rate // 2, stft.shape[1])  # Frequency bins

    # Pick a random start frequency for masking
    max_start_freq = (sample_rate // 2) - mask_size
    start_freq = random.randint(0, max_start_freq)
    end_freq = start_freq + mask_size

    # Apply mask
    mask = (freqs >= start_freq) & (freqs <= end_freq)
    stft[:, mask, :] = 0  # Zero out the masked frequency bins

    # Convert back to time domain
    masked_waveform = torch.istft(stft, n_fft=n_fft, hop_length=n_fft//4)

    return masked_waveform

def augment_audio_fixed(waveform, sample_rate):
    """
    Apply audio augmentation return the augmented waveform and a list of the applied augmentations
    """
    # Randomly apply augmentations
    augmentations = []
    if torch.rand(1) > 0.8:
        # Volume Change
        vol_factor = 1.0 + 1.5 * (torch.rand(1).item()) - 0.5  # Random volume change factor
        waveform = waveform * vol_factor
    if torch.rand(1) > 0.8:
        # Time Masking (Applied to Waveform)
        mask_duration = torch.randint(250, 1000, (1,)).item()  # Random mask duration between 250-1000ms
        waveform = time_mask_waveform(waveform, sample_rate, mask_duration_ms=mask_duration)
    if torch.rand(1) > 0.8:
        mask_size= torch.randint(500, 3000, (1,)).item()
        waveform = frequency_mask_waveform(waveform, sample_rate, mask_size=mask_size)
    if torch.rand(1) > 0.8:
        waveform = waveform + 0.005 * torch.randn_like(waveform)  # Mild Noise Injection

    return waveform, augmentations

class RecursiveFakeAudioDataset(Dataset):
    def __init__(self, root_dir, split="Fake", fraction=False):
        """
        Recursively loads audio files from a directory structure: main_folder->language->technique->audio.wav
        and assigns them all label 1 (Fake).

        Args:
            root_dir (str): Path to the main folder containing language subfolders.
            split (str): Only used for consistency with existing loader interface.
            fraction (float or bool): If provided as float (0-1), loads only that fraction of data.
        """
        self.data = []
        # self.augment_prob = 0.20
        self.sample_rate = 16000
        self.expected_length = self.sample_rate * 4  # 4 seconds
        self.dataset_type = split
        self.history = {}

        # Recursively find all .wav files
        for lang_dir in os.listdir(root_dir):
            lang_path = os.path.join(root_dir, lang_dir)
            if not os.path.isdir(lang_path):
                continue

            for technique_dir in os.listdir(lang_path):
                technique_path = os.path.join(lang_path, technique_dir)
                if technique_dir not in self.history:
                    self.history[technique_dir] = [0, 0]
                if not os.path.isdir(technique_path):
                    continue

                # Find all WAV files in this technique directory
                for file in os.listdir(technique_path):
                    if file.endswith('.wav') or file.endswith('.flac'):
                        self.data.append((technique_path, file, 1))  # Always assign label 1 (Fake)
                        self.history[technique_path][0] += 1

        # Shuffle all (path, filename, label) entries together
        random.shuffle(self.data)
        if fraction and isinstance(fraction, float) and 0 < fraction < 1:
            self.data = self.data[:int(len(self.data) * fraction)]

        # Unpack shuffled data into separate lists
        self.file_list = [(entry[0], entry[1]) for entry in self.data]  # (source_path, filename)
        self.labels = [entry[2] for entry in self.data]

        print(f"Loaded {len(self.data)} fake audio files from {root_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_dir, filename = self.file_list[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Full path to the audio file
        audio_path = os.path.join(audio_dir, filename)
        if audio_path.endswith('.wav'):
            waveform, sr = torchaudio.load(audio_path, format="wav")

        elif audio_path.endswith('.flac'):
            waveform, sr = torchaudio.load(audio_path, format="flac")

        else:
            raise NotImplementedError ("Format not expected")

        # Decide whether to apply augmentation
        use_augmented = random.random() < self.augment_prob
        if use_augmented and self.dataset_type == "Train":
            if DATA_AUGMENTATION:
                waveform, _ = augment_audio_fixed(waveform, self.sample_rate)
            else:
                waveform, _ = augment_audio(waveform, sr)

        # Ensure exact length using padding or truncation
        if waveform.shape[1] < self.expected_length:
            pad_size = self.expected_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_size))  # Pad with zeros
        elif waveform.shape[1] > self.expected_length:
            waveform = waveform[:, :self.expected_length]  # Truncate

        # Extract LFCC features from the waveform
        lfcc_input = lfcc_transform(waveform)
        # For fine-tuning Wav2Vec, use the raw waveform
        wav2vec_input = waveform

        return lfcc_input, wav2vec_input, label


def extract_lfcc_torchaudio(waveform, sample_rate=16000, n_lfcc=80, n_filter=128, log_lf=False):
    """
    Extract LFCC features from waveform using torchaudio.

    Args:
        waveform (torch.Tensor): Audio tensor of shape (1, samples)
        sample_rate (int): Sample rate of audio (default: 16kHz)
        n_lfcc (int): Number of LFCC coefficients (default: 40)
        n_filter (int): Number of linear filters (default: 128)
        log_lf (bool): Whether to apply log scale on LFCC (default: False)

    Returns:
        torch.Tensor: LFCC features of shape (1, n_lfcc, time_steps)
    """
    lfcc_transform = T.LFCC(
        sample_rate=sample_rate,
        n_lfcc=n_lfcc,
        n_filter=n_filter,
        log_lf=log_lf
    )

    lfcc_features = lfcc_transform(waveform)  # (1, n_lfcc, time_steps)

    return lfcc_features



    #COMMENT
    # bundle = pipelines.WAV2VEC2_ASR_BASE_960H
    # bundle = pipelines.WAV2VEC2_XLSR_53 #1024 features but more suited for multi lingual
    # model = bundle.get_model()

# Function to create DataLoader
def get_dataloader(split, root_dir, pin_memory=False, batch_size=32, shuffle=True, num_workers=4, fraction =None, data_aug = 0):
    """
    Creates a DataLoader for the given CSV (defining the dataset split) and data root directory.

    Args:
        csv_path (str): Path to the CSV file containing the list of files (e.g., "Train.csv" or "Validation.csv").
        root_dir (str): Path to the root directory containing the data (e.g., the folder with 'Real' and 'Fake' subdirectories).
        pin_memory (bool): Whether to pin memory (useful for GPUs).
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes.

    Returns:
        DataLoader: The DataLoader instance for the dataset.
    """
    if DATASET == "CUSTOM":
        if "Fake" == split: # for testing only
            dataset = RecursiveFakeAudioDataset(root_dir=root_dir, split=split, fraction=fraction)
        else:
            dataset = RawAudioDatasetLoader(root_dir=root_dir, split=split, fraction = fraction, data_aug = data_aug)

    elif DATASET == "ASVSPOOF":
        dataset = ASVspoof5Dataset(root_dir, split, data_aug, track=TRACK, fraction=fraction)

    else: raise ValueError(f"Dataset {split} not supported yet.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                        pin_memory=pin_memory, drop_last=True)
    return dataloader

# Function to create tensors for training/validation batches from CSV data
def create_tensors_from_csv(x_paths, Xfeatures, labels, start_idx, block_num, target_shape=None):
    """
    Creates tensors from wav2vec2 matrices and labels.

    Parameters:
    - x_paths (list): List of paths to .npy files containing wav2vec2 matrices.
    - labels (list): Corresponding labels for the wav2vec2 matrices.
    - start_idx (int): Starting index in the dataset.
    - block_num (int): Number of samples to process in one block.
    - target_shape (tuple): Desired shape for the wav2vec2 matrices (e.g., (T, D)).

    Returns:
    - x (torch.Tensor): Tensor of wav2vec2 matrices (with added channel dimension).
    - y (torch.Tensor): Tensor of labels.
    """
    x_wav2vec, x_vectors, y = [], [], []
    for i in range(start_idx, min(start_idx + block_num, len(x_paths))):

        # Load wav2vec matrix for a sample
        wav2vec_matrix = np.load(x_paths[i], allow_pickle=True)

        # Convert the matrix to a tensor and add channel dimension
        wav2vec_matrix = wav2vec_matrix.clone().detach()
        x_wav2vec.append(wav2vec_matrix)  # Directly append tensor (not wrapped in a list)

        tensor_vector = torch.tensor(Xfeatures[i], dtype=torch.float).detach()
        x_vectors.append(tensor_vector)

        y.append(labels[i])

    # Stack tensors into a single batch tensor
    x_wav2vec = torch.stack(x_wav2vec)  # Shape: (batch_size, T, D) or (batch_size, channels, T, D)
    x_vectors = torch.stack(x_vectors)
    y = torch.tensor(y, dtype=torch.float)  # Convert labels to tensor
    if(DEBUGMODE):
        print(x_wav2vec.shape)
        print(y.shape)
    return x_wav2vec, x_vectors, y

# Function to calculate evaluation metrics
def calculate_metrics(y_true, y_pred):
    y_pred_labels = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
    acc = accuracy_score(y_true, y_pred_labels)  # Calculate accuracy
    recall = recall_score(y_true, y_pred_labels)  # Calculate recall
    f1 = f1_score(y_true, y_pred_labels)  # Calculate F1-score
    return acc, recall, f1

# Function to calculate evaluation metrics
def calculate_metrics_4(y_true, y_pred):
    """
    Calculate accuracy, recall, F1-score, and precision.

    :param y_true: Ground truth labels (numpy array).
    :param y_pred: Predicted probabilities (numpy array).
    :return: Dictionary containing accuracy, recall, F1-score, and precision.
    """
    y_pred_labels = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
    acc = accuracy_score(y_true, y_pred_labels)
    recall = recall_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels)

    return {
        "accuracy": acc,
        "recall": recall,
        "f1_score": f1,
        "precision": precision,
    }

def calculate_eer(y_true, y_pred):
    """
    Calculate the Equal Error Rate (EER).

    :param y_true: Ground truth labels (numpy array).
    :param y_pred: Predicted probabilities (numpy array).
    :return: EER value
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    fnr = 1 - tpr
    eer_threshold_index = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2
    return eer
