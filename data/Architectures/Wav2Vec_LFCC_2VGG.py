import torch
import torchvision.models as models
import torchaudio.pipelines as pipelines
import torch.nn as nn

class VGG16_LFCC_FineTune(nn.Module):
    """
    A custom version of VGG for fine-tunning.
    """
    def __init__(self, num_classes=1, input_channels=1, layers_keep = 5):
        super(VGG16_LFCC_FineTune, self).__init__()

        # Load Pretrained VGG-16 Model
        self.vgg16 = models.vgg16(pretrained=True)

        # Modify first convolutional layer (from 3 RGB channels → 1 LFCC channel)
        self.vgg16.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)

        # Freeze convolutional blocks (Conv1 & Conv2)
        for layer in range(layers_keep):  # Freeze first 5 layers (Conv1 + Conv2)
            for param in self.vgg16.features[layer].parameters():
                param.requires_grad = False

        # Trainable layers remain unfrozen
        for layer in range(layers_keep, len(self.vgg16.features)):
            for param in self.vgg16.features[layer].parameters():
                param.requires_grad = True

        # Modify the final classification layer
        self.vgg16.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.vgg16(x)
        return torch.sigmoid(x)  # Binary classification output


# Load Pretrained Wav2Vec2 Model
bundle = pipelines.WAV2VEC2_ASR_LARGE_960H
wav2vec_model = bundle.get_model()
layers_to_train = 4

# Freeze early layers (feature extractor)
for param in wav2vec_model.feature_extractor.parameters():
    param.requires_grad = False  # Keep the base model frozen

# Fine-tune the last Transformer layers
for param in wav2vec_model.encoder.layers[-layers_to_train:].parameters():
    param.requires_grad = True  # Unfreeze last 4 Transformer layers

# Additional Linear Layer for Classification
class Wav2VecClassifier(nn.Module):
    def __init__(self):
        super(Wav2VecClassifier, self).__init__()
        self.wav2vec = wav2vec_model
        self.fc = nn.Linear(1024, 256)  # Reduce feature dim before merging with LFCC

    def forward(self, x):
        x = self.wav2vec.extract_features(x)[0]  # Extract Wav2Vec embeddings
        x = torch.mean(x, dim=1)  # Pooling over time
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x  # Shape: (batch, 256)