import torch
import torch.nn as nn
import torchvision.models as models
from transformers import Wav2Vec2Model


# =============================================================================
# 1. VGG16 Feature Extractor with Partial Freezing
# =============================================================================
class VGG16FeatureExtractor(nn.Module):
    def __init__(self, freeze=True, freeze_vgg_layers=None):
        """
        Loads a pretrained VGG16 and uses only its convolutional (features) part.

        Args:
            freeze (bool): Whether to freeze layers.
            freeze_vgg_layers (int or None): If None, freeze all layers when freeze is True.
              Otherwise, only freeze the first `freeze_vgg_layers` modules.
        """
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16_bn(pretrained=True)

        # Modify first convolution layer to accept 1-channel input
        vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        #keeping only the convolutions (cutting the dense)
        self.features = vgg16.features

        if freeze:
            if freeze_vgg_layers is None:
                for param in self.features.parameters():
                    param.requires_grad = False
            else:
                for i, layer in enumerate(self.features):
                    if i < freeze_vgg_layers:
                        for param in layer.parameters():
                            param.requires_grad = False

    def forward(self, x):
        """
        x: Tensor of shape [B, 3, H, W] (e.g., spectrogram-like input)
        Returns: feature maps of shape [B, 512, H_out, W_out]
        """
        return self.features(x)


# =============================================================================
# 2. ResNet Feature Extractor with Partial Freezing
# =============================================================================
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, model_name="resnet34", freeze=True, freeze_resnet_layers=None):
        """
        Loads a pretrained ResNet model and uses the convolutional trunk.

        Args:
            model_name (str): e.g., "resnet50", "resnet34", etc.
            freeze (bool): Whether to freeze layers.
            freeze_resnet_layers (int or None): If None, freeze all layers when freeze is True.
              Otherwise, freeze only the first `freeze_resnet_layers` modules in the features.
        """
        super(ResNetFeatureExtractor, self).__init__()
        resnet = getattr(models, model_name)(pretrained=True)

        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Build a sequential model that stops before the average pool and fc layers.
        self.features = nn.Sequential(
            resnet.conv1,  # typically outputs 64 channels
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # output channels depend on the model; for resnet50: 256
            resnet.layer2,  # for resnet50: 512
            resnet.layer3,  # for resnet50: 1024
            resnet.layer4  # for resnet50: 2048
        )

        # 🔹 Modify first layer to accept 1-channel input

        if freeze:
            if freeze_resnet_layers is None:
                for param in self.features.parameters():
                    param.requires_grad = False
            else:
                # Freeze only the first freeze_resnet_layers modules in the sequential container.
                for i, layer in enumerate(self.features):
                    if i < freeze_resnet_layers:
                        for param in layer.parameters():
                            param.requires_grad = False

    def forward(self, x):
        """
        x: Tensor of shape [B, 3, H, W] (e.g., spectrogram-like input)
        Returns: feature maps of shape [B, C, H_out, W_out] (for resnet50, C=2048)
        """
        return self.features(x)


# =============================================================================
# 3. Wav2Vec2 Feature Extractor with Partial Freezing
# =============================================================================
class Wav2VecFeatureExtractor(nn.Module):
    def __init__(self, freeze=True, freeze_feature_extractor=True, freeze_encoder_layers=0):
        """
        Loads a pretrained Wav2Vec2 model from transformers.

        Args:
            freeze (bool): Whether to freeze parts of the model.
            freeze_feature_extractor (bool): If True, freeze the convolutional feature extractor.
            freeze_encoder_layers (int): Number of initial transformer encoder layers to freeze.
        """
        super(Wav2VecFeatureExtractor, self).__init__()
        # self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")

        if freeze:
            if freeze_feature_extractor:
                for param in self.model.feature_extractor.parameters():
                    param.requires_grad = False
            if freeze_encoder_layers > 0:
                for i in range(min(freeze_encoder_layers, len(self.model.encoder.layers))):
                    for param in self.model.encoder.layers[i].parameters():
                        param.requires_grad = False

    def forward(self, x):
        """
        x: Tensor of shape [B, T] (raw audio waveform)
        Returns: last hidden state [B, T, hidden_dim] (for wav2vec2-large, hidden_dim=1024)
        """
        outputs = self.model(x)
        return outputs.last_hidden_state


# =============================================================================
# 4. Fusion Transformer Module
# =============================================================================
class FusionTransformer(nn.Module):
    def __init__(self, cnn_in_channels, wav2vec_in_dim, d_model=256, nhead=8, num_layers=2, dropout=0.1):
        """
        Projects features from the CNN branch and the Wav2Vec branch to a common dimension,
        concatenates them, and fuses via a Transformer encoder.

        Args:
            cnn_in_channels: Number of channels from the CNN output (e.g., 512 for VGG, 2048 for ResNet).
            wav2vec_in_dim: Hidden dimension from the Wav2Vec model (e.g., 1024).
            d_model: Common projection dimension.
            nhead: Number of attention heads.
            num_layers: Number of Transformer encoder layers.
            dropout: Dropout rate.
        """
        super(FusionTransformer, self).__init__()
        self.cnn_proj = nn.Linear(cnn_in_channels, d_model)
        self.wav2vec_proj = nn.Linear(wav2vec_in_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, cnn_feat, wav2vec_feat):
        """
        Args:
            cnn_feat: Tensor of shape [B, C, H, W] from the CNN extractor.
            wav2vec_feat: Tensor of shape [B, T, wav2vec_in_dim] from the Wav2Vec extractor.

        Process:
            - Reshape CNN features: [B, C, H, W] -> [B, H*W, C] and project.
            - Project Wav2Vec features.
            - Concatenate along the sequence dimension.
            - Fuse with Transformer encoder and pool to get a fixed-length representation.

        Returns:
            fused_feature: Tensor of shape [B, d_model]
        """
        B, C, H, W = cnn_feat.shape
        cnn_tokens = cnn_feat.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        cnn_tokens = self.cnn_proj(cnn_tokens)  # [B, H*W, d_model]

        wav2vec_tokens = self.wav2vec_proj(wav2vec_feat)  # [B, T, d_model]

        # Concatenate tokens and apply dropout before fusion
        tokens = torch.cat([cnn_tokens, wav2vec_tokens], dim=1)  # [B, H*W + T, d_model]
        tokens = self.dropout(tokens)

        fused_tokens = self.transformer_encoder(tokens)
        fused_feature = fused_tokens.mean(dim=1)
        return fused_feature


# =============================================================================
# 5. Dense Classifier Module
# =============================================================================
class DenseClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, dropout=0.1):
        """
        A fully connected classifier with a decreasing schedule of neurons ending in a single output.

        Args:
            input_dim: Dimension of the input vector.
            hidden_dims: List of hidden layer sizes. If None, defaults to [input_dim//2, input_dim//4].
            dropout: Dropout rate.
        """
        super(DenseClassifier, self).__init__()
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4]
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


# =============================================================================
# 6. Full DeepFake Detector Model with Backbone Choice
# =============================================================================
class DeepFakeDetector(nn.Module):
    def __init__(self,
                 backbone="vgg",  # "vgg" or "resnet"
                 freeze_cnn=True, freeze_cnn_layers=None,
                 freeze_wav2vec=True, freeze_feature_extractor=True, freeze_encoder_layers=0,
                 d_model=256, nhead=8, num_layers=2, dense_hidden_dims=None):
        """
        Combines a CNN-based feature extractor (VGG16 or ResNet), a Wav2Vec2 extractor,
        a Transformer fusion module, and a dense classifier for binary deepfake detection.

        Args:
            backbone (str): Choose "vgg" or "resnet" for the CNN branch.
            freeze_cnn (bool): Whether to freeze the CNN extractor.
            freeze_cnn_layers (int or None): Number of initial CNN modules to freeze.
                For VGG16, this applies to vgg.features; for ResNet, to self.features.
            freeze_wav2vec (bool): Whether to freeze parts of the Wav2Vec model.
            freeze_feature_extractor (bool): Whether to freeze the Wav2Vec feature extractor.
            freeze_encoder_layers (int): Number of initial Wav2Vec encoder layers to freeze.
            d_model, nhead, num_layers: Parameters for the fusion Transformer.
            dense_hidden_dims: Hidden layer sizes for the dense classifier.
        """
        super(DeepFakeDetector, self).__init__()

        self.config = {
            "backbone": backbone,
            "freeze_cnn": freeze_cnn,
            "freeze_cnn_layers": freeze_cnn_layers,
            "freeze_wav2vec": freeze_wav2vec,
            "freeze_feature_extractor": freeze_feature_extractor,
            "freeze_encoder_layers": freeze_encoder_layers,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dense_hidden_dims": dense_hidden_dims
        }

        # Select CNN backbone and set the expected output channels.
        if backbone.lower() == "vgg":
            self.cnn_extractor = VGG16FeatureExtractor(freeze=freeze_cnn, freeze_vgg_layers=freeze_cnn_layers)
            cnn_channels = 512

        elif backbone.lower() == "resnet":
            self.cnn_extractor = ResNetFeatureExtractor(model_name="resnet50", freeze=freeze_cnn,
                                                        freeze_resnet_layers=freeze_cnn_layers)
            cnn_channels = 2048 # for resnet50

        elif backbone.lower() == "resnet34":
            self.cnn_extractor = ResNetFeatureExtractor(model_name="resnet34", freeze=freeze_cnn,
                                                        freeze_resnet_layers=freeze_cnn_layers)
            cnn_channels = 512 # for resnet34

        else:
            raise ValueError("Unsupported backbone. Choose 'vgg' or 'resnet'.")

        self.wav2vec_extractor = Wav2VecFeatureExtractor(freeze=freeze_wav2vec,
                                                         freeze_feature_extractor=freeze_feature_extractor,
                                                         freeze_encoder_layers=freeze_encoder_layers)
        self.fusion = FusionTransformer(cnn_in_channels=cnn_channels,
                                        wav2vec_in_dim=1024,  # for wav2vec2-large
                                        d_model=d_model,
                                        nhead=nhead,
                                        num_layers=num_layers)

        self.bn = nn.BatchNorm1d(d_model)
        self.classifier = DenseClassifier(input_dim=d_model, hidden_dims=dense_hidden_dims)

    def forward(self, image, audio):
        """
        Args:
            image: Tensor of shape [B, 3, H, W] for the CNN extractor (spectrogram-like representation).
            audio: Tensor of shape [B, T] (raw audio waveform for Wav2Vec2).
        Returns:
            Tensor of shape [B, 1] (logit for binary classification).
        """
        cnn_feat = self.cnn_extractor(image)  # shape depends on backbone
        audio = audio.squeeze(1)  # Removes the channel dimension
        wav2vec_feat = self.wav2vec_extractor(audio)  # [B, T]
        fused_feature = self.fusion(cnn_feat, wav2vec_feat)  # [B, d_model]
        fused_feature = self.bn(fused_feature)
        output = self.classifier(fused_feature)  # [B, 1]
        return output


# =============================================================================
# Example Usage:
# =============================================================================
if __name__ == "__main__":
    # Dummy inputs:
    dummy_image = torch.randn(2, 3, 224, 224)  # For CNN branch (spectrogram-like)
    dummy_audio = torch.randn(2, 16000)  # Raw audio waveform (2 samples, 1 sec at 16kHz)

    # Example 1: Using VGG backbone (freezing first 10 modules of VGG features)
    model_vgg = DeepFakeDetector(backbone="vgg", freeze_cnn=True, freeze_cnn_layers=10,
                                 freeze_wav2vec=True, freeze_feature_extractor=True, freeze_encoder_layers=4)
    output_vgg = model_vgg(dummy_image, dummy_audio)
    print("VGG-based model output shape:", output_vgg.shape)

    # Example 2: Using ResNet backbone (freezing first 3 modules of ResNet features)
    model_resnet = DeepFakeDetector(backbone="resnet", freeze_cnn=True, freeze_cnn_layers=3,
                                    freeze_wav2vec=True, freeze_feature_extractor=True, freeze_encoder_layers=4)
    output_resnet = model_resnet(dummy_image, dummy_audio)
    print("ResNet-based model output shape:", output_resnet.shape)
