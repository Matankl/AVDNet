import torch
import torch.nn as nn
import torch.nn.functional as F


# Double CNN Feature Extractors
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels, output_dim=128, seq_length=50):  # Ensure seq_length > 2
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Conv2d(128, output_dim, kernel_size=(1, 1))  # Project to feature dim
        self.seq_length = seq_length

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))  # (batch, 128, freq_dim, time_dim)

        # Ensure output shape is (batch, time_steps, feature_dim)
        x = self.flatten(x)  # Reduce spatial dimensions while keeping temporal info
        x = x.permute(0, 3, 1, 2).squeeze(2)  # (batch, time_dim, feature_dim)
        return x  # Shape: (batch, time_steps, feature_dim)


# Transformer Fusion Module
class TransformerFusion(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=256, seq_length=50):
        super(TransformerFusion, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.seq_length = seq_length

    def forward(self, lfcc_features, wav2vec_features):
        # Ensure both inputs are sequences (batch, time_steps, feature_dim)
        assert lfcc_features.shape[1] == wav2vec_features.shape[1], "Sequence lengths must match"

        # Merge sequences together (concatenate along sequence dimension)
        fused_input = torch.cat((lfcc_features, wav2vec_features), dim=1)  # Shape: (batch, 2*time_steps, feature_dim)

        # Self-attention across the full sequence
        attn_output, _ = self.attention(fused_input, fused_input, fused_input)

        # Apply dense projection, normalization, and dropout
        attn_output = self.fc(attn_output)
        attn_output = self.layer_norm(F.relu(attn_output))
        attn_output = self.dropout(attn_output)

        # Reduce sequence dimension to a single vector (mean pooling)
        return attn_output.mean(dim=1)  # (batch, feature_dim)


# Complete Model
class DeepFakeDetector(nn.Module):
    def __init__(self, lfcc_input_channels, wav2vec_input_channels, handcrafted_feature_dim, seq_length=50):
        super(DeepFakeDetector, self).__init__()

        # CNNs for LFCC and Wav2Vec
        self.cnn_lfcc = CNNFeatureExtractor(input_channels=lfcc_input_channels, seq_length=seq_length)
        self.cnn_wav2vec = CNNFeatureExtractor(input_channels=wav2vec_input_channels, seq_length=seq_length)

        # Transformer Fusion
        self.transformer_fusion = TransformerFusion(input_dim=128, seq_length=128)  # Assuming CNN outputs 128 features

        # Fully Connected Layers
        total_input_dim = 256 + handcrafted_feature_dim  # Transformer output + handcrafted features
        self.fc1 = nn.Linear(total_input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Binary classification (real vs fake)
        self.dropout = nn.Dropout(0.3)

    def forward(self, lfcc_input, wav2vec_input, handcrafted_features):
        # CNN Feature Extraction
        lfcc_features = self.cnn_lfcc(lfcc_input)
        wav2vec_features = self.cnn_wav2vec(wav2vec_input)

        # Transformer Fusion
        fused_features = self.transformer_fusion(lfcc_features, wav2vec_features)

        # Concatenate handcrafted features
        combined_features = torch.cat((fused_features, handcrafted_features), dim=1)

        # Fully Connected Layers
        x = F.relu(self.fc1(combined_features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return torch.sigmoid(x)  # Binary classification output

