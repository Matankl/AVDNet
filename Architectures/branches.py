from torch import nn
import torchvision.models as models
from transformers import Wav2Vec2Model
import torch.nn.functional as F
import torch

class VGG_only(nn.Module):

    def __init__(self):
        super(VGG_only, self).__init__()
        self.VGG = models.vgg16_bn(pretrained=False)

        self.config = {}

        # Modify first convolution layer to accept 1-channel input
        self.VGG.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        feats_list = list(self.VGG.features)
        new_feats_list = []
        for feat in feats_list:
            new_feats_list.append(feat)
            # if isinstance(feat, nn.Conv2d):
            #     new_feats_list.append(nn.Dropout(p=0.5))

        # modify convolution layers
        self.VGG.features = nn.Sequential(*new_feats_list)

        self.final_head = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.VGG(x)
        x = self.final_head(x)
        return x



class Wav2Vec2DeepfakeClassifier(nn.Module):
    def __init__(self, pretrained_name="facebook/wav2vec2-large-960h"):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_name)
        self.pooling = AttentionPooling(self.wav2vec2.config.hidden_size)  # or use attention pooling
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)  # single logit for BCEWithLogitsLoss
        )

        self.config = {}

    def forward(self, lffcc, input_values, attention_mask=None):
        input_values.squeeze_(dim=1)
        out = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = out.last_hidden_state  # shape (B, T, 1024)

        # pool across time axis (T)
        pooled = self.pooling(hidden_states, mask=attention_mask)  # (B, D)

        logits = self.classifier(pooled)  # (B, 1)
        return logits.squeeze(-1)  # return (B,) for BCEWithLogitsLoss



class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x, mask=None):
        """
        x: Tensor of shape (B, T, D)
        mask: Optional tensor of shape (B, T), with 1 for valid tokens and 0 for padding
        """
        scores = self.attention(x).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=1)  # (B, T)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, D)
        return pooled
