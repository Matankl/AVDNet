"""******************************************************************
The code is based on : https://github.com/a-nagrani/VGGVox/issues/1
******************************************************************"""

from torch import nn
import constants as const
from constants import *
import torch
from torch import nn
from collections import OrderedDict

class FeaturesOnly(nn.Module):

    def __init__(self, batch_size, learning_rate, mean, std, dense_layers = 3):

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.numbers_of_dense_layers = dense_layers
        self.input_size = 156  # Starting size after convolutional layers
        self.output_size = 1  # Final output size

        self.mean_t = torch.tensor(mean, dtype=torch.float32)
        std_t = torch.tensor(std, dtype=torch.float32)
        self.var_t = std_t ** 2
        assert dense_layers >= 2

        super().__init__()

        # Create BatchNorm
        self.features_norm = nn.BatchNorm1d(
            num_features=self.mean_t.shape[0],
            affine=False,  # no learnable gamma/beta
            track_running_stats=True  # so we can store your stats in running_mean/var
        )
        # Set precomputed mean and variance
        with torch.no_grad():
            self.features_norm.running_mean.copy_(self.mean_t)
            self.features_norm.running_var.copy_(self.var_t)

        self.layer_sizes = [
            int(self.input_size - i * (self.input_size - self.output_size) / (dense_layers - 1))
            for i in range(dense_layers)
        ]

        self.dense_layers = self.build_dynamic_sequential()

    def forward(self, X_Features):

        #normalizing the incoming Features vector
        X_Features_Normalized = self.features_norm(X_Features)

        # Dense layers of the VGG
        x = self.dense_layers(X_Features_Normalized)

        y = nn.Sigmoid()(x)  # Binary classification

        return y

    def build_dynamic_sequential(self, dropout=0.5):
        """
        layer_sizes: e.g. [784, 512, 256, 10]
        dropout: dropout probability
        """
        seq_layers = OrderedDict()
        for i in range(len(self.layer_sizes) - 1):
            in_dim = self.layer_sizes[i]
            out_dim = self.layer_sizes[i + 1]

            # Add a linear layer
            seq_layers[f"linear_{i}"] = nn.Linear(in_dim, out_dim)
            # Add ReLU except on the last layer
            if i < len(self.layer_sizes) - 2:
                seq_layers[f"relu_{i}"] = nn.ReLU()
                seq_layers[f"drop_{i}"] = nn.Dropout(p=dropout)

        return nn.Sequential(seq_layers)

    def get_learning_rate(self):
        return self.learning_rate

    def get_batch_size(self):
        return const.BATCH_SIZE