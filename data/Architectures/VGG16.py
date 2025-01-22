"""******************************************************************
The code is based on : https://github.com/a-nagrani/VGGVox/issues/1
******************************************************************"""

from torch import nn
import constants as const
from constants import *
import torch
from torch import nn
from collections import OrderedDict

class DeepFakeDetection(nn.Module):

    def __init__(self, batch_size, learning_rate, mean, std, dense_layers = 3):

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.numbers_of_dense_layers = dense_layers
        self.input_size = 4096 + 156  # Starting size after convolutional layers
        self.output_size = 1  # Final output size

        self.mean_t = torch.tensor(mean, dtype=torch.float32)
        std_t = torch.tensor(std, dtype=torch.float32)
        self.var_t = std_t ** 2
        assert dense_layers >= 2

        super().__init__()


        self.conv_2d_1 = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(1, 1), padding=1)
        self.bn_1 = nn.BatchNorm2d(96)
        self.max_pool_2d_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1))

        self.conv_2d_2 = nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=1)
        self.bn_2 = nn.BatchNorm2d(256)
        self.max_pool_2d_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1))

        self.conv_2d_3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1)
        self.bn_3 = nn.BatchNorm2d(384)

        self.conv_2d_4 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1)
        self.bn_4 = nn.BatchNorm2d(256)

        self.conv_2d_5 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.bn_5 = nn.BatchNorm2d(256)
        self.max_pool_2d_3 = nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2))

        self.conv_2d_6 = nn.Conv2d(256, 4096, kernel_size=(9, 1), padding=0)
        self.drop_1 = nn.Dropout(p=DROP_OUT)
        self.global_avg_pooling_2d = nn.AdaptiveAvgPool2d((1, 1))

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

    def forward(self, X_Wav2Vec, X_Features):

        if DEBUGMODE:
            print(f"Input shape: {X_Wav2Vec.shape}")
        x = nn.ReLU()(self.conv_2d_1(X_Wav2Vec))
        x = self.bn_1(x)
        x = self.max_pool_2d_1(x)
        if DEBUGMODE:
            print(f"After max_pool_2d_1: {x.shape}")  # Debug shape

        x = nn.ReLU()(self.conv_2d_2(x))
        x = self.bn_2(x)
        x = self.max_pool_2d_2(x)
        if DEBUGMODE:
            print(f"After max_pool_2d_2: {x.shape}")  # Debug shape

        x = nn.ReLU()(self.conv_2d_3(x))
        x = self.bn_3(x)
        if DEBUGMODE:
            print(f"After conv_2d_3: {x.shape}")  # Debug shape

        x = nn.ReLU()(self.conv_2d_4(x))
        x = self.bn_4(x)
        if DEBUGMODE:
            print(f"After conv_2d_4: {x.shape}")  # Debug shape

        x = nn.ReLU()(self.conv_2d_5(x))
        x = self.bn_5(x)
        x = self.max_pool_2d_3(x)
        if DEBUGMODE:
            print(f"After max_pool_2d_3: {x.shape}")  # Debug shape

        x = nn.ReLU()(self.conv_2d_6(x))
        x = self.global_avg_pooling_2d(x)
        if DEBUGMODE:
            print(f"After conv_2d_6: {x.shape}")  # Debug shape

        #flatning the conv to be a vector for the dense layer
        x = torch.flatten(x, 1)  # Correctly flattens to (batch_size, -1)

        #normalizing the incoming Features vector
        X_Features_Normalized = self.features_norm(X_Features)

        #append Xfeatures to the vector
        x = torch.cat((x, X_Features_Normalized), dim=1)

        if DEBUGMODE:
            print(f"After reshape: {x.shape}")

        # Dense layers of the VGG
        x = self.dense_layers(x)

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