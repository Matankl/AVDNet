"""******************************************************************
The code is based on : https://github.com/a-nagrani/VGGVox/issues/1
******************************************************************"""

from torch import nn
import constants as const
from constants import *

class DeepFakeDetection(nn.Module):

    def cal_paddind_shape(self, new_shape, old_shape, kernel_size, stride_size):
        return (stride_size * (new_shape - 1) + kernel_size - old_shape) / 2

    def __init__(self, epochs, batch_size, learning_rate, dense_layers = 3):

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dense_layers = dense_layers

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

        # # Dense Layers
        # self.dense_layers = nn.ModuleList()  # Flexible list of dense layers
        # self.dropouts = nn.ModuleList()  # Flexible list of Dropout layers
        #
        # input_size = 4096  # Starting size after convolutional layers
        # output_size = 1  # Final output size
        #
        # # Calculate evenly decreasing layer sizes
        # layer_sizes = [
        #     int(input_size - i * (input_size - output_size) / (dense_layers - 1))
        #     for i in range(dense_layers)
        # ]
        #
        # Create dense layers and corresponding dropout layers
        # for i in range(len(layer_sizes) - 1):
        #     self.dense_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        #     self.dropouts.append(nn.Dropout(p=DROP_OUT))
        # self.output_layer = nn.Linear(layer_sizes[-1], output_size)

        # self.output_layer = nn.Linear(layer_sizes[-1], output_size)
        self.dense_1 = nn.Linear(4096 + 0, 512)
        self.drop_2 = nn.Dropout(p=DROP_OUT)

        self.dense_2 = nn.Linear(512, 1)

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
        x = self.drop_1(x)
        x = self.global_avg_pooling_2d(x)
        if DEBUGMODE:
            print(f"After conv_2d_6: {x.shape}")  # Debug shape

        #flatning the conv to be a vector for the dense layer
        x = torch.flatten(x, 1)  # Correctly flattens to (batch_size, -1)

        #append Xfeatures to the vector
        # x = torch.cat((x, X_Features), dim=1)

        if DEBUGMODE:
            print(f"After reshape: {x.shape}")

        # # Dense Layers
        # for dense_layer, dropout in zip(self.dense_layers, self.dropouts):
        #     x = nn.ReLU()(dense_layer(x))
        #     x = dropout(x)

        x = nn.ReLU()(self.dense_1(x))

        x = self.drop_2(x)
        x = self.dense_2(x)

        # Final Output Layer
        # x = self.output_layer(x)
        y = nn.Sigmoid()(x)  # Binary classification

        return y

    def get_epochs(self):
        return self.epochs

    def get_learning_rate(self):
        return self.learning_rate

    def get_batch_size(self):
        return const.BATCH_SIZE