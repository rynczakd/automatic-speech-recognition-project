import torch
from torch import nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        # Define Convolutional and MaxPooling layers for Feature Extractor
        # CONV1
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=64,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))
        # MAX-POOL
        self.pool = nn.MaxPool2d(kernel_size=(2, 2),
                                 stride=(2, 2),
                                 padding=(1, 1))
        # CONV2
        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))
        # CONV3
        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))
        # CONV4
        self.conv4 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))

    def forward(self, x):
        # Define forward method for Feature Extractor
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Perform averaging of feature maps over Channel dimension
        x = torch.mean(x, dim=1, keepdim=True)

        return x
    