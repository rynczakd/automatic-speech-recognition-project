import torch
from torch import nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, reduce_mean: bool = True):
        super(FeatureExtractor, self).__init__()
        self.reduce_mean = reduce_mean

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

        if not self.reduce_mean:
            # LINEAR LAYERS
            self.fc1 = nn.Linear(in_features=6144, out_features=4096)
            self.fc2 = nn.Linear(in_features=4096, out_features=4096)
            self.fc3 = nn.Linear(in_features=4096, out_features=128)

            # DROPOUT LAYER
            self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Define forward method for Feature Extractor
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Perform averaging of feature maps over Channel dimension
        if self.reduce_mean:
            x = torch.mean(x, dim=1, keepdim=True)
            x = x.squeeze(1)  # Remove Channel dimension
            x = x.permute(0, 2, 1).contiguous()  # Reshape feature maps into (B, W, H)

            return x  # Feature maps dimension - (B, W, H)

        else:
            x = x.permute(0, 3, 1, 2).contiguous()  # Reshape feature maps into (B, W, C, H)
            x = x.view(x.size(0), x.size(1), -1)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(F.relu(self.fc2(x)))
            x = F.relu(self.fc3(x))

            return x  # Feature maps dimension - (B, W, H)
    