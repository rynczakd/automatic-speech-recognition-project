import gin
import torch
from torch import nn
import torch.nn.functional as F


@gin.configurable
class FeatureExtractor(nn.Module):
    def __init__(self, reduce_mean: bool = False):
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

        self.conv2_bn = nn.BatchNorm2d(128)

        # CONV3
        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))

        self.conv3_bn = nn.BatchNorm2d(256)

        # CONV4
        self.conv4 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))

        if not self.reduce_mean:
            # LINEAR LAYERS
            self.dense = self._create_fully_connected()

    @staticmethod
    @gin.configurable
    def _create_fully_connected(input_size: int = 16896,
                                output_dim: int = 512,
                                hidden_size: int = 4096,
                                num_layers: int = 3,
                                dropout: float = 0.2) -> nn.Module:
        assert num_layers >= 1, "Number of layers must be greater than or equal to 1"

        if num_layers == 1:
            layers = [nn.Linear(in_features=input_size, out_features=output_dim),
                      nn.LayerNorm(normalized_shape=output_dim),
                      nn.ReLU()]

            return nn.Sequential(*layers)

        layers = [nn.Linear(in_features=input_size, out_features=hidden_size),
                  nn.LayerNorm(normalized_shape=hidden_size),
                  nn.ReLU(),
                  nn.Dropout(p=dropout)]

        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(in_features=hidden_size, out_features=hidden_size),
                           nn.LayerNorm(normalized_shape=hidden_size),
                           nn.ReLU(),
                           nn.Dropout(p=dropout)])

        layers.extend([nn.Linear(in_features=hidden_size, out_features=output_dim),
                       nn.LayerNorm(normalized_shape=output_dim),
                       nn.ReLU()])

        return nn.Sequential(*layers)

    def forward(self, x):
        # Define forward method for Feature Extractor
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4(x))

        # Perform averaging of feature maps over Channel dimension
        if self.reduce_mean:
            x = torch.mean(x, dim=1, keepdim=True)
            x = x.squeeze(1)  # Remove Channel dimension
            x = x.permute(0, 2, 1).contiguous()  # Reshape feature maps into (B, W, H)

        else:
            x = x.permute(0, 3, 1, 2).contiguous()  # Reshape feature maps into (B, W, C, H)
            x = x.view(x.size(0), x.size(1), -1)
            x = self.dense(x)

        return x  # Feature maps dimension - (B, W, H)
