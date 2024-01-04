import gin
import torch
from torch import nn
import numpy as np

@gin.configurable
class FeatureExtractor(nn.Module):
    def __init__(self,
                 feature_extractor: str,
                 input_channels: int,
                 output_channels: int,
                 num_mel_filters: int,
                 reduce_mean: bool = False):
        super(FeatureExtractor, self).__init__()

        # Define input channels and output channels for Feature Extractor
        self.feature_extractor = feature_extractor
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_mel_filters = num_mel_filters

        # Feature embeddings
        if self.feature_extractor == 'vgg-based':
            self.conv_net = nn.Sequential(
                nn.Conv2d(in_channels=self.input_channels,  # CONV1
                          out_channels=64,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),  # MAX POOLING
                             stride=(2, 2),
                             padding=(0, 0)),
                nn.Conv2d(in_channels=64,  # CONV2
                          out_channels=128,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(in_channels=128,  # CONV3
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=(1, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,  # CONV4
                          out_channels=self.output_channels,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU()
            )

        elif self.feature_extractor == 'vgg-cnn':
            self.conv_net = nn.Sequential(
                nn.Conv2d(in_channels=self.input_channels,
                          out_channels=64,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(in_channels=64,
                          out_channels=64,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2),
                             padding=(0, 0)),

                nn.Conv2d(in_channels=64,
                          out_channels=128,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(in_channels=128,
                          out_channels=self.output_channels,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2),
                             padding=(0, 0))
            )

        self.reduce_mean = reduce_mean

        if not self.reduce_mean:
            # LINEAR LAYERS
            self.input_dense = int(self._get_conv_output_height(num_mel_filters=self.num_mel_filters) *
                                   self.output_channels)
            self.dense = self._create_fully_connected(input_size=self.input_dense)

    def _get_feature_extractor_parameters(self) -> dict:
        # Create dictionary to store the parameters
        parameters = dict()

        for name, module in self.conv_net.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.MaxPool2d)):
                parameters[name] = {
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding
                }

        return parameters

    def _get_conv_output_height(self, num_mel_filters: torch.tensor) -> int:
        # Load feature extractor parameters
        conf_cfg_dict = self._get_feature_extractor_parameters()

        for conv_cfg in conf_cfg_dict.values():
            kernel_size, stride, padding = conv_cfg['kernel_size'], conv_cfg['stride'], conv_cfg['padding']
            num_mel_filters = np.floor(((num_mel_filters + 2 * padding[1] - (kernel_size[1] - 1) - 1) /
                                        stride[1]) + 1)

        return num_mel_filters

    @staticmethod
    @gin.configurable(denylist=['input_size'])
    def _create_fully_connected(input_size: int,
                                output_dim: int,
                                hidden_size: int,
                                num_layers: int,
                                dropout: float) -> nn.Module:
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
        if self.feature_extractor == 'vgg-based' or self.feature_extractor == 'vgg-cnn':
            x = self.conv_net(x)

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
