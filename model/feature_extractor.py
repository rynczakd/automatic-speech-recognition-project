# Feature Extractor model is inspired by VGG-network: https://arxiv.org/pdf/1409.1556.pdf
# Residual Block is inspired by: https://arxiv.org/pdf/1603.05027.pdf
#                                https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/

import gin
import numpy as np
import torch
from torch import nn
from typing import Tuple


class CnnLayerNorm(nn.Module):
    def __init__(self, features_dim: int):
        super(CnnLayerNorm, self).__init__()
        # Define normalization layer for CNN
        self.layer_norm = nn.LayerNorm(features_dim)

    def forward(self, x):
        # Transpose batch to calculate normalization over frequencies
        x = x.transpose(2, 3).contiguous()  # (B, C, W, H)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (B, C, H, W)


class ResnetBlock(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 n_features_dim: int,
                 dropout: float):
        super(ResnetBlock, self).__init__()

        self.conv_block = self.build_conv_block(input_channels=input_channels,
                                                output_channels=output_channels,
                                                n_features_dim=n_features_dim,
                                                dropout=dropout)

    @staticmethod
    def build_conv_block(input_channels: int,
                         output_channels: int,
                         n_features_dim: int,
                         kernel_size: Tuple[int, int] = (3, 3),
                         stride: Tuple[int, int] = (1, 1),
                         dropout: float = 0.1):
        # Prepare empty list for storing Layers
        conv_block = []

        conv_block += [CnnLayerNorm(features_dim=n_features_dim),
                       nn.GELU(),
                       nn.Dropout(p=dropout),
                       nn.Conv2d(in_channels=input_channels,
                                 out_channels=output_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=(kernel_size[0] // 2, kernel_size[1] // 2)),
                       CnnLayerNorm(features_dim=n_features_dim),
                       nn.GELU(),
                       nn.Dropout(p=dropout),
                       nn.Conv2d(in_channels=output_channels,
                                 out_channels=output_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=(np.floor_divide(kernel_size[0], 2), np.floor_divide(kernel_size[1], 2))),
                       ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)

        return out


@gin.configurable
class FeatureExtractor(nn.Module):
    def __init__(self,
                 feature_extractor_type: str,
                 input_channels: int,
                 output_channels: int,
                 num_mel_filters: int = 64,
                 residual_blocks: int = None,
                 reduce_mean: bool = False):
        super(FeatureExtractor, self).__init__()

        # Define input channels and output channels for Feature Extractor
        self.feature_extractor = feature_extractor_type

        # Define Convolutional part of feature extractor
        self.conv_net = None

        # Feature embeddings
        if self.feature_extractor == 'vgg-based':
            self.conv_net = nn.Sequential(
                nn.Conv2d(in_channels=input_channels,  # CONV1
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
                          out_channels=output_channels,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU()
            )

        elif self.feature_extractor == 'vgg-cnn':
            self.conv_net = nn.Sequential(
                nn.Conv2d(in_channels=input_channels,
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
                          out_channels=output_channels,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2),
                             padding=(0, 0))
            )
        elif self.feature_extractor == 'residual-cnn':
            # Prepare empty list for storing Conv-blocks
            conv_layers = []

            conv_layers += [nn.Conv2d(in_channels=input_channels,
                                      out_channels=32,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=(1, 1))]

            residual_blocks = [ResnetBlock(input_channels=32,
                                           output_channels=output_channels,
                                           n_features_dim=32,
                                           dropout=0.1)
                               for _ in range(residual_blocks) if residual_blocks is not None and residual_blocks > 0]

            conv_layers += residual_blocks

            self.conv_net = nn.Sequential(*conv_layers)

        self.reduce_mean = reduce_mean

        if not self.reduce_mean and self.conv_net is not None:
            # LINEAR LAYERS
            input_dense_size = int(self._get_conv_output_height(num_mel_filters=num_mel_filters) * output_channels)
            self.dense = self._create_fully_connected(input_size=input_dense_size)

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
            layers = [nn.Linear(in_features=input_size, out_features=output_dim)]

            return nn.Sequential(*layers)

        layers = [nn.Linear(in_features=input_size, out_features=hidden_size),
                  nn.LayerNorm(normalized_shape=hidden_size),
                  nn.GELU(),
                  nn.Dropout(p=dropout)]

        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(in_features=hidden_size, out_features=hidden_size),
                           nn.LayerNorm(normalized_shape=hidden_size),
                           nn.GELU(),
                           nn.Dropout(p=dropout)])

        layers.extend([nn.Linear(in_features=hidden_size, out_features=output_dim),
                       nn.GELU()])

        return nn.Sequential(*layers)

    def forward(self, x):
        # Define forward method for Feature Extractor
        if self.feature_extractor == 'vgg-based' or self.feature_extractor == 'vgg-cnn' or \
                self.feature_extractor == 'residual-cnn':
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
