import gin
from typing import Callable
from typing import Any
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from utils.modelUtils import mask_to_lengths
from utils.modelUtils import get_conv_output_widths


class NormGRU(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bidirectional: bool = True,
                 layer_norm: bool = True):
        super(NormGRU, self).__init__()

        self.bidirectional = bidirectional
        self.layer_norm = nn.LayerNorm(input_size) if layer_norm else None

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=1,
                          bidirectional=bidirectional,
                          dropout=0.0,
                          bias=True,
                          batch_first=True)

    def forward(self, x, output_lengths, h_0=None):
        # Forward-pass for Normalized GRU-Net
        if self.layer_norm:
            x = self.layer_norm(x)

        # Prepare data for GRU Neural Network
        x = pack_padded_sequence(input=x, lengths=output_lengths.to('cpu'), batch_first=True, enforce_sorted=True)

        # Feed-forward feature maps into GRU Neural Network
        x, h = self.gru(x, h_0)

        # Unpack values
        x, _ = pad_packed_sequence(x, batch_first=True)

        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)

        return x, h


@gin.configurable
class SpeechRecognition(nn.Module):
    def __init__(self,
                 feature_extractor: Callable[..., nn.Module],
                 use_norm_gru: bool,
                 gru_layers: int) -> None:
        super(SpeechRecognition, self).__init__()

        # FEATURE EXTRACTOR
        self.feature_extractor = feature_extractor()
        # GRU
        self.use_norm_gru = use_norm_gru
        self.gru_layers = gru_layers
        self.gru = self._create_gru(gru_num_layers=self.gru_layers)
        # CLASSIFIER
        self.ctc_encoder = self._create_classifier()
        # CONFIGURATION
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @gin.configurable(denylist=['gru_num_layers'])
    def _create_gru(self,
                    input_size: int,
                    hidden_size: int,
                    gru_num_layers: int,
                    bidirectional: bool,
                    gru_dropout: float) -> Any:
        assert gru_num_layers >= 1, "Number of layers must be greater than or equal to 1"

        if self.use_norm_gru:
            gru_layers = [NormGRU(input_size=input_size,
                                  hidden_size=hidden_size,
                                  bidirectional=bidirectional,
                                  layer_norm=True)]

            gru_norm_layers = [NormGRU(input_size=hidden_size,
                                       hidden_size=hidden_size,
                                       bidirectional=bidirectional,
                                       layer_norm=True)
                               for _ in range(gru_num_layers - 1)]

            gru_layers += gru_norm_layers

            return nn.Sequential(*gru_layers)

        else:
            # Return n-layer GRU
            return nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=gru_num_layers,
                          bidirectional=bidirectional,
                          dropout=gru_dropout,
                          bias=True,
                          batch_first=True)

    @staticmethod
    @gin.configurable
    def _create_classifier(input_size: int,
                           output_size: int,
                           hidden_size: int,
                           num_layers: int,
                           classifier_dropout: float) -> nn.Module:
        assert num_layers >= 1, "Number of layers must be greater than or equal to 1"

        if num_layers == 1:
            return nn.Linear(in_features=input_size, out_features=output_size)

        layers = [nn.Linear(in_features=input_size, out_features=hidden_size),
                  nn.GELU(),
                  nn.Dropout(p=classifier_dropout)]

        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(in_features=hidden_size, out_features=hidden_size),
                           nn.GELU(),
                           nn.Dropout(p=classifier_dropout)])

        layers.append(nn.Linear(in_features=hidden_size, out_features=output_size))

        return nn.Sequential(*layers)

    @staticmethod
    @gin.configurable(denylist=['batch_size'])
    def _init_hidden_state(batch_size: int,
                           hidden_size: int,
                           random_init: bool = False,
                           use_bidirectional: bool = True) -> torch.Tensor:
        # Initialize hidden state for GRU network
        num_directions = 2 if use_bidirectional else 1

        if random_init:
            h0 = torch.randn(num_directions, batch_size, hidden_size)
        else:
            h0 = torch.zeros(num_directions, batch_size, hidden_size)

        return h0

    def forward(self, input_data: torch.Tensor, padding_mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # Compute spectrograms lengths without padding
        spectrograms_lengths = mask_to_lengths(mask=padding_mask)

        # Calculate feature lengths after FeatureExtractor for spectrograms without padding
        feature_lengths = get_conv_output_widths(input_widths=spectrograms_lengths)

        # Feed-forward input data through FeatureExtractor
        x = self.feature_extractor(input_data)

        # Initialize hidden state for GRU
        h0 = self._init_hidden_state(batch_size=input_data.shape[0]).to(self.device)

        if self.use_norm_gru:
            # Iterate over GRU layers
            for i, gru in enumerate(self.gru):
                x, _ = gru(x, feature_lengths, h0)

        else:
            # Prepare data for RNN (GRU) Neural Network
            x = pack_padded_sequence(input=x, lengths=feature_lengths.to('cpu'), batch_first=True, enforce_sorted=True)

            # Feed-forward feature maps into GRU Neural Network
            x, _ = self.gru(x, torch.cat([h0] * self.gru_layers, dim=0).to(self.device))

            x, _ = pad_packed_sequence(x, batch_first=True)

        # Feed-forward GRU hidden states into CTC decoder
        x = self.ctc_encoder(x)

        return x
    