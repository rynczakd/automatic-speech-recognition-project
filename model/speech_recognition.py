import gin
from typing import Callable
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from utils.modelUtils import mask_to_lengths
from utils.modelUtils import get_conv_output_widths


@gin.configurable
class SpeechRecognition(nn.Module):
    def __init__(self,
                 feature_extractor: Callable[..., nn.Module]) -> None:
        super(SpeechRecognition, self).__init__()

        self.feature_extractor = feature_extractor()
        self.gru = self._create_gru()
        self.ctc_encoder = self._create_classifier()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    @gin.configurable(denylist=['arch'])
    def _create_gru(input_size: int = 512,
                    hidden_size: int = 256,
                    arch: nn.Module = nn.GRU) -> nn.Module:

        return arch(input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    dropout=0.0,
                    bidirectional=True,
                    batch_first=True)

    @staticmethod
    @gin.configurable(denylist=['batch_size'])
    def _init_hidden_state(batch_size: int, random_init: bool = False):
        # Initialize hidden state for GRU network
        if random_init:
            h0 = torch.randn(2, batch_size, 256)
        else:
            h0 = torch.zeros(2, batch_size, 256)

        return h0

    @staticmethod
    @gin.configurable
    def _create_classifier(input_size: int = 512,
                           output_size: int = 29,
                           hidden_size: int = 128,
                           num_layers: int = 2) -> nn.Module:
        assert num_layers >= 1, "Number of layers must be greater than or equal to 1"

        if num_layers == 1:
            return nn.Linear(in_features=input_size, out_features=output_size)

        layers = [nn.Linear(in_features=input_size, out_features=hidden_size),
                  nn.LayerNorm(normalized_shape=hidden_size),
                  nn.GELU()]

        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(in_features=hidden_size, out_features=hidden_size),
                           nn.LayerNorm(normalized_shape=hidden_size),
                           nn.GELU()])

        layers.append(nn.Linear(in_features=hidden_size, out_features=output_size))

        return nn.Sequential(*layers)

    def forward(self, input_data: torch.Tensor, padding_mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # Compute spectrograms lengths without padding
        spectrograms_lengths = mask_to_lengths(mask=padding_mask)

        # Calculate feature lengths after FeatureExtractor for spectrograms without padding
        feature_lengths = get_conv_output_widths(input_widths=spectrograms_lengths)

        # Feed-forward input data through FeatureExtractor
        x = self.feature_extractor(input_data)

        # Prepare data for RNN (GRU) Neural Network
        x = pack_padded_sequence(input=x, lengths=feature_lengths, batch_first=True, enforce_sorted=True)

        # Initialize hidden state for GRU
        h0 = self._init_hidden_state(batch_size=input_data.shape[0], random_init=False).to(self.device)

        # Feed-forward feature maps into GRU Neural Network
        packed_output, _ = self.gru(x, h0)

        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # Feed-forward GRU hidden states into CTC decoder
        x = self.ctc_encoder(output)

        return x
    