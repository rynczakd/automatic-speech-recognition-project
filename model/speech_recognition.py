from torch import nn
from model.feature_extractor import FeatureExtractor


class SpeechRecognition(nn.Module):
    def __init__(self):
        super(SpeechRecognition, self).__init__()

        self.feature_extractor = FeatureExtractor(reduce_mean=False)
        self.gru = self._create_gru()

    @staticmethod
    def _create_gru(input_size: int = 512,
                    hidden_size: int = 256,
                    arch: nn.Module = nn.GRU) -> nn.Module:

        return arch(input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    dropout=0.1,
                    bidirectional=True,
                    batch_first=True)
