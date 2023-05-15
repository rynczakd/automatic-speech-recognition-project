import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset

from typing import Any, List, Optional, Tuple

from utils.audioUtils import img2spec


class SpectrogramDataset(Dataset):

    def __init__(self,
                 data_feather: str,
                 root_dir: str,
                 spectrogram_column: str,
                 token_column: str,
                 transforms: Optional[List] = None) -> None:
        # Data loading
        self.data = pd.read_feather(data_feather)
        self.spectrograms = self.data[spectrogram_column]
        self.tokens = self.data[token_column]

        self.root_dir = root_dir
        self.transforms = transforms

    def __getitem__(self, item: int) -> Tuple[str, Any, Optional[list]]:
        spectrogram_path = os.path.join(self.root_dir, self.spectrograms.iloc[item])

        return spectrogram_path, self.tokens.iloc[item], self.transforms

    def __len__(self):
        return len(self.spectrograms)

    @staticmethod
    def spectrogram_collate(batch: List, convert_to_array: bool = True):
        spectrograms_path, tokens = list(), list()
        # Iterate over samples in batch
        for sample in batch:
            spectrograms_path.append(sample[0])
            tokens.append(sample[1])

        loaded_samples = list()
        # Iterate over spectrograms paths in batch and load images
        for path in spectrograms_path:
            spectrogram = Image.open(path)
            # Convert image to numpy.ndarray
            if convert_to_array:
                spectrogram = img2spec(spectrogram)

            loaded_samples.append(spectrogram)



