import os
import pandas as pd
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Any, List, Optional, Tuple
from ctc_tokenizer.ctcTokenizer import CtcTokenizer
from utils.audioUtils import img2spec
from utils.datasetUtils import pad_and_sort_batch


class SpectrogramDataset(Dataset):

    def __init__(self,
                 data_feather: str,
                 root_dir: str,
                 spectrogram_column: str,
                 transcription_column: str,
                 vocabulary_dir: str,
                 transform: Optional[List] = None) -> None:
        # Data loading
        self.data = pd.read_feather(data_feather)
        self.root_dir = root_dir
        self.spectrograms = self.data[spectrogram_column]
        self.transcriptions = self.data[transcription_column]
        self.ctc_vocabulary = pd.read_feather(vocabulary_dir).set_index('Character')['Index'].to_dict()
        self.transform = transforms.Compose(transform) if transform else nn.Identity()

    def __getitem__(self, item: int) -> Tuple[str, Any, Any, nn.Module]:
        spectrogram_path = os.path.join(self.root_dir, self.spectrograms.iloc[item])
        # Return sample from the dataset
        return spectrogram_path, self.transcriptions.iloc[item], self.ctc_vocabulary, self.transform

    def __len__(self):
        # Return length of the dataset
        return len(self.spectrograms)

    @staticmethod
    def spectrogram_collate(batch: List, convert_to_array: bool = True):
        spectrograms_path, transcripts = list(), list()
        # Iterate over samples in batch
        for sample in batch:
            spectrograms_path.append(sample[0])
            transcripts.append(sample[1])
        vocabulary = batch[0][2]
        transform = batch[0][3]

        loaded_samples = list()
        # Iterate over spectrograms paths in batch and load images
        for path in spectrograms_path:
            spectrogram = Image.open(str(path))
            # Convert image to numpy.ndarray
            if convert_to_array:
                spectrogram = img2spec(spectrogram)

            loaded_samples.append(spectrogram)

        tokens = list()
        # Iterate over transcriptions in batch and convert them into tokens for CTC loss
        for transcript in transcripts:
            token = CtcTokenizer.tokenizer(vocabulary=vocabulary, sentence=transcript)
            tokens.append(token)

        spectrograms, tokens, padding_mask, token_mask, sequence_lengths, token_lengths = \
            pad_and_sort_batch(batch=loaded_samples,
                               tokens=tokens)

        return spectrograms, tokens, padding_mask, token_mask, sequence_lengths, token_lengths
