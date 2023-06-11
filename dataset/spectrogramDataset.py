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
from dataset.spectrogramAugmentation import TimeMasking, FrequencyMasking, ToTensor


class SpectrogramDataset(Dataset):

    def __init__(self,
                 data_feather: str,
                 root_dir: str,
                 spectrogram_column: str,
                 transcription_column: str,
                 vocabulary_dir: str,
                 transform: Optional[List] = None,
                 spec_augment: Optional[bool] = True) -> None:
        # Data loading
        self.data = pd.read_feather(data_feather)
        self.root_dir = root_dir
        self.spectrograms = self.data[spectrogram_column]
        self.transcriptions = self.data[transcription_column]
        self.ctc_vocabulary = pd.read_feather(vocabulary_dir).set_index('Character')['Index'].to_dict()
        self.transform = transforms.Compose(transform) if transform else nn.Identity()
        self.spec_augment = spec_augment

    def __getitem__(self, item: int) -> Tuple[str, Any, Any, nn.Module, bool]:
        spectrogram_path = os.path.join(self.root_dir, self.spectrograms.iloc[item])
        # Return sample from the dataset
        return spectrogram_path, self.transcriptions.iloc[item], self.ctc_vocabulary, self.transform, self.spec_augment

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
        spec_augment = batch[0][4]

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

        spectrograms, tokens, padding_mask, token_mask, spectrograms_widths = \
            pad_and_sort_batch(batch=loaded_samples,
                               tokens=tokens)

        # Apply custom SpecAugment
        if spec_augment:
            time_masking = TimeMasking(max_time_mask_percent=0.1, num_time_masks=1, zero_masking=True)
            freq_masking = FrequencyMasking(max_freq_mask_percent=0.15, num_time_masks=1, zero_masking=True)

            spectrograms = time_masking(spectrograms, spectrograms_widths)
            spectrograms = freq_masking(spectrograms)

        # Convert numpy arrays to torch Tensor
        to_tensor = ToTensor()
        spectrograms, tokens, padding_mask, token_mask = to_tensor(batch=(spectrograms, tokens, padding_mask,
                                                                          token_mask))

        spectrograms = transform(spectrograms)

        return spectrograms, tokens, padding_mask, token_mask
