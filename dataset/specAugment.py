# Implementation of SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
# Ref: https://arxiv.org/pdf/1904.08779.pdf
# Ref: https://github.com/pyyush/SpecAugment

import gin
import torch
import random
import numpy as np


@gin.configurable
class TimeMasking:
    def __init__(self,
                 max_time_mask: int,
                 num_time_masks: int,
                 zero_masking: bool) -> None:

        self.max_time_mask = max_time_mask  # Maximum possible length of the mask
        self.num_time_masks = num_time_masks  # Number of time masks to apply
        self.zero_masking = zero_masking

    def __call__(self, batch: np.ndarray, spectrograms_widths: np.array) -> np.ndarray:
        return self.time_mask(batch, spectrograms_widths)

    def time_mask(self, spectrograms: np.ndarray, spectrograms_widths: np.array) -> np.ndarray:

        for i in range(spectrograms.shape[0]):

            # Check whether the specified value is within the desired range
            if not 0 <= self.max_time_mask <= spectrograms_widths[i]:
                raise ValueError(f"Time Mask parameter must not be greater than the length of the spectrogram "
                                 f"({self.max_time_mask} given).")

            for _ in range(self.num_time_masks):
                # Define parameters for Time Masking
                t = int(np.random.uniform(0, self.max_time_mask))
                t0 = random.randint(0, spectrograms_widths[i] - t)

                # Mask spectrograms
                if self.zero_masking:
                    spectrograms[i, :, :, t0:t0 + t] = np.float64(0.0)
                else:
                    spectrograms[i, :, :, t0:t0 + t] = np.mean(spectrograms[i, :, :, :])

        return spectrograms


@gin.configurable
class FrequencyMasking:
    def __init__(self,
                 max_freq_mask: int,
                 num_time_masks: int,
                 zero_masking: bool) -> None:

        self.max_freq_mask = max_freq_mask  # Maximum possible length of the mask
        self.num_time_masks = num_time_masks  # Number of time masks to apply
        self.zero_masking = zero_masking

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        return self.freq_mask(batch)

    def freq_mask(self, spectrograms: np.ndarray) -> np.ndarray:
        # Get spectrograms number of bins
        freq_bins = spectrograms.shape[2]

        # Check whether the specified value is within the desired range
        if not 0 <= self.max_freq_mask <= freq_bins:
            raise ValueError(f"Frequency Mask parameter must not be greater than the width of the spectrogram "
                             f"({self.max_freq_mask} given).")

        for i in range(spectrograms.shape[0]):
            for _ in range(self.num_time_masks):
                # Define parameters for Frequency Masking
                f = int(np.random.uniform(0, self.max_freq_mask))
                f0 = random.randint(0, freq_bins - f)

                # Mask spectrograms
                if self.zero_masking:
                    spectrograms[i, :, f0:f0 + f, :] = np.float64(0.0)
                else:
                    spectrograms[i, :, f0:f0 + f, :] = np.mean(spectrograms[i, :, :, :])

        return spectrograms


class ToTensor:
    def __call__(self, batch: tuple):
        spectrograms, tokens, padding_mask, token_mask = batch
        return torch.from_numpy(spectrograms), torch.from_numpy(tokens), torch.from_numpy(padding_mask), \
            torch.from_numpy(token_mask)
