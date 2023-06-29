# Custom SpecAugment for padded spectrograms

import torch
import random
import numpy as np


class TimeMasking:
    def __init__(self,
                 max_time_mask_percent: float,
                 num_time_masks: int,
                 zero_masking: bool) -> None:
        # Check whether the specified value is within the desired range
        if not 0.0 <= max_time_mask_percent <= 1.0:
            raise ValueError(f"The value of p must be between 0.0 and 1.0 ({max_time_mask_percent} given).")

        self.max_time_mask_percent = max_time_mask_percent  # Maximum possible length of the mask
        self.num_time_masks = num_time_masks  # Number of time masks to apply
        self.zero_masking = zero_masking

    def __call__(self, batch: np.ndarray, spectrograms_widths: np.array) -> np.ndarray:
        return self.time_mask(batch, spectrograms_widths)

    def time_mask(self, spectrograms: np.ndarray, spectrograms_widths: np.array) -> np.ndarray:

        for i in range(spectrograms.shape[0]):
            for _ in range(self.num_time_masks):
                # Define parameters for Time Masking
                time_mask_percent = np.random.uniform(low=0.0, high=self.max_time_mask_percent)
                t = int(time_mask_percent * spectrograms_widths[i])
                t0 = random.randint(0, spectrograms_widths[i] - t)

                # Mask spectrograms
                if self.zero_masking:
                    spectrograms[i, :, :, t0:t0+t] = np.float32(0.0)
                else:
                    spectrograms[i, :, :, t0:t0+t] = np.mean(spectrograms[i, :, :, :])

        return spectrograms


class FrequencyMasking:
    def __init__(self,
                 max_freq_mask_percent: float,
                 num_time_masks: int,
                 zero_masking: bool) -> None:
        # Check whether the specified value is within the desired range
        if not 0.0 <= max_freq_mask_percent <= 1.0:
            raise ValueError(f"The value of p must be between 0.0 and 1.0 ({max_freq_mask_percent} given).")

        self.max_freq_mask_percent = max_freq_mask_percent  # Maximum possible length of the mask
        self.num_time_masks = num_time_masks  # Number of time masks to apply
        self.zero_masking = zero_masking

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        return self.freq_mask(batch)

    def freq_mask(self, spectrograms: np.ndarray) -> np.ndarray:
        # Get spectrograms number of bins
        freq_bins = spectrograms.shape[2]

        for i in range(spectrograms.shape[0]):
            for _ in range(self.num_time_masks):
                # Define parameters for Frequency Masking
                freq_mask_percent = np.random.uniform(low=0.0, high=self.max_freq_mask_percent)
                f = int(freq_mask_percent * freq_bins)
                f0 = random.randint(0, freq_bins - f)

                # Mask spectrograms
                if self.zero_masking:
                    spectrograms[i, :, f0:f0+f, :] = np.float32(0.0)
                else:
                    spectrograms[i, :, f0:f0+f, :] = np.mean(spectrograms[i, :, :, :])

        return spectrograms


class ToTensor:
    def __call__(self, batch: tuple):
        spectrograms, tokens, padding_mask, token_mask = batch
        return torch.from_numpy(spectrograms), torch.from_numpy(tokens), torch.from_numpy(padding_mask), \
            torch.from_numpy(token_mask)

