# datasetUtils.py

import torch
import numpy as np
from typing import List


def sort_batch(batch: torch.Tensor,
               tokens: torch.Tensor,
               padding_mask: torch.Tensor,
               widths: torch.Tensor):
    # Sort samples by the width of the spectrograms with the longest sequence first
    sequence_lengths, permutation_index = widths.sort(dim=0, descending=True)
    # Sort spectrogram batch and token batch
    spectrograms_tensor = batch[permutation_index]
    tokens_tensor = tokens[permutation_index]
    padding_tensor = padding_mask[permutation_index]

    return spectrograms_tensor, tokens_tensor, padding_tensor, sequence_lengths


def pad_and_sort_batch(batch: List, tokens: List, value: float = 0.0):
    # Define spectrogram dimensions
    spectrogram_channel, spectrogram_height = batch[0].shape[0], batch[0].shape[1]
    spectrogram_widths = [sample.shape[2] for sample in batch]
    max_spectrogram_width = np.max(spectrogram_widths)

    # Define output shape for spectrograms in batch
    output_shape = (len(batch), spectrogram_channel, spectrogram_height, max_spectrogram_width)

    # Prepare empty arrays for padded samples and padding mask
    padded_samples = np.ones(output_shape) * value
    padding_mask = np.ones(output_shape)

    for i, spectrogram in enumerate(batch):
        padded_samples[i, :, :, :spectrogram.shape[2]] = spectrogram
        padding_mask[i, :, :, :spectrogram.shape[2]] = 0

    return sort_batch(batch=torch.tensor(padded_samples),
                      tokens=torch.tensor(tokens),
                      padding_mask=torch.tensor(padding_mask),
                      widths=torch.tensor(spectrogram_widths))
