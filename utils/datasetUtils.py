# datasetUtils.py

import torch
import numpy as np
from typing import List


def sort_batch(batch: torch.Tensor,
               tokens: torch.Tensor,
               padding_mask: torch.Tensor,
               token_mask: torch.Tensor,
               widths: torch.Tensor):
    # Sort samples by the width of the spectrograms with the longest sequence first
    sequence_lengths, permutation_index = widths.sort(dim=0, descending=True)
    # Sort spectrogram batch and token batch
    spectrograms_tensor = batch[permutation_index]
    tokens_tensor = tokens[permutation_index]
    padding_mask_tensor = padding_mask[permutation_index]
    token_mask_tensor = token_mask[permutation_index]

    return spectrograms_tensor, tokens_tensor, padding_mask_tensor, token_mask_tensor, sequence_lengths


def pad_and_sort_batch(batch: List, tokens: List, batch_value: np.float64 = 0.0, token_value: int = 99):
    # Define spectrogram dimensions
    spectrogram_channel, spectrogram_height = batch[0].shape[0], batch[0].shape[1]
    spectrogram_widths = [sample.shape[2] for sample in batch]
    max_spectrogram_width = np.max(spectrogram_widths)

    # Define output shape for spectrograms in batch
    output_shape = (len(batch), spectrogram_channel, spectrogram_height, max_spectrogram_width)

    # Prepare arrays for padded samples and padding mask
    padded_samples = np.ones(output_shape, dtype=np.float64) * batch_value
    padding_mask = np.ones(output_shape, dtype=np.float64)

    for i, spectrogram in enumerate(batch):
        padded_samples[i, :, :, :spectrogram.shape[2]] = spectrogram
        padding_mask[i, :, :, :spectrogram.shape[2]] = np.float64(0)

    # Determine maximum token length
    max_token_length = max([len(token) for token in tokens])

    # Define output shape for tokens in batch
    token_output_shape = (len(batch), max_token_length)

    # Prepare arrays for padded tokens and token mask
    padded_tokens = np.ones(token_output_shape, dtype=int) * token_value
    token_mask = np.ones(token_output_shape, dtype=int)

    for i, token in enumerate(tokens):
        padded_tokens[i, :len(token)] = token
        token_mask[i, :len(token)] = int(0)

    return padded_samples, padding_mask, padded_tokens, token_mask

    # return sort_batch(batch=torch.tensor(padded_samples),
    #                   tokens=torch.tensor(padded_tokens),
    #                   padding_mask=torch.tensor(padding_mask),
    #                   token_mask=torch.tensor(token_mask),
    #                   widths=torch.tensor(spectrogram_widths))
