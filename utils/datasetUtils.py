# datasetUtils.py

import numpy as np
from typing import List


def sort_batch(batch: np.ndarray,
               tokens: np.ndarray,
               padding_mask: np.ndarray,
               token_mask: np.ndarray,
               widths: np.ndarray,
               token_lengths: np.ndarray):
    # Sort samples by the width of the spectrograms with the longest sequence first
    permutation_index = np.argsort(widths, axis=0)

    # Rearrange spectrogram batch, tokens, padding masks and lengths
    batch = batch[permutation_index]
    tokens = tokens[permutation_index]
    padding_mask = padding_mask[permutation_index]
    token_mask = token_mask[permutation_index]
    widths = widths[permutation_index]
    token_lengths = token_lengths[permutation_index]

    return batch, tokens, padding_mask, token_mask, widths, token_lengths


def pad_and_sort_batch(batch: List, tokens: List, batch_value: np.float64 = 0.0, token_value: int = 99):
    # Define spectrogram dimensions
    spectrogram_channel, spectrogram_height = batch[0].shape[0], batch[0].shape[1]
    spectrogram_widths = np.array([sample.shape[2] for sample in batch])
    max_spectrogram_width = np.max(spectrogram_widths)

    # Define output shape for spectrograms in batch
    output_shape = (len(batch), spectrogram_channel, spectrogram_height, max_spectrogram_width)

    # Prepare arrays for padded samples and padding mask
    padded_samples = np.ones(output_shape, dtype=np.float64) * batch_value
    padding_mask = np.ones(output_shape, dtype=np.float64)

    for i, spectrogram in enumerate(batch):
        padded_samples[i, :, :, :spectrogram.shape[2]] = spectrogram
        padding_mask[i, :, :, :spectrogram.shape[2]] = np.float64(0.0)

    # Determine maximum token length
    token_lengths = np.array([len(token) for token in tokens])
    max_token_length = np.max(token_lengths)

    # Define output shape for tokens in batch
    token_output_shape = (len(batch), max_token_length)

    # Prepare arrays for padded tokens and token mask
    padded_tokens = np.ones(token_output_shape, dtype=int) * token_value
    token_mask = np.ones(token_output_shape, dtype=int)

    for i, token in enumerate(tokens):
        padded_tokens[i, :len(token)] = token
        token_mask[i, :len(token)] = int(0)

    return sort_batch(batch=padded_samples,
                      tokens=padded_tokens,
                      padding_mask=padding_mask,
                      token_mask=token_mask,
                      widths=spectrogram_widths,
                      token_lengths=token_lengths)
