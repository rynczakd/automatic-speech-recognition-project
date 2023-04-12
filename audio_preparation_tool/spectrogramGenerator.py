# Most of the spectrogram code is taken from: https://timsainburg.com/python-mel-compression-inversion.html
import numpy as np
from numpy.fft import rfft


class SpectrogramGenerator:

    def __init__(self, window_length, step_size, fft_size):
        self.window_length = window_length
        self.step_size = step_size
        self.fft_size = fft_size

    @staticmethod
    def overlap(audio_signal: np.array, window_size: int, step_size: int) -> np.ndarray:

        if window_size % 2 != 0:
            raise ValueError("Window size must be even!")

        fill_zeros = np.zeros((window_size - len(audio_signal) % window_size))
        audio_signal = np.hstack((audio_signal, fill_zeros))

        ws = window_size
        ss = step_size
        x = audio_signal

        valid_range = len(x) - ws
        num_windows = valid_range // ss
        signal_matrix = np.ndarray((num_windows, ws), dtype=x.dtype)

        for i in np.arange(num_windows):
            start = i * ss
            stop = start + ws
            signal_matrix[i] = x[start:stop]

        return signal_matrix

    def stft(self, audio_signal: np.array, window_length: int, fft_size: int, step_size: int,
             real: bool = False, compute_one_sided: bool = True) -> np.ndarray:
        # Real-valued input signals
        if real:
            local_fft = np.fft.rfft
            cut = -1
        # Real-valued and complex-valued input signals
        else:
            local_fft = np.fft.fft
            cut = None

        # Compute only non-negative frequency components
        if compute_one_sided:
            cut = fft_size // 2

        # Create matrix with signal frames
        audio_matrix = self.overlap(audio_signal, window_length, step_size)

        # Multiplying signal values in frames by window function
        size = window_length
        window_function = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
        audio_matrix = audio_matrix * window_function[None]

        # Compute STFT
        audio_matrix = local_fft(audio_matrix, n=fft_size)[:, :cut]

        return audio_matrix

    def log_spectrogram(self, audio_signal: np.array, log: bool = True, threshold: int = 4,
                        periodogram: bool = False) -> np.ndarray:
        # Create spectrogram
        spectrogram = np.abs(self.stft(audio_signal=audio_signal,
                                       window_length=self.window_length,
                                       fft_size=self.fft_size,
                                       step_size=self.step_size,
                                       real=False,
                                       compute_one_sided=True))
        # Periodogram estimate
        if periodogram:
            spectrogram = np.power(spectrogram, 2)
            spectrogram = spectrogram / self.fft_size

        if log:
            spectrogram /= spectrogram.max()
            spectrogram = np.log10(spectrogram)
            spectrogram[spectrogram < -threshold] = -threshold
        else:
            spectrogram[spectrogram < threshold] = threshold

        return spectrogram
