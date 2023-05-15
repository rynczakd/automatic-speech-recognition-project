# Most of the spectrogram code is taken from: https://timsainburg.com/python-mel-compression-inversion.html
import numpy as np
import scipy.ndimage
from numpy.fft import rfft
from utils.audioUtils import get_filter_bank


class SpectrogramGenerator:

    def __init__(self, window_length, step_size, fft_size, sample_rate):
        self.window_length = window_length
        self.step_size = step_size
        self.fft_size = fft_size
        self.sample_rate = sample_rate

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

    def create_mel_filter(self, n_freq_components: int, lower_freq: int, upper_freq: int) -> (np.ndarray, np.ndarray):

        # Create a filter to convolve with the spectrogram to get ot mel-scale values
        mel_inversion_filter = get_filter_bank(num_filters=n_freq_components,
                                               nfft=self.fft_size,
                                               sample_rate=self.sample_rate,
                                               lower_freq=lower_freq,
                                               upper_freq=upper_freq)

        # Normalize filter
        mel_filter = mel_inversion_filter.T / mel_inversion_filter.sum(axis=1)

        return mel_filter, mel_inversion_filter

    @staticmethod
    def make_mel(spectrogram: np.ndarray, mel_filter: np.ndarray, shorten_factor: int = 1) -> np.ndarray:
        # Compute mel spectrogram as matrix multiplication (TxS):
        mel_spectrogram = np.transpose(mel_filter).dot(np.transpose(spectrogram))
        mel_spectrogram = scipy.ndimage.zoom(
            mel_spectrogram.astype("float32"), [1, 1.0 / shorten_factor]
        ).astype("float16")
        mel_spectrogram = mel_spectrogram[:, 1:-1]  # a little hacky but seemingly needed for clipping

        return mel_spectrogram

    def log_mel_spectrogram(self, spectrogram: np.ndarray, num_filters: int = 23) -> np.ndarray:
        # Generate the mel filters
        mel_filter, mel_inversion_filter = self.create_mel_filter(n_freq_components=num_filters,
                                                                  lower_freq=0,
                                                                  upper_freq=int(self.sample_rate/2))

        # Compute log-mel spectrogram
        mel_spectrogram = self.make_mel(spectrogram, mel_filter, shorten_factor=1)

        return mel_spectrogram
