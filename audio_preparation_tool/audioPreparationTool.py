import os
import math
import utils
import numpy as np
import soundfile as sf
import scipy.signal as sig
from spectrogramGenerator import SpectrogramGenerator


class AudioPreparationTool:

    def __init__(self, filepath: str):
        self.filepath = filepath

    def read_flac(self, preprocess: bool = True) -> dict:
        # Check if the filepath is valid
        if os.path.exists(self.filepath):
            data, sample_rate = sf.read(self.filepath)

            if preprocess:
                data = self.preprocess_audio(data)

            sample = {'data': data, 'fs': sample_rate}

            return sample

        else:
            print('The filepath {} is not valid.'.format(self.filepath))

    @staticmethod
    def preprocess_audio(data: np.array) -> np.array:
        # Remove mean value from loaded signal
        data = data - np.mean(data)
        # Normalize values
        data = data / np.max(np.abs(data))

        # Convert stereo to mono by selecting one channel
        if data.ndim == 2 and data.shape[1] == 2:
            data = np.mean(data, axis=1)

        return data
            
    @staticmethod
    def resample_signal(sample: dict, sampling_frequency: int = 16000) -> dict:
        # Extract data and sample rate from dictionary
        audio_signal = sample['data']
        sample_rate = sample['fs']

        # Prepare up-sample and down-sample factors
        last_common_multiple = (sample_rate * sampling_frequency) / math.gcd(sample_rate, sampling_frequency)
        up_sample_factor = int(last_common_multiple // sample_rate)
        down_sample_factor = int(last_common_multiple // sampling_frequency)

        # Increase number of samples
        audio_up = np.zeros(len(audio_signal) * up_sample_factor)
        audio_up[up_sample_factor // 2::up_sample_factor] = audio_signal

        # Anti-aliasing filter
        alias_filter = sig.firwin(301, cutoff=sampling_frequency / 2, fs=sample_rate * up_sample_factor)
        audio_up = down_sample_factor * sig.filtfilt(alias_filter, 1, audio_up)

        # Decrease number of samples
        audio_down = audio_up[down_sample_factor // 2::down_sample_factor]
        audio_signal = audio_down

        sample = {'data': audio_signal, 'fs': sampling_frequency}

        return sample

    def process_audio_data(self) -> dict:
        # Read .flac file
        sample = self.read_flac(preprocess=True)

        # Resample signal if sample rate is not equal to 16kHz
        if sample['fs'] > 0 and sample['fs'] != 16000:
            sample = self.resample_signal(sample)

        return sample

    @staticmethod
    def compute_spectrogram(sample: dict, window_length: int, overlap: int, log_mel: bool = True) -> np.ndarray:

        # Prepare variables for STFT
        window_size = int(sample['fs'] * window_length / 1000)
        fft_size = utils.first_power_of_2(int(sample['fs'] * window_length / 1000))
        step_size = int(sample['fs'] * overlap / 1000)

        # Create instance of SpectrogramGenerator
        spectrogram_generator = SpectrogramGenerator(window_length=window_size,
                                                     step_size=step_size,
                                                     fft_size=fft_size,
                                                     sample_rate=sample['fs'])

        spectrogram = spectrogram_generator.log_spectrogram(audio_signal=sample['data'],
                                                            log=True,
                                                            threshold=4,
                                                            periodogram=False)
        if log_mel:
            spectrogram = spectrogram_generator.log_mel_spectrogram(spectrogram)

        return spectrogram
