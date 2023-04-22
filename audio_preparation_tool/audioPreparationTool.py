import os
import math
import utils
import numpy as np
import soundfile as sf
import scipy.signal as sig
from ctcTokenizer import CtcTokenizer
from spectrogramGenerator import SpectrogramGenerator


class AudioPreparationTool:

    def __init__(self, root_dir: str, database_dir: str, labels_dir: str):
        self.root_dir = root_dir
        self.database_dir = database_dir
        self.labels_dir = labels_dir

    def read_flac(self, filepath: str, preprocess: bool = True) -> dict:
        # Check if the filepath is valid
        if os.path.exists(filepath):
            data, sample_rate = sf.read(filepath)

            if preprocess:
                data = self.preprocess_audio(data)

            sample = {'data': data, 'fs': sample_rate}

            return sample

        else:
            print('The filepath {} is not valid.'.format(filepath))

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

    def process_audio_data(self, filepath: str) -> dict:
        # Read .flac file
        sample = self.read_flac(filepath=filepath,
                                preprocess=True)

        # Resample signal if sample rate is not equal to 16kHz
        if sample['fs'] > 0 and sample['fs'] != 16000:
            sample = self.resample_signal(sample)

        return sample

    @staticmethod
    def compute_spectrogram(sample: dict, window_length: int, overlap: int,
                            log_mel: bool = True, n_freq_components: int = 23) -> np.ndarray:

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
            spectrogram = spectrogram_generator.log_mel_spectrogram(spectrogram=spectrogram,
                                                                    num_filters=n_freq_components)

        return spectrogram

    def generate_database(self):
        # Prepare vocabulary for CTC loss function
        ctc_tokenizer = CtcTokenizer(root_dir=self.root_dir)
        vocabulary, _ = ctc_tokenizer.prepare_vocabulary(remove_punctuation=False)

        # Prepare empty dictionary for spectrogram and label pairs
        dataset_dictionary = dict()
        for sub_dir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".trans.txt"):
                    with open(os.path.join(sub_dir, file), "r") as f:
                        line = f.readline()
                        while line:
                            # PROCESS AUDIO...
                            filename = line.split(" ", 1)[0]  # Get filename from .trans.txt file
                            filepath = os.path.join(sub_dir, filename) + ".flac"

                            # Load and preprocess .flac file
                            data = self.process_audio_data(filepath=filepath)
                            # Generate spectrogram
                            log_mel_spectrogram = self.compute_spectrogram(sample=data,
                                                                           window_length=25,
                                                                           overlap=10,
                                                                           log_mel=True,
                                                                           n_freq_components=23)
                            log_mel_image = utils.spec2img(log_mel_spectrogram)
                            # log_mel_image.save("<filepath>.png>")

                            # PROCESS TRANSCRIPTIONS...
                            transcript = line.split(" ", 1)[1]  # Get transcription from .trans.txt file
                            transcript = transcript.strip().lower().replace("\n", "")
                            token = ctc_tokenizer.tokenizer(vocabulary=vocabulary,
                                                            sentence=transcript)

                            line = f.readline()

                            # TODO:
                            #  - Prepare saving spectrograms to PNG format images in the given directory,
                            #  - Add storing spectrograms and corresponding tokens in the dictionary,
                            #  - Convert dictionary into pandas dataframe
                            #  - Convert pandas dataframe to feather type