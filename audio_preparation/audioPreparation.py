import os
import gin
import math
import feather
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as sig
from ctc_tokenizer.ctcTokenizer import CtcTokenizer
from audio_preparation.spectrogramGenerator import SpectrogramGenerator
from utils.audioUtils import first_power_of_2
from utils.audioUtils import spec2img


@gin.configurable
class AudioPreparation:

    def __init__(self, root_dir: str, database_dir: str, labels_dir: str, vocabulary_name: str, decoder_name: str,
                 dataset_name: str, dataset_dict_name: str) -> None:
        self.root_dir = root_dir  # Audio-dataset directory
        self.database_dir = database_dir
        self.labels_dir = labels_dir

        # Variables for attribute to be created
        self.vocabulary_name = vocabulary_name
        self.decoder_name = decoder_name
        self.dataset_name = dataset_name
        self.dataset_dict_name = dataset_dict_name

        if not os.path.exists(self.database_dir):
            os.mkdir(self.database_dir)

        if not os.path.exists(self.labels_dir):
            os.mkdir(self.labels_dir)

    @gin.configurable(denylist=['filepath'])
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
        sample = self.read_flac(filepath=filepath)

        # Resample signal if sample rate is not equal to 16kHz
        if sample['fs'] > 0 and sample['fs'] != 16000:
            sample = self.resample_signal(sample)

        return sample

    @staticmethod
    @gin.configurable(denylist=['sample'])
    def compute_spectrogram(sample: dict, window_length: int, overlap: int,
                            log_mel: bool = True, n_freq_components: int = 23) -> np.ndarray:

        # Prepare variables for STFT
        window_size = int(sample['fs'] * window_length / 1000)
        fft_size = first_power_of_2(int(sample['fs'] * window_length / 1000))
        step_size = int(sample['fs'] * overlap / 1000)

        # Create instance of SpectrogramGenerator
        spectrogram_generator = SpectrogramGenerator(window_length=window_size,
                                                     step_size=step_size,
                                                     fft_size=fft_size,
                                                     sample_rate=sample['fs'])

        spectrogram = spectrogram_generator.log_spectrogram(audio_signal=sample['data'],
                                                            threshold=4)
        if log_mel:
            spectrogram = spectrogram_generator.log_mel_spectrogram(spectrogram=spectrogram,
                                                                    num_filters=n_freq_components)

        return spectrogram

    def generate_database(self):
        # Prepare vocabulary for CTC loss function
        print("PREPARING VOCABULARY...")
        ctc_tokenizer = CtcTokenizer(root_dir=self.root_dir)
        vocabulary, ctc_decoder = ctc_tokenizer.prepare_vocabulary()

        # Create vocabulary and decoder dataframe
        vocabulary_dataframe = pd.DataFrame([(letter, index) for letter, index in vocabulary.items()],
                                            columns=['Character', 'Index'])
        decoder_dataframe = pd.DataFrame([(index, letter) for index, letter in ctc_decoder.items()],
                                         columns=['Index', 'Character'])
        # Write dataframe to .feather file
        feather.write_dataframe(vocabulary_dataframe, os.path.join(self.labels_dir, self.vocabulary_name + '.feather'))
        feather.write_dataframe(decoder_dataframe, os.path.join(self.labels_dir, self.decoder_name + '.feather'))

        # Prepare empty dictionary for spectrogram and label pairs
        print("GENERATING DATASET...")
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
                            spectrogram = self.compute_spectrogram(sample=data)

                            # Convert spectrogram into Image object
                            spectrogram = spec2img(spectrogram)
                            spectrogram.save(os.path.join(self.database_dir, filename) + ".png")

                            # PROCESS TRANSCRIPTIONS...
                            transcript = line.split(" ", 1)[1]  # Get transcription from .trans.txt file
                            transcript = transcript.strip().lower().replace("\n", "")

                            # Add spectrogram and corresponding transcription to the dictionary
                            spectrogram_filename = filename + '.png'
                            if spectrogram_filename not in dataset_dictionary:
                                dataset_dictionary[spectrogram_filename] = transcript

                            line = f.readline()

        print("SAVING DATASET...")
        # Convert dictionary into pandas dataframe
        dataset_dataframe = pd.DataFrame([(spec, token) for spec, token in dataset_dictionary.items()],
                                         columns=['Spectrogram', 'Transcription'])
        # Save the DataFrame to a CSV file
        dataset_dataframe.to_csv(os.path.join(self.labels_dir, self.dataset_name + '.csv'), index=False)

        # Save the DataFrame to a feather file
        feather.write_dataframe(dataset_dataframe, os.path.join(self.labels_dir, self.dataset_name + '.feather'))

        # Save dictionary as .npy file:
        np.save(os.path.join(self.labels_dir, self.dataset_dict_name + '.npy'), dataset_dictionary)
