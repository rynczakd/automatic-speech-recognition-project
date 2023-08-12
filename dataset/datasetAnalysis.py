import os
import sys
import gin
import feather
import numpy as np
import pandas as pd
from PIL import Image
from ctc_tokenizer.ctcTokenizer import CtcTokenizer
from utils.trainingUtils import load_vocabulary


@gin.configurable
class DatasetAnalysis:

    def __init__(self, data_path, labels_path, dataset_name, vocabulary_path, cleaned_dataset_name,
                 spec_scaling_factor, use_cudnn):
        self.data_path = data_path
        self.labels_path = labels_path
        self.dataset_name = dataset_name
        self.vocabulary = load_vocabulary(vocabulary_path)
        self.cleaned_dataset_name = cleaned_dataset_name
        self.dataset_path = os.path.join(self.labels_path, self.dataset_name + '.feather')

        # Basing on the PyTorch CTC Loss documentation:
        # In order to use CuDNN, the following must be satisfied:
        # - targets must be in concatenated format,
        # - all input_lengths must be T,
        # - blank = 0,
        # - target_lengths ≤ 256,
        # - the integer arguments must be of dtype torch.int32.
        self.use_cudnn = use_cudnn

        if os.path.exists(self.dataset_path):
            try:
                self.dataset_samples = pd.read_feather(self.dataset_path)
                print("DATASET LOADED SUCCESSFULLY")
            except pd.errors.EmptyDataError:
                print("ERROR: THE DATASET FILE IS EMPTY")
                sys.exit("ERROR WHILE LOADING DATASET")
            except pd.errors.ParserError:
                print("ERROR: UNABLE TO PARSE THE DATASET FILE")
                sys.exit("ERROR WHILE LOADING DATASET")
        else:
            print("ERROR: DATASET FILE IS NOT FOUND")

        # Define maximum sequence length for CuDNN and CPU training
        if self.use_cudnn:
            self.max_target_length = 256
            self.min_spec_width = int(self.max_target_length * spec_scaling_factor * 2)

        else:
            self.min_spec_width = 840
            self.max_spec_width = 1718
            self.max_target_length = int(np.floor(self.min_spec_width / (2 * spec_scaling_factor)))

    def get_spectrogram_and_transcription_size(self) -> dict:
        # Prepare empty dictionary for spectrogram widths and token lengths
        spectrograms_dict = dict()

        # Get width of each spectrogram and length of each token
        for idx in self.dataset_samples.index:
            spec = Image.open(os.path.join(self.data_path, self.dataset_samples['Spectrogram'][idx]))
            spec_width, spec_height = spec.size

            token = CtcTokenizer.tokenizer(vocabulary=self.vocabulary,
                                           sentence=self.dataset_samples['Transcription'][idx])

            if self.dataset_samples['Spectrogram'][idx] not in spectrograms_dict:
                spectrograms_dict[self.dataset_samples['Spectrogram'][idx]] = (spec_width, len(token))

        return spectrograms_dict

    def detect_outliers(self, spectrogram_widths: dict) -> list:
        # Prepare empty list for outliers
        outliers = list()

        # Detect outliers from dataset basing on selected device (GPU/CPU)
        if self.use_cudnn:
            for name, sizes in spectrogram_widths.items():
                if sizes[0] < self.min_spec_width or sizes[1] > self.max_target_length:
                    outliers.append(name)
        else:
            for name, sizes in spectrogram_widths.items():

                if (sizes[0] < self.min_spec_width or sizes[0] > self.max_spec_width) \
                        or sizes[1] > self.max_target_length:
                    outliers.append(name)

        print('Total number of outliers: {}'.format(len(outliers)))

        return outliers

    def remove_outliers(self) -> None:
        spectrogram_widths = self.get_spectrogram_and_transcription_size()
        outliers = self.detect_outliers(spectrogram_widths=spectrogram_widths)

        # Remove outliers from dataset
        clean_dataset = self.dataset_samples[~self.dataset_samples['Spectrogram'].isin(outliers)]

        # Save clean dataset to .feather file
        feather.write_dataframe(clean_dataset, os.path.join(self.labels_path, self.cleaned_dataset_name + '.feather'))
