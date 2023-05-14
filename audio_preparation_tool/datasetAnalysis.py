import os
import feather
import pandas as pd
from PIL import Image


class DatasetAnalysis:

    def __init__(self, data_path, labels_path, dataset_name):
        self.data_path = data_path
        self.labels_path = labels_path
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(self.labels_path, self.dataset_name)

        if os.path.exists(self.dataset_path):
            # Load dataset samples from .feather file
            self.dataset_samples = pd.read_feather(self.dataset_path)
            # Extract spectrogram names
            self.spectrograms = self.dataset_samples['Spectrogram']

        # Define minimum and maximum sequence length for spectrograms
        self.min_sequence_length = 626
        self.max_sequence_length = 1708

    def get_spectrogram_widths(self) -> dict:
        # Prepare empty dictionary for spectrogram widths
        spectrograms_dict = dict()

        # Get width of each spectrogram
        for spec_name in self.spectrograms:
            spec = Image.open(os.path.join(self.data_path, spec_name))
            width, height = spec.size

            if spec_name not in spectrograms_dict:
                spectrograms_dict[spec_name] = width

        return spectrograms_dict

    def detect_outliers(self, spectrogram_widths: dict) -> list:
        # Prepare empty list for outliers
        iqr_outliers = list()

        # Detect outliers from dataset
        for name, width in spectrogram_widths.items():
            if width < self.min_sequence_length or width > self.max_sequence_length:
                iqr_outliers.append(name)

        print('Total number of outliers: {}'.format(len(iqr_outliers)))

        return iqr_outliers

    def remove_outliers(self) -> None:
        spectrogram_widths = self.get_spectrogram_widths()
        outliers = self.detect_outliers(spectrogram_widths=spectrogram_widths)

        # Remove outliers from dataset
        clean_dataset = self.dataset_samples[~self.dataset_samples['Spectrogram'].isin(outliers)]

        # Save clean dataset to .feather file
        feather.write_dataframe(clean_dataset, os.path.join(self.labels_path, 'clean_dataset.feather'))
