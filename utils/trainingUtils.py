# trainingUtils.py

import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter


def load_and_split_dataset(data_feather: str,
                           test_size: float,
                           random_state: int,
                           shuffle: bool) -> (pd.DataFrame, pd.DataFrame):
    # Load data from .feather file
    data = pd.read_feather(data_feather)

    # Split data into training and validation subset
    data_train, data_validation = train_test_split(data,
                                                   test_size=test_size,
                                                   random_state=random_state,
                                                   shuffle=shuffle)

    return data_train, data_validation


def load_vocabulary(vocabulary_dir: str) -> dict:
    # Return vocabulary as dict
    return pd.read_feather(vocabulary_dir).set_index('Character')['Index'].to_dict()


def set_seed(seed: int = 42) -> None:
    # Set fixed seed for Python, Numpy and PyTorch for reproducibility of experiments
    random.seed(seed)  # Python seed
    np.random.seed(seed)  # NumPy
    torch.random.manual_seed(seed)  # PyTorch


def model_weights_histograms(writer: SummaryWriter, step: int, model: torch.nn.Module) -> None:
    # Iterate over all model parameters
    for name, parameter in model.named_parameters():
        # Extract layer name and flattened weights
        tag = name.lower()
        flattened_weights = parameter.data.flatten()

        # Save a histogram of model weights
        writer.add_histogram(tag=tag, values=flattened_weights, global_step=step, bins='tensorflow')
