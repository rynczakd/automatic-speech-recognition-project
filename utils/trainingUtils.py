import pandas as pd
from sklearn.model_selection import train_test_split


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
    return pd.read_feather(vocabulary_dir).set_index('Character')['Index'].to_dict()





