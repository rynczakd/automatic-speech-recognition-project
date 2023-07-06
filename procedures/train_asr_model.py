import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


class BaselineTraining:
    @staticmethod
    def _create_subset(dataset_size: int,
                       validation_split: float,
                       random_seed: int,
                       shuffle_dataset: bool = True) -> (SubsetRandomSampler, SubsetRandomSampler):

        # Compute indices and size of validation dataset
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))

        # Shuffle dataset:
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_indices, validation_indices = indices[split:], indices[:split]

        # Create sampler for both training and validation steps
        train_sampler = SubsetRandomSampler(train_indices)
        validation_sampler = SubsetRandomSampler(validation_indices)

        return train_sampler, validation_sampler
