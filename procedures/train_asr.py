import gin
import config
import os
import torch
import torch.nn as nn
from typing import Any
from typing import Callable
from torch.utils.data import DataLoader
from dataset.spectrogramDataset import SpectrogramDataset
from model.ctc_wrapper import CTCLoss
from utils.trainingUtils import load_and_split_dataset
from utils.trainingUtils import load_vocabulary
from utils.trainingUtils import set_seed
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from procedures.early_stopping import EarlyStopping

# TODO: Prepare gin.configurable for whole training process and clean up the code
@gin.configurable
class BaselineTraining:
    def __init__(self,
                 dataset_filepath: str,
                 database_path: str,
                 vocabulary_path: str,
                 validation_split: int,
                 subset_random_state: Any,
                 subset_shuffle: bool,
                 model: Callable[..., nn.Module],
                 model_name: str,
                 results_path: str,
                 batch_size: int,
                 num_epochs: int,
                 device: str,
                 random_seed: bool = True) -> None:

        # DATA
        self.train_dataset, self.validation_dataset = self._create_subsets(data_feather=dataset_filepath,
                                                                           root_dir=database_path,
                                                                           vocabulary_dir=vocabulary_path,
                                                                           validation_split=validation_split,
                                                                           random_state=subset_random_state,
                                                                           shuffle_subset=subset_shuffle)
        # MODEL
        self.model_init = model
        self.model = None
        self.criterion = CTCLoss()
        self.learning_rate = config.LEARNING_RATE
        self.optimizer_init = torch.optim.AdamW
        self.optimizer = None
        self.scheduler_init = torch.optim.lr_scheduler.ReduceLROnPlateau
        self.scheduler = None

        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Device configuration
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # Initialize the procedure
        self.random_seed = random_seed

        if random_seed:
            set_seed(self.random_seed)

        # RESULTS
        self.checkpoint_epoch_num = 50
        self.total_iters = 0
        self.checkpoint = dict()
        self.model_name = model_name
        self.results_dir = config.RESULTS_DIR
        self.models_path = os.path.join(self.results_dir, 'models', self.model_name)
        os.makedirs(self.models_path, exist_ok=True)

        self.writer = SummaryWriter(log_dir=config.LOG_DIR)
        self.early_stopping = EarlyStopping(log_path=self.models_path,
                                            model_name=self.model_name)

    @staticmethod
    def _create_subsets(data_feather: str,
                        root_dir: str,
                        vocabulary_dir: str,
                        validation_split: float = 0.2,
                        random_state: int = None,
                        shuffle_subset: bool = True) -> (SpectrogramDataset, SpectrogramDataset):

        # Load train and validation subset from .feather file:
        train_subset, validation_subset = load_and_split_dataset(data_feather=data_feather,
                                                                 test_size=validation_split,
                                                                 random_state=random_state,
                                                                 shuffle=shuffle_subset)

        # Load vocabulary once to avoid loading it in each subset
        vocabulary = load_vocabulary(vocabulary_dir=vocabulary_dir)

        # Create datasets for training and validation
        train_dataset = SpectrogramDataset(data=train_subset,
                                           root_dir=root_dir,
                                           vocabulary=vocabulary,
                                           spec_augment=True)

        validation_dataset = SpectrogramDataset(data=validation_subset,
                                                root_dir=root_dir,
                                                vocabulary=vocabulary,
                                                spec_augment=False)

        return train_dataset, validation_dataset

    def train(self):
        # Configure model and optimizer
        self.model = self.model_init().to(self.device)
        self.optimizer = self.optimizer_init(params=self.model.parameters(),
                                             lr=config.LEARNING_RATE,
                                             weight_decay=config.WEIGHT_DECAY)
        self.scheduler = self.scheduler_init(optimizer=self.optimizer,
                                             mode="min",
                                             factor=0.1,
                                             patience=10,
                                             verbose=True)

        # Prepare Train and Validation loader
        train_loader = DataLoader(dataset=self.train_dataset,
                                  batch_size=self.batch_size,
                                  collate_fn=SpectrogramDataset.spectrogram_collate,
                                  shuffle=config.SHUFFLE_DATASET)

        validation_loader = DataLoader(dataset=self.validation_dataset,
                                       batch_size=self.batch_size,
                                       collate_fn=SpectrogramDataset.spectrogram_collate,
                                       shuffle=False)

        # Define variables for training
        train_losses, validation_losses = list(), list()

        # Main training loop
        for epoch in range(self.num_epochs):
            print("Epoch {}/{}".format(epoch + 1, self.num_epochs))

            # Halt training and point to the first place where something went wrong
            torch.autograd.set_detect_anomaly(True)

            # Prepare TQDM for visualization
            with tqdm(total=len(train_loader), unit="batch") as progress:
                progress.set_description(desc="Epoch {}".format(epoch + 1))

                # Set model to training mode
                self.model.train(mode=True)

                # Prepare variables for storing training loss
                running_loss = 0.
                validation_loss = 0.

                for i, batch in enumerate(train_loader):
                    # Get samples from single batch
                    spectrograms, tokens, padding_mask, token_mask = batch
                    # Move spectrograms and tokens tensors to the default device
                    spectrograms, tokens = spectrograms.to(self.device), tokens.to(self.device)

                    # Zero gradients for every batch
                    self.optimizer.zero_grad()

                    # Feed-forward pass - make predictions for current batch
                    outputs = self.model(spectrograms, padding_mask)

                    # Computing CTC loss and its gradients
                    loss = self.criterion(outputs, tokens, padding_mask, token_mask)
                    loss.backward()

                    # Gradient clipping - clip the gradient norm to given value
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # Perform a single optimization step
                    self.optimizer.step()

                    # Update the running loss - multiply by batch_size to get sum of losses from each sample
                    running_loss += loss.item() * spectrograms.size(0)

                    # Update TQDM progress bar with loss metric
                    progress.set_postfix(ordered_dict={"train_loss - running ": running_loss})

                    # Add loss for current step
                    self.writer.add_scalar('Training loss', loss.item(), self.total_iters)
                    self.total_iters += 1

                    # Update progress bar
                    progress.update(n=1)

                # Save model after each checkpoint epoch
                if epoch % self.checkpoint_epoch_num == 0:
                    self.checkpoint = {"epoch": epoch,
                                       "model_state": self.model.state_dict(),
                                       "optim_state": self.optimizer.state_dict()}
                    torch.save(self.checkpoint, os.path.join(self.models_path,
                                                             '{}_{}.pt'.format(self.model_name, epoch)))

                # Set model to evaluation mode
                self.model.eval()

                # Turn off the gradients for validation
                with torch.no_grad():

                    for batch in validation_loader:
                        spectrograms, tokens, padding_mask, token_mask = batch
                        spectrograms, tokens = spectrograms.to(self.device), tokens.to(self.device)

                        # Make predictions for current validation batch
                        outputs = self.model(spectrograms, padding_mask)

                        # Calculate CTC loss in validation mode
                        loss = self.criterion(outputs, tokens, padding_mask, token_mask)
                        validation_loss += loss.item() * spectrograms.size(0)

                # Update both train and validation losses
                train_losses.append(running_loss / len(self.train_dataset))
                validation_losses.append(validation_loss / len(self.validation_dataset))

                # Turn on the scheduler
                self.scheduler.step(validation_losses[-1])

                # Call Early Stopping
                self.early_stopping(epoch=epoch,
                                    val_loss=validation_losses[-1],
                                    model=self.model,
                                    optimizer=self.optimizer)
                if self.early_stopping.early_stop:
                    print("Early stopping - activate...")
                    break

                # Update TQDM progress bar with per-epoch train and validation loss
                progress.set_postfix({"train_loss ": train_losses[-1], "val_loss ": validation_losses[-1]})

                # Save metrics using TensorBoard
                self.writer.add_scalars("Training vs. Validation Loss",
                                        {"Training": train_losses[-1],
                                         "Validation": validation_losses[-1]}, epoch + 1)

        # Close SummaryWriter after training
        self.writer.close()
