import gin
import os
import torch
import torch.nn as nn
from typing import Any
from typing import Callable
from typing import Optional
from torch.utils.data import DataLoader
from dataset.spectrogramDataset import SpectrogramDataset
from model.ctc_wrapper import CTCLoss
from utils.trainingUtils import load_and_split_dataset
from utils.trainingUtils import load_vocabulary
from utils.trainingUtils import set_seed
from utils.modelUtils import token_mask_to_lengths
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from procedures.early_stopping import EarlyStopping
from procedures.ctc_greedy_search import CtcGreedyDecoder
from torchmetrics.text import WordErrorRate
from torchmetrics.text import CharErrorRate

gin.external_configurable(torch.optim.AdamW, module='torch.optim')
gin.external_configurable(torch.optim.lr_scheduler.ReduceLROnPlateau, module='torch.optim.lr_scheduler')
gin.external_configurable(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, module='torch.optim.lr_scheduler')


@gin.configurable
class BaselineTraining:
    def __init__(self,
                 device: str,  # DEVICE CONFIGURATION
                 random_seed: Optional[int],
                 dataset_filepath: str,  # DATASET/DATALOADER PART
                 database_path: str,
                 vocabulary_path: str,
                 validation_split: int,
                 subset_random_state: Any,
                 subset_shuffle: bool,
                 batch_size: int,
                 optimizer: torch.optim,
                 scheduler: torch.optim.lr_scheduler,
                 model: Callable[..., nn.Module],  # MODEL PART
                 model_name: str,
                 num_epochs: int,  # TRAINING PARAMETERS
                 checkpoint_epoch_num: int,
                 results_dir: str,  # RESULTS
                 logging_dir: str) -> None:

        # DEVICE CONFIGURATION AND PROCEDURE INITIALIZATION
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if random_seed:
            set_seed(random_seed)

        # DATASET/DATALOADER PARAMETERS
        self.train_dataset, self.validation_dataset = self._create_subsets(data_feather=dataset_filepath,
                                                                           root_dir=database_path,
                                                                           vocabulary_dir=vocabulary_path,
                                                                           validation_split=validation_split,
                                                                           random_state=subset_random_state,
                                                                           shuffle_subset=subset_shuffle)

        # Prepare Train and Validation loader
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=batch_size,
                                       collate_fn=SpectrogramDataset.spectrogram_collate,
                                       shuffle=True,
                                       drop_last=True)

        self.validation_loader = DataLoader(dataset=self.validation_dataset,
                                            batch_size=batch_size,
                                            collate_fn=SpectrogramDataset.spectrogram_collate,
                                            shuffle=False,
                                            drop_last=False)

        # MODEL
        self.model = model().to(self.device)
        self.model_name = model_name

        # CRITERION AND OPTIMIZER
        self.criterion = CTCLoss()
        self.optimizer = optimizer(params=self.model.parameters())

        # TRAINING LOOP
        self.num_epochs = num_epochs
        self.checkpoint_epoch_num = checkpoint_epoch_num
        self.total_iters = 0
        self.checkpoint = dict()

        # PER-EPOCH ACTIVITY
        self.scheduler = scheduler(optimizer=self.optimizer)

        # RESULTS
        self.models_path = os.path.join(results_dir, 'models', self.model_name)
        os.makedirs(self.models_path, exist_ok=True)

        # VALIDATION
        self.decoder = CtcGreedyDecoder()
        self.wer = WordErrorRate()
        self.cer = CharErrorRate()

        # LOGGING

        # Training metrics
        self.logging_train_dir = os.path.join(logging_dir, 'training-metrics')
        os.makedirs(self.logging_train_dir, exist_ok=True)
        # Train writer
        self.train_writer = SummaryWriter(log_dir=self.logging_train_dir)

        # Validation metrics
        self.logging_validation_dir = os.path.join(logging_dir, 'validation-metrics')
        os.makedirs(self.logging_validation_dir, exist_ok=True)
        # Validation writer
        self.validation_writer = SummaryWriter(log_dir=self.logging_validation_dir)

        self.early_stopping = EarlyStopping(log_path=self.models_path,
                                            model_name=self.model_name)

    @staticmethod
    def _create_subsets(data_feather: str,
                        root_dir: str,
                        vocabulary_dir: str,
                        validation_split: float,
                        random_state: int,
                        shuffle_subset: bool) -> (SpectrogramDataset, SpectrogramDataset):

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

    def _model_weights_histograms(self, step: int) -> None:
        # Iterate over all model parameters
        for name, parameter in self.model.named_parameters():
            # Extract layer name and flattened weights
            tag = name.lower()
            flattened_weights = parameter.data.flatten()

            # Save a histogram of model weights
            self.train_writer.add_histogram(tag=tag, values=flattened_weights, global_step=step, bins='tensorflow')

    def _current_lr(self) -> float:
        for p in self.optimizer.param_groups:
            return p['lr']

    def train_one_epoch(self, epoch: int, progress: tqdm):
        # Set model to training mode
        self.model.train(mode=True)

        # Prepare variables for storing training loss
        running_loss = 0.

        for i, batch in enumerate(self.train_loader):
            # Get samples from single batch
            spectrograms, tokens, padding_mask, token_mask = batch
            # Move spectrograms and tokens tensors to the default device
            spectrograms, tokens, padding_mask, token_mask = \
                spectrograms.to(self.device), tokens.to(self.device), \
                padding_mask.to(self.device), token_mask.to(self.device)

            # Add example image (spectrogram) to TensorBoard for data-monitoring purposes
            if i == 0:
                self.train_writer.add_image("Spec. input", spectrograms[0], global_step=epoch + 1)
                # Add Model-Graph to TensorBoard
                if epoch == 0:
                    self.train_writer.add_graph(model=self.model,
                                                input_to_model=[batch[0].to(self.device),
                                                                batch[2].to(self.device)],
                                                verbose=False)

            # Zero gradients for every batch
            self.optimizer.zero_grad()

            # Feed-forward pass - make predictions for current batch
            outputs, output_lengths = self.model(spectrograms, padding_mask)

            # Calculate target lengths without padding
            target_lengths = token_mask_to_lengths(token_mask=token_mask)

            # Computing CTC loss and its gradients
            loss = self.criterion(outputs, tokens, output_lengths, target_lengths)
            loss.backward()

            # Gradient clipping - clip the gradient norm to given value
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Perform a single optimization step
            self.optimizer.step()

            # Update the running loss - multiply by batch_size to get sum of losses from each sample
            running_loss += loss.item()

            # Update TQDM progress bar with loss metric
            progress.set_postfix(ordered_dict={"train_loss - batch ": loss.item()})

            # Add loss for current step
            self.train_writer.add_scalar(f'Training loss/batch', loss.item(), self.total_iters)
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
        return running_loss

    def validate(self):
        # Set model to evaluation mode
        self.model.eval()

        # Prepare variables for storing validation loss
        validation_loss = 0.
        running_wer = 0.
        running_cer = 0.

        # Turn off the gradients for validation
        with torch.no_grad():
            for batch in self.validation_loader:
                spectrograms, tokens, padding_mask, token_mask = batch
                spectrograms, tokens, padding_mask, token_mask = spectrograms.to(self.device), tokens.to(self.device), \
                    padding_mask.to(self.device), token_mask.to(self.device)

                # Make predictions for current validation batch
                outputs, output_lengths = self.model(spectrograms, padding_mask)

                # Calculate target lengths without padding
                target_lengths = token_mask_to_lengths(token_mask=token_mask.detach())

                # Decode predictions and targets
                decoded_preds, decoded_targets = self.decoder.decode(output=outputs,
                                                                     output_lengths=output_lengths,
                                                                     labels=tokens,
                                                                     label_lengths=target_lengths)

                for i in range(outputs.shape[0]):
                    running_wer += self.wer(decoded_preds[i], decoded_targets[i]).item()
                    running_cer += self.cer(decoded_preds[i], decoded_targets[i]).item()

                # Calculate CTC loss in validation mode
                loss = self.criterion(outputs, tokens, output_lengths, target_lengths)
                validation_loss += loss.detach().item() * spectrograms.size(0)

        return validation_loss, running_wer, running_cer

    def train(self) -> None:
        # Define variables for training
        train_losses, validation_losses = [], []
        validation_wer, validation_cer = [], []

        # Main training loop
        for epoch in range(self.num_epochs):
            print("Epoch {}/{}".format(epoch + 1, self.num_epochs))

            # Add model parameters to TensorBoard-logger (histograms)
            self._model_weights_histograms(step=epoch + 1)

            # Halt training and point to the first place where something went wrong
            torch.autograd.set_detect_anomaly(True)

            # Prepare TQDM for visualization
            with tqdm(total=len(self.train_loader), unit="batch") as progress:
                progress.set_description(desc="Epoch {}, LR {}".format(epoch + 1, self._current_lr()))

                # Train one epoch:
                running_loss = self.train_one_epoch(epoch=epoch, progress=progress)
                train_losses.append(running_loss / len(self.train_loader))

                # Validate:
                validation_loss, wer, cer = self.validate()

                validation_losses.append(validation_loss / len(self.validation_dataset))
                validation_wer.append(wer / len(self.validation_dataset))
                validation_cer.append(cer / len(self.validation_dataset))

                # Turn on the scheduler
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(validation_losses[-1])
                elif isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    self.scheduler.step(epoch + 1)
                else:
                    self.scheduler.step()

                # Save metrics using TensorBoard - create separate scalars for training and validation
                self.train_writer.add_scalar('Avg Loss', train_losses[-1], epoch + 1)

                self.validation_writer.add_scalar('Avg Loss', validation_losses[-1], epoch + 1)
                self.validation_writer.add_scalar('Avg WER', validation_wer[-1], epoch + 1)
                self.validation_writer.add_scalar('Avg CER', validation_cer[-1], epoch + 1)

                # Call Early Stopping
                self.early_stopping(epoch=epoch,
                                    val_loss=validation_losses[-1],
                                    model=self.model,
                                    optimizer=self.optimizer)
                if self.early_stopping.early_stop:
                    print("Early stopping - activate...")
                    break

                # Update TQDM progress bar with per-epoch train and validation loss
                progress.set_postfix({"train_loss ": train_losses[-1],
                                      "val_loss ": validation_losses[-1],
                                      "val_wer": validation_wer[-1]})

        # Close SummaryWriter after training
        self.train_writer.close()
        self.validation_writer.close()
