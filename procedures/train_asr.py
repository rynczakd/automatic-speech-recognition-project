import config
import torch
from torch.utils.data import DataLoader
from dataset.spectrogramDataset import SpectrogramDataset
from model.speech_recognition import SpeechRecognition
from model.ctc_wrapper import CTCLoss
from utils.trainingUtils import load_and_split_dataset
from utils.trainingUtils import load_vocabulary
from utils.trainingUtils import set_seed


class BaselineTraining:
    def __init__(self, random_seed: bool = True) -> None:  # TODO: Implement results part with TensorBoard access
        # Device configuration
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        # Initialize the procedure
        self.random_seed = random_seed

        # DATA
        self.train_dataset, self.validation_dataset = self._create_subsets(data_feather=config.DATA_FEATHER,
                                                                           root_dir=config.ROOT_DIR,
                                                                           vocabulary_dir=config.VOCABULARY_DIR,
                                                                           validation_split=config.VALIDATION_SPLIT,
                                                                           random_seed=config.RANDOM_SEED,
                                                                           shuffle_subset=config.SHUFFLE_SUBSET)
        self.batch_size = config.BATCH_SIZE
        self.num_epochs = config.EPOCHS

        if random_seed:
            set_seed(self.random_seed)

        # MODEL
        self.model_init = SpeechRecognition
        self.model = SpeechRecognition()
        self.criterion = CTCLoss()
        self.learning_rate = config.LEARNING_RATE
        self.optimizer_init = torch.optim.AdamW
        self.optimizer = None

        # RESULTS
        # TODO: Implement variables for results

    @staticmethod
    def _create_subsets(data_feather: str,
                        root_dir: str,
                        vocabulary_dir: str,
                        validation_split: float = 0.2,
                        random_seed: int = None,
                        shuffle_subset: bool = True) -> (SpectrogramDataset, SpectrogramDataset):

        # Load train and validation subset from .feather file:
        train_subset, validation_subset = load_and_split_dataset(data_feather=data_feather,
                                                                 test_size=validation_split,
                                                                 random_state=random_seed,
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
        self.model = self.model_init().to(self.device)
        self.optimizer = self.optimizer_init(params=self.model.parameters(),
                                             lr=config.LEARNING_RATE,
                                             weight_decay=config.WEIGHT_DECAY)
        # TODO: Implement WARM-UP-SCHEDULER, COSINE-ANNEALING-LR, LEARNING-RATE-DECAY

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
        num_accumulation_steps = 3

        # Main training loop
        for epoch in range(self.num_epochs):
            # Halt training and point to the first place where something went wrong
            torch.autograd.set_detect_anomaly(True)
            print('EPOCH {}: '.format(epoch + 1))

            # Set model to training mode
            self.model.train(mode=True)

            running_loss = 0.
            accumulation_loss = 0.

            for i, batch in enumerate(train_loader):
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

                # Update the running loss
                running_loss += loss.item()

                # Update accumulation loss for its calculating for every n-minibatch
                accumulation_loss += loss.item()
                if i % num_accumulation_steps == 2:
                    accumulation_loss = accumulation_loss / num_accumulation_steps
                    print('  batch {} loss: {}'.format(i + 1, accumulation_loss))
                    accumulation_loss = 0.
            else:
                validation_loss = 0.

                # Turn off the gradients for validation
                with torch.no_grad():
                    # Set model to evaluation mode
                    self.model.eval()

                    for batch in validation_loader:
                        spectrograms, tokens, padding_mask, token_mask = batch
                        spectrograms, tokens = spectrograms.to(self.device), tokens.to(self.device)

                        # Make predictions for current validation batch
                        outputs = self.model(spectrograms, padding_mask)

                        # Calculate CTC loss in validation mode
                        loss = self.criterion(outputs, tokens, padding_mask, token_mask)
                        validation_loss += loss.item()

                # Update both train and validation losses
                train_losses.append(running_loss / len(train_loader))
                validation_losses.append(validation_loss / len(validation_loader))

                print("Epoch: {}/{}.. ".format(epoch + 1, self.num_epochs),
                      "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                      "Validation Loss: {:.3f}.. ".format(validation_losses[-1]))