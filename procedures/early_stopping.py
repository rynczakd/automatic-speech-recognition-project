import os
import torch
import numpy as np


class EarlyStopping:
    def __init__(self, patience, verbose, delta, log_path, model_name):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.models_path = log_path
        self.model_name = model_name

        self.counter = 0
        self.best_score = None
        self.val_loss_min = float(np.Inf)
        self.checkpoint = dict()
        self.early_stop = False

    def __call__(self, epoch, val_loss, model, optimizer):
        # Convert validation loss into score
        score = -val_loss

        # First iteration - set best score as current score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, optimizer)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            # If counter is greater or equal to patience early stopping is performed
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, epoch, val_loss, model, optimizer):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        self.checkpoint = {"epoch": epoch,
                           "model_state": model.state_dict(),
                           "optim_state": optimizer.state_dict()}
        torch.save(self.checkpoint, os.path.join(self.models_path,
                                                 '{}_{}.pt'.format(self.model_name, epoch)))
        self.val_loss_min = val_loss
