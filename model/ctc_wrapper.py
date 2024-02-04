import gin
import torch
from torch import nn
import torch.nn.functional as F


@gin.configurable
class CTCLoss(nn.Module):
    def __init__(self, blank: int = 0, pack_predictions: bool = False) -> None:
        super(CTCLoss, self).__init__()
        # Initialize CTC Loss parameters:
        self.blank = blank
        self.pack_predictions = pack_predictions

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                predictions_lengths: torch.Tensor,
                target_lengths: torch.Tensor) -> torch.Tensor:

        # Calculate LogSoftmax for CTC Loss function
        predictions = predictions.log_softmax(dim=-1)

        # Permute from (batch, sequence_length, num_classes) to (sequence_length, batch, num_classes)
        batch, seq_length, classes = predictions.shape
        predictions = predictions.permute(1, 0, 2)  # (seq_length, batch, num_classes)

        if self.pack_predictions or predictions_lengths is None:
            predictions_lengths = torch.full(size=(batch, ), fill_value=seq_length, dtype=torch.int32)

        else:
            predictions_lengths = predictions_lengths.type(torch.int32)

        target_lengths = target_lengths.type(torch.int32)

        return F.ctc_loss(log_probs=predictions,
                          targets=targets,
                          input_lengths=predictions_lengths,
                          target_lengths=target_lengths,
                          blank=self.blank,
                          zero_infinity=True)
    