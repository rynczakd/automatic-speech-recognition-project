import torch
from torch import nn
import torch.nn.functional as F
from utils.modelUtils import mask_to_lengths
from utils.modelUtils import get_conv_output_widths
from utils.modelUtils import token_mask_to_lengths


class CTCLoss(nn.Module):
    def __init__(self, blank: int = 0, pack_predictions: bool = True) -> None:
        super(CTCLoss, self).__init__()
        # Initialize CTC Loss parameters:
        self.blank = blank
        self.pack_predictions = pack_predictions

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                predictions_mask: torch.Tensor,
                targets_mask: torch.Tensor) -> torch.Tensor:

        # Calculate LogSoftmax for CTC Loss function
        predictions = predictions.log_softmax(dim=-1)

        # Permute from (batch, sequence_length, num_classes) to (sequence_length, batch, num_classes)
        batch, seq_length, classes = predictions.shape
        predictions = predictions.permute(1, 0, 2)  # (seq_length, batch, num_classes)

        if not self.pack_predictions:
            predictions_lengths = torch.full(size=(batch, ), fill_value=seq_length, dtype=torch.long)

        else:
            spectrograms_lengths = mask_to_lengths(mask=predictions_mask)
            features_lengths = get_conv_output_widths(input_widths=spectrograms_lengths)
            predictions_lengths = features_lengths.long()

        target_lengths = token_mask_to_lengths(targets_mask).long()

        return F.ctc_loss(log_probs=predictions,
                          targets=targets,
                          input_lengths=predictions_lengths,
                          target_lengths=target_lengths,
                          blank=self.blank,
                          zero_infinity=True)
    