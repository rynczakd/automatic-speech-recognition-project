import torch
from typing import List
from typing import Tuple
from ctc_tokenizer.ctcTokenizer import CtcTokenizer
from procedures.ctc_beam_search import CtcBeamSearch
from utils.trainingUtils import load_decoder


class Decoder:
    def __init__(self,
                 int_to_char_decoder_path,
                 beam_size,
                 blank_idx):

        self.beam_search = CtcBeamSearch(beam_size=beam_size, blank_idx=blank_idx)
        self.int_to_char = CtcTokenizer.decoder
        self.int_to_char_dict = load_decoder(decoder_dir=int_to_char_decoder_path)

    def __call__(self,
                 preds: torch.Tensor,
                 preds_lengths: torch.Tensor,
                 targets: torch.Tensor,
                 target_lengths: torch.Tensor):
        return self.decode(preds=preds, preds_lengths=preds_lengths, targets=targets, target_lengths=target_lengths)

    def decode_batch(self, probs_batch: torch.Tensor, probs_lengths: torch.Tensor) -> List[tuple]:
        # Calculate LogSoftmax for CTC Beam Search function
        probs_batch = probs_batch.log_softmax(dim=-1)
        probs_batch = probs_batch.detach().numpy()

        # Prepare empty list for decodes (indexes, log-likelihood)
        decoded_batch = []

        # Iterate over predictions in batch (without padding)
        for probs, length in zip(probs_batch, probs_lengths):
            decoded, log_sum = self.beam_search.decode(probs=probs[:length], calculate_log=False)
            decoded = self.int_to_char(idx_to_text=self.int_to_char_dict, labels=decoded)

            decoded_batch.append((decoded, log_sum))

        return decoded_batch  # List of tuples with length equal to batch size

    def decode_targets(self, targets: torch.Tensor, target_lengths: torch.Tensor) -> List[str]:
        targets = targets.detach().numpy()

        # Prepare empty list for target decodes
        decoded_targets = []

        for token, length in zip(targets, target_lengths):
            decoded = self.int_to_char(idx_to_text=self.int_to_char_dict, labels=token[:length])
            decoded_targets.append(decoded)

        return decoded_targets

    def decode(self,
               preds: torch.Tensor,
               preds_lengths: torch.Tensor,
               targets: torch.Tensor,
               target_lengths: torch.Tensor) -> Tuple[List[Tuple], List[str]]:

        decoded_preds = self.decode_batch(probs_batch=preds, probs_lengths=preds_lengths)
        decoded_tokens = self.decode_targets(targets=targets, target_lengths=target_lengths)

        return decoded_preds, decoded_tokens
