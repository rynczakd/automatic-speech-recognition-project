import gin
import torch

from ctc_tokenizer.ctcTokenizer import CtcTokenizer
from utils.trainingUtils import load_decoder


@gin.configurable
class CtcGreedyDecoder:
    def __init__(self,
                 int_to_char_decoder_path: str,
                 blank_label: int = 0,
                 collapse_repeated: bool = True):

        self.blank_label = blank_label
        self.collapse_repeated = collapse_repeated
        self.int_to_char = CtcTokenizer.decoder
        self.int_to_char_dict = load_decoder(decoder_dir=int_to_char_decoder_path)

    def decode(self, output, output_lengths, labels, label_lengths):
        # Prepare empty lists for decodes and targets
        decodes, targets = [], []
        arg_maxes = torch.argmax(input=output, dim=2)

        for i, args in enumerate(arg_maxes):
            decode = []
            args = args[:output_lengths[i]]
            targets.append(self.int_to_char(idx_to_text=self.int_to_char_dict,
                                            labels=labels[i][:label_lengths[i]].tolist()))

            for j, index in enumerate(args):
                if index != self.blank_label:
                    if self.collapse_repeated and j != 0 and index == args[j - 1]:
                        continue
                    decode.append(index.item())

            decodes.append(self.int_to_char(idx_to_text=self.int_to_char_dict, labels=decode))

        return decodes, targets
