# Most of the code was taken from: https://gist.github.com/awni/56369a90d03953e370f3964c826ed4b0
# For more details checkout either of these references:
# - https://arxiv.org/abs/1408.2873
# - https://distill.pub/2017/ctc/#inference

import math
import numpy as np
import collections


class CtcBeamSearch:
    def __init__(self, beam_size, blank_idx):
        self.beam_size = beam_size
        self.blank = blank_idx

        self.NEG_INF = -float("inf")

    def make_new_beam(self) -> collections.defaultdict:
        def initialize_beam() -> tuple:
            return self.NEG_INF, self.NEG_INF
        return collections.defaultdict(initialize_beam)

    def logsumexp(self, *args):
        if all(a == self.NEG_INF for a in args):
            return self.NEG_INF

        a_max = np.max(args)
        lsp = math.log(sum(math.exp(a - a_max) for a in args))

        return a_max + lsp

    def decode(self, probs, calculate_log: bool = False):
        # Get shape of output probabilities
        T, S = probs.shape

        # Calculate logarithm of the output probabilities if the log-softmax function was not used
        probs = probs if not calculate_log else np.log(probs)

        # Initialize the beam with empty sequence and:
        # - Probability of 1 for ending in blank (0.0 in log space)
        # - Probability of 0 for ending in non-blank (-inf in log space)
        beam = [(tuple(), (0.0, self.NEG_INF))]

        # Iterate over time steps
        for t in range(T):
            # Default dictionary to store the next step candidates
            next_beam = self.make_new_beam()

            # Iterate over characters in vocabulary
            for s in range(S):
                # Get single probability value
                p = probs[t, s]

                # Iterate over current beam
                for prefix, (p_b, p_nb) in beam:

                    # If we propose a blank symbol the prefix does not change
                    # We have to update only the probability of ending in blank;
                    # We have to include probability of ending in blank and ending in non-blank symbol
                    # since the blank can appear after the previous blank as well as after any symbol
                    # from the vocabulary - but we do not include blank symbol in prefixes
                    if s == self.blank:
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_b = self.logsumexp(n_p_b, p_b + p, p_nb + p)
                        next_beam[prefix] = (n_p_b, n_p_nb)
                        continue

                    # Extend the prefix by the new character and add it to the beam
                    # Only the probability of not ending in blank gets updated;
                    # We check the case of occurrence of individual characters from the vocabulary except for the blank
                    # - blank symbol was handled previously
                    end_t = prefix[-1] if prefix else None
                    n_prefix = prefix + (s, )
                    n_p_b, n_p_nb = next_beam[n_prefix]

                    # If the current character differs from the last element of the prefix,
                    # we have to consider the situation where the last element of the prefix is
                    # another character or blank;
                    # We have to include probability of ending in blank and ending in non-blank symbol
                    if s != end_t:
                        n_p_nb = self.logsumexp(n_p_nb, p_b + p, p_nb + p)

                    # We do not include the previous probability of not ending in blank if character
                    # is repeated at the end - we include only the probability of ending in blank
                    # The CTC algorithm mergers characters not separated by a blank;
                    # If the current character is the same as the last element of the prefix,
                    # we have to consider the situation where characters are separated with blank
                    else:
                        n_p_nb = self.logsumexp(n_p_nb, p_b + p)

                    next_beam[n_prefix] = (n_p_b, n_p_nb)

                    # If character is repeated at the end we also update the unchanged prefix - merging case
                    if s == end_t:
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_nb = self.logsumexp(n_p_nb, p_nb + p)
                        next_beam[prefix] = (n_p_b, n_p_nb)
            # Sort and trim the beam before moving on to the next time-step
            beam = sorted(next_beam.items(),
                          key=lambda x: self.logsumexp(*x[1]),
                          reverse=True)
            beam = beam[:self.beam_size]

        # Select beam with the highest probabilities
        best = beam[0]

        # Return the output label sequence and the corresponding negative log-likelihood estimated by the decoder
        return best[0], -self.logsumexp(*best[1])
    