import torch


def mask_to_lengths(mask: torch.Tensor) -> torch.Tensor:
    # Convert padding mask to sample lengths
    mask_mean = torch.mean(mask, dim=2)
    return ((1 - mask_mean).sum(-1)).type(torch.int)
