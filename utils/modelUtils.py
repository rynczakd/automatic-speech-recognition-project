import torch
from model.feature_extractor import FeatureExtractor


def mask_to_lengths(mask: torch.Tensor) -> torch.Tensor:
    # Convert padding mask to sample lengths
    mask_mean = torch.mean(mask, dim=2)
    return ((1 - mask_mean).sum(-1)).type(torch.int)


def get_feature_extractor_parameters(reduce_mean: bool = False) -> dict:
    # Instantiate feature extractor:
    feature_extractor = FeatureExtractor(reduce_mean=reduce_mean)

    # Create dictionary to store the parameters
    parameters = dict()

    for name, module in feature_extractor.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.MaxPool2d)):
            parameters[name] = {
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding
            }

    return parameters


def get_conv_output_widths(input_widths: torch.Tensor, conv_parameters: dict = None):
    # Load feature extractor parameters
    conf_cfg_dict = conv_parameters if conv_parameters else get_feature_extractor_parameters()

    for conv_cfg in conf_cfg_dict.values():
        kernel_size, stride, padding = conv_cfg['kernel_size'], conv_cfg['stride'], conv_cfg['padding']
        input_widths = torch.floor(((input_widths + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1]) + 1)

    return input_widths.to(torch.float64)
