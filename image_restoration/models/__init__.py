import torch
from models.SRCNN import SRCNN

def model_factory(model_config: dict) -> torch.nn.Module:
    """ model factory

    Args:
        model_config (dict): model configuration

    Raises:
        NotImplementedError: model type is not implemented

    Returns:
        torch.nn.Module: model
    """
    model_type = model_config['type']
    
    if model_type == 'SRCNN':
        img_channel = model_config['img_channel']
        c_expand = model_config['c_expand']
        kernel_sizes = model_config['kernel_sizes']
        strides = model_config['strides']
        return SRCNN(img_channel, c_expand, kernel_sizes, strides)
    else:
        raise NotImplementedError(f'{model_type} is not implemented.')
    