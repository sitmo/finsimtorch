"""PyTorch utility functions."""

import torch


def get_best_device() -> torch.device:
    """
    Get the best available PyTorch device.
    
    Returns
    -------
    torch.device
        CUDA device if available, otherwise CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
