"""Numerical mask application shared by the backbone's network blocks."""

import torch
from torch import Tensor


def clean(tokens: Tensor, active: Tensor) -> Tensor:
    """Zero inactive values with where, avoiding NaN * 0."""
    return torch.where(active.unsqueeze(-1), tokens, 0.0)
