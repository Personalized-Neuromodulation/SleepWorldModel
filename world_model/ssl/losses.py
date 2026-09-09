"""The complete objective: two-view consistency plus one SIGReg term."""

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .sigreg import SIGReg


class SSLLossOutput(NamedTuple):
    total: torch.Tensor
    invariance: torch.Tensor
    sigreg: torch.Tensor


class SSLLoss(nn.Module):
    def __init__(self, sigreg_weight=0.05, num_projections=256, num_frequencies=17):
        super().__init__()
        if not 0 < sigreg_weight < 1:
            raise ValueError("sigreg_weight must be between 0 and 1")
        self.sigreg_weight = sigreg_weight
        self.sigreg = SIGReg(num_projections, num_frequencies)

    def forward(self, output):
        first = output.view1[output.valid].float()
        second = output.view2[output.valid].float()
        if len(first) < 2:
            raise ValueError(
                "SSL needs at least two valid epochs; increase batch size/window length"
            )
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise ValueError("valid projections contain NaN or infinity")
        invariance = F.mse_loss(first, second)
        regularization = (self.sigreg(first) + self.sigreg(second)) / 2
        weight = self.sigreg_weight
        total = (1 - weight) * invariance + weight * regularization
        return SSLLossOutput(total, invariance.detach(), regularization.detach())
