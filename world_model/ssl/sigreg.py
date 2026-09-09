"""Sliced Gaussian characteristic-function matching (SIGReg).

Reference: https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md
Quadrature uses the positive half of the symmetric frequency domain.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SIGReg(nn.Module):
    def __init__(self, num_projections=256, num_frequencies=17, max_frequency=3.0):
        super().__init__()
        if num_projections < 1 or num_frequencies < 2:
            raise ValueError("need at least one projection and two frequencies")
        if not math.isfinite(max_frequency) or max_frequency <= 0:
            raise ValueError("max_frequency must be finite and positive")
        self.num_projections = num_projections
        frequencies = torch.linspace(0, max_frequency, num_frequencies)
        self.register_buffer("frequencies", frequencies)
        self.register_buffer("gaussian_cf", torch.exp(-frequencies.square() / 2))

    def forward(self, embeddings, valid=None):
        """A scalar for [N,D]; remove masked rows before doing any arithmetic."""
        if embeddings.ndim != 2 or embeddings.shape[1] < 1:
            raise ValueError("embeddings must have shape [samples, dimensions]")
        if valid is not None:
            if valid.dtype != torch.bool or valid.shape != embeddings.shape[:1]:
                raise ValueError("valid must be a boolean mask with shape [samples]")
            embeddings = embeddings[valid]
        if len(embeddings) < 2:
            raise ValueError("SIGReg needs at least two valid samples")
        if not torch.isfinite(embeddings).all():
            raise ValueError("valid embeddings contain NaN or infinity")
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            values = embeddings.float()
            directions = torch.randn(
                values.shape[1],
                self.num_projections,
                device=values.device,
                dtype=torch.float32,
            )
            directions = F.normalize(directions, dim=0)
            frequencies = self.frequencies.to(values.device, torch.float32)
            gaussian = self.gaussian_cf.to(values.device, torch.float32)
            phase = (values @ directions).unsqueeze(-1) * frequencies
            real = phase.cos().mean(dim=0)
            imaginary = phase.sin().mean(dim=0)
            error = (real - gaussian).square() + imaginary.square()
            integral = 2 * torch.trapezoid(error * gaussian, frequencies, dim=-1)
            return len(values) * integral.mean()
