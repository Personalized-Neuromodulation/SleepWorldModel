"""Small shared readout used for patches, channels and modalities."""

import math

import torch
from torch import nn

from ..architectures.sequence import clean


class MaskedReadout(nn.Module):
    """Reduce the penultimate axis; fully masked rows produce exact zeros."""

    def __init__(self, dim: int, kind: str):
        super().__init__()
        if kind not in ("mean", "attention"):
            raise ValueError("readout must be mean or attention")
        self.score = nn.Linear(dim, 1, bias=False) if kind == "attention" else None

    def forward(self, tokens, active):
        tokens = clean(tokens, active)
        if self.score is None:
            weight = active.float()
        else:
            logits = self.score(tokens).squeeze(-1).float()
            logits = logits.masked_fill(~active, torch.finfo(logits.dtype).min)
            weight = logits.softmax(-1) * active
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-12)
        return (tokens * weight.to(tokens.dtype).unsqueeze(-1)).sum(-2)


def sinusoidal(positions, dim, dtype):
    """Recording-relative seconds or patch-local sample positions → embedding."""
    frequency = torch.exp(
        torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32)
        * (-math.log(10000) / dim)
    )
    angles = positions.float()[..., None] * frequency
    result = torch.stack((angles.sin(), angles.cos()), -1).flatten(-2)[..., :dim]
    return result.to(dtype)


def bounding_context(intervals, active, dim):
    huge = torch.iinfo(torch.int64).max
    start = torch.where(active, intervals[..., 0], huge).amin(dim)
    end = torch.where(active, intervals[..., 1], -1).amax(dim)
    result = torch.stack((start, end), -1)
    return torch.where(active.any(dim)[..., None], result, -1)


def latest_available(available, active, dim):
    unknown = (active & (available < 0)).any(dim)
    latest = torch.where(active, available, -1).amax(dim)
    return torch.where(unknown, -1, latest)
