"""Small shared readout used for patches, channels and modalities."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..architectures.sequence import clean


class MaskedReadout(nn.Module):
    """Reduce the penultimate axis; fully masked rows produce exact zeros."""

    def __init__(self, dim: int, kind: str, *, score_kind="linear", cosine_scale=2.0):
        super().__init__()
        if kind not in ("mean", "attention"):
            raise ValueError("readout must be mean or attention")
        self.score = nn.Linear(dim, 1, bias=False) if kind == "attention" else None
        if score_kind not in ("linear", "normalized_linear", "cosine"):
            raise ValueError("unknown pooling score_kind")
        if not math.isfinite(cosine_scale) or cosine_scale <= 0:
            raise ValueError("cosine_scale must be finite and positive")
        self.score_kind, self.cosine_scale = score_kind, cosine_scale

    def logits(self, tokens):
        if self.score_kind == "linear":
            return self.score(tokens).squeeze(-1).float()
        if self.score_kind == "normalized_linear":
            return (
                self.score(F.layer_norm(tokens, (tokens.shape[-1],)))
                .squeeze(-1)
                .float()
            )
        # Fixed scale bounds scores even if the query or token norm grows.
        # FP32 avoids low-precision normalization of large residual activations.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            keys = F.normalize(tokens.float(), dim=-1, eps=1e-6)
            query = F.normalize(self.score.weight.float(), dim=-1, eps=1e-6)
            return F.linear(keys, query).squeeze(-1) * self.cosine_scale

    def forward(self, tokens, active, *, return_weights=False):
        tokens = clean(tokens, active)
        if self.score is None:
            weight = active.float()
        else:
            logits = self.logits(tokens)
            logits = logits.masked_fill(~active, torch.finfo(logits.dtype).min)
            weight = logits.softmax(-1) * active
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-12)
        # Batched reduction avoids materializing another full [*, C, D] tensor.
        result = (weight.to(tokens.dtype).unsqueeze(-2) @ tokens).squeeze(-2)
        return (result, weight) if return_weights else result


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
