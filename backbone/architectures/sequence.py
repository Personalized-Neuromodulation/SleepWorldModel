"""Shared mask and dependency operations for sequence architectures.

Numerical states are always zero outside active positions. Dependency intervals
are conservative bounding intervals, not claims of nonzero attention weights.
"""

from dataclasses import replace

import torch
from torch import Tensor

from ..contracts import SequenceState


def clean(tokens: Tensor, active: Tensor) -> Tensor:
    return torch.where(active.unsqueeze(-1), tokens, 0.0)


def update_state(
    query: SequenceState, source: SequenceState, tokens: Tensor, allowed: Tensor
) -> SequenceState:
    """Union source dependencies with the residual query dependencies."""
    edges = allowed & source.active[:, None, :] & query.active[:, :, None]
    intervals = source.context_intervals_ns
    huge = torch.iinfo(torch.int64).max
    starts = torch.where(edges, intervals[:, None, :, 0], huge).amin(-1)
    ends = torch.where(edges, intervals[:, None, :, 1], -1).amax(-1)
    old = query.context_intervals_ns
    starts = torch.minimum(starts, torch.where(old[..., 0] >= 0, old[..., 0], huge))
    ends = torch.maximum(ends, old[..., 1])
    context = torch.stack((starts, ends), -1)
    context = torch.where(query.active[..., None], context, -1)
    unknown = (edges & (source.available_at_ns[:, None, :] < 0)).any(-1) | (
        query.available_at_ns < 0
    )
    available = torch.where(edges, source.available_at_ns[:, None, :], -1).amax(-1)
    available = torch.maximum(available, query.available_at_ns)
    available = torch.where(query.active & ~unknown, available, -1)
    return replace(
        query,
        tokens=clean(tokens, query.active),
        context_intervals_ns=context,
        available_at_ns=available,
    )
