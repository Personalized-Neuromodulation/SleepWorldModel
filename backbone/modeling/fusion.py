"""Fuse modalities on an explicitly identical patch time grid.

Cross-attention is per time point: one pooled query attends to the modality
tokens at that same time. Asynchronous resampling is deliberately not implicit.
"""

import torch
from torch import nn

from ..contracts import SequenceState, TokenSequence
from .pooling import MaskedReadout, bounding_context, latest_available


class Fusion(nn.Module):
    def __init__(
        self, names: tuple[str, ...], dim: int, identity: bool, cross_attention=None
    ):
        super().__init__()
        self.names = names
        self.identity = nn.Embedding(len(names), dim) if identity else None
        self.readout = MaskedReadout(dim, "mean")
        self.cross_attention = cross_attention

    def forward(self, features: dict[str, TokenSequence]) -> TokenSequence:
        values = [features[name] for name in self.names]
        first = values[0]
        for value in values[1:]:
            if not torch.equal(value.time_intervals_ns, first.time_intervals_ns):
                raise ValueError(
                    "fusion requires identical time grids; align in adapter first"
                )
        tokens = torch.stack([v.tokens for v in values], 2)  # [B,N,G,D]
        active = torch.stack([v.active for v in values], 2)
        if self.identity is not None:
            tokens = tokens + self.identity.weight[None, None]
        output = self.readout(tokens, active)
        context = torch.stack([v.context_intervals_ns for v in values], 2)
        available = torch.stack([v.available_at_ns for v in values], 2)
        pooled_context = bounding_context(context, active, 2)
        pooled_available = latest_available(available, active, 2)
        if self.cross_attention is not None:
            b, n, g, d = tokens.shape
            source_times = first.time_intervals_ns[:, :, None].expand(b, n, g, 2)
            source = SequenceState(
                tokens.reshape(b * n, g, d),
                active.reshape(b * n, g),
                active.reshape(b * n, g),
                source_times[..., 0].reshape(b * n, g),
                source_times.reshape(b * n, g, 2),
                context.reshape(b * n, g, 2),
                available.reshape(b * n, g),
            )
            query_active = active.any(2).reshape(b * n, 1)
            query_times = first.time_intervals_ns.reshape(b * n, 1, 2)
            query = SequenceState(
                output.reshape(b * n, 1, d),
                query_active,
                query_active,
                query_times[..., 0],
                query_times,
                pooled_context.reshape(b * n, 1, 2),
                pooled_available.reshape(b * n, 1),
            )
            result = self.cross_attention(query, source)
            output = result.tokens.reshape(b, n, d)
            pooled_context = result.context_intervals_ns.reshape(b, n, 2)
            pooled_available = result.available_at_ns.reshape(b, n)
        counts = torch.stack([v.support_count for v in values], 2)
        coverage = torch.stack([v.coverage for v in values], 2)
        return TokenSequence(
            output,
            torch.stack([v.data_valid for v in values], 2).any(2),
            active.any(2),
            (coverage * counts).sum(2) / counts.sum(2),
            first.time_intervals_ns,
            pooled_context,
            pooled_available,
            counts.sum(2),
        )
