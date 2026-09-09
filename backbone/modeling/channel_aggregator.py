"""Pool channels at each time without losing quality or timing metadata."""

import torch
from torch import nn

from ..contracts import TokenGrid, TokenSequence
from .pooling import MaskedReadout


class ChannelAggregator(nn.Module):
    def __init__(self, channels: int, dim: int, kind: str, identity: bool):
        super().__init__()
        self.identity = nn.Embedding(channels, dim) if identity else None
        self.readout = MaskedReadout(dim, kind)

    def forward(self, grid: TokenGrid) -> TokenSequence:
        tokens = grid.tokens
        if self.identity is not None:
            tokens = tokens + self.identity.weight[None, :, None]
        result = self.readout(tokens.transpose(1, 2), grid.active.transpose(1, 2))
        return TokenSequence(
            result,
            grid.data_valid.any(1),
            grid.active.any(1),
            grid.coverage.mean(1),
            grid.time_intervals_ns,
            grid.context_intervals_ns,
            grid.available_at_ns,
            torch.full_like(grid.coverage[:, 0], tokens.shape[1]),
        )
