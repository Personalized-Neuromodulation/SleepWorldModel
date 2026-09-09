"""Apply externally chosen visibility; no target or random policy lives here."""

from dataclasses import replace

import torch
from torch import nn

from ..contracts import TokenGrid


class MaskApplier(nn.Module):
    def forward(self, grid: TokenGrid, visible=None) -> TokenGrid:
        if visible is None:
            return grid
        if visible.shape != grid.visible.shape or visible.dtype != torch.bool:
            raise ValueError("token visibility must be bool [B,C,N]")
        visible = grid.visible & visible
        active = grid.data_valid & visible
        any_active = active.any(1)
        return replace(
            grid,
            tokens=torch.where(active[..., None], grid.tokens, 0.0),
            visible=visible,
            context_intervals_ns=torch.where(
                any_active[..., None], grid.context_intervals_ns, -1
            ),
            available_at_ns=torch.where(any_active, grid.available_at_ns, -1),
        )
