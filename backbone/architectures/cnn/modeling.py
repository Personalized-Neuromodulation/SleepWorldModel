"""Mask-aware residual convolution; each kernel edge respects connection_mask."""

import torch
from torch import nn

from ...contracts import SequenceState
from ..sequence import clean, update_state
from .configuration import CNNConfig


class CNNSequenceBlock(nn.Module):
    def __init__(self, dim: int, config: CNNConfig):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, config.kernel_size)
        self.activation = nn.GELU()

    def forward(self, state: SequenceState) -> SequenceState:
        x = clean(self.norm(clean(state.tokens, state.active)), state.active)
        length = x.shape[1]
        kernel = self.conv.kernel_size[0]
        positions = torch.arange(length, device=x.device)
        offsets = torch.arange(kernel, device=x.device) - kernel // 2
        indices = positions[:, None] + offsets
        inside = (indices >= 0) & (indices < length)
        indices = indices.clamp(0, length - 1)
        active = state.active[:, indices] & inside[None] & state.active[:, :, None]
        if state.connection_mask is not None:
            active = active & state.connection_mask[:, positions[:, None], indices]
        neighbors = torch.where(active[..., None], x[:, indices], 0.0)
        output = torch.einsum("mtkh,ohk->mto", neighbors, self.conv.weight)
        output = clean(state.tokens, state.active) + self.activation(
            output + self.conv.bias
        )
        edges = torch.zeros(
            x.shape[0], length, length, dtype=torch.bool, device=x.device
        )
        # Invalid boundary indices may repeat; use OR rather than overwriting edges.
        for k in range(kernel):
            edges[:, positions, indices[:, k]] |= active[:, :, k]
        return update_state(state, state, output, edges)
