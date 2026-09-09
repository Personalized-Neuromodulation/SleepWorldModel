"""Encode independent patches using Linear or a stack over subpatches."""

import torch
from torch import nn

from ..configuration import PatchEncoderConfig
from ..contracts import PatchBatch, SequenceState, TokenGrid
from .pooling import MaskedReadout, sinusoidal


class PatchEncoder(nn.Module):
    def __init__(
        self, patch_samples: int, dim: int, config: PatchEncoderConfig, blocks
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        if config.variant == "direct_linear":
            self.projection = nn.Linear(patch_samples, dim)
        else:
            if patch_samples % config.subpatch_samples:
                raise ValueError("patch_samples must be divisible by subpatch_samples")
            self.projection = nn.Linear(config.subpatch_samples, config.hidden_dim)
            self.blocks = nn.ModuleList(blocks)
            self.readout = MaskedReadout(config.hidden_dim, config.readout)
            self.output_projection = nn.Linear(config.hidden_dim, dim)

    def forward(self, patches: PatchBatch) -> TokenGrid:
        sample_active = patches.sample_visible & patches.data_valid[..., None]
        values = torch.where(sample_active, patches.values, 0.0)
        b, c, n, length = values.shape
        visible = sample_active.any(-1)
        active = patches.data_valid & visible
        if self.config.variant == "direct_linear":
            tokens = self.projection(values)
        else:
            sub = self.config.subpatch_samples
            p = length // sub
            x = self.projection(values.reshape(b * c * n, p, sub))
            mask = sample_active.reshape(b * c * n, p, sub).any(-1)
            positions = torch.arange(p, device=values.device).expand(b * c * n, p)
            if self.config.position_embedding:
                x = x + sinusoidal(positions, x.shape[-1], x.dtype)
            x = torch.where(mask[..., None], x, 0.0)
            # No cross-patch mixing: M=B*C*N is an independent batch axis.
            starts = patches.layout.time_intervals_ns[..., 0][:, None, :, None]
            offset = torch.arange(p, device=values.device) * sub
            offset_ns = (
                (offset.double() * (1e9 / patches.layout.sample_rate_hz)).round().long()
            )
            starts = (starts + offset_ns).expand(b, c, n, p).reshape(b * c * n, p)
            ends = starts + round(sub * 1e9 / patches.layout.sample_rate_hz)
            times = torch.stack((starts, ends), -1)
            times = torch.where(mask[..., None], times, -1)
            state = SequenceState(
                x,
                mask,
                mask,
                times[..., 0],
                times,
                times,
                torch.full_like(positions, -1),
            )
            for block in self.blocks:
                state = block(state)
            tokens = self.output_projection(self.readout(state.tokens, state.active))
            tokens = tokens.reshape(b, c, n, self.dim)
        times = patches.layout.time_intervals_ns
        return TokenGrid(
            torch.where(active[..., None], tokens, 0.0),
            patches.data_valid,
            visible,
            patches.data_valid.float(),
            times,
            torch.where(active.any(1)[..., None], times, -1),
            torch.full_like(times[..., 0], -1),
            patches.channel_ids,
            patches.layout,
        )
