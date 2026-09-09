"""Pack true local windows before sequence computation; then restore the grid."""

from dataclasses import replace

from torch import nn
from torch.nn import functional as F

from ..architectures.sequence import clean
from ..configuration import SequenceEncoderConfig
from ..contracts import SequenceState, TokenGrid
from .pooling import bounding_context, latest_available, sinusoidal


class PatchSequenceEncoder(nn.Module):
    def __init__(
        self, dim: int, patch_seconds: float, config: SequenceEncoderConfig, blocks
    ):
        super().__init__()
        q = config.window_seconds / patch_seconds
        if abs(q - round(q)) > 1e-6 or q < 1:
            raise ValueError(
                "window_seconds must be an integer multiple of patch_seconds"
            )
        self.window_patches = round(q)
        self.config = config
        self.blocks = nn.ModuleList(blocks)

    def forward(self, grid: TokenGrid) -> TokenGrid:
        if not self.blocks:
            return grid
        b, c, n, d = grid.tokens.shape
        e, p = grid.patch_layout.epoch_count, grid.patch_layout.patches_per_epoch
        q = self.window_patches
        if q > p:
            raise ValueError("first version limits context windows to one epoch")
        windows = (p + q - 1) // q
        padding = windows * q - p

        def pack(value, fill=0):
            # value is [B,C,N,F]; never merge across epoch or recording boundaries.
            width = value.shape[-1]
            value = value.reshape(b, c, e, p, width)
            value = F.pad(value, (0, 0, 0, padding), value=fill)
            return value.reshape(b * c * e * windows, q, width)

        def unpack(value):
            width = value.shape[-1]
            return value.reshape(b, c, e, windows * q, width)[..., :p, :].reshape(
                b, c, n, width
            )

        def expand(value):
            return value[:, None].expand(b, c, n, value.shape[-1])

        positions = grid.time_intervals_ns[..., 0]
        tokens = grid.tokens
        if self.config.position_embedding:
            # Convert ns in float64 first to preserve useful precision on long nights.
            embedding = sinusoidal(positions.double() / 1e9, d, tokens.dtype)
            tokens = tokens + embedding[:, None]
        times = pack(expand(grid.time_intervals_ns), -1)
        valid = pack(grid.data_valid[..., None]).squeeze(-1)
        visible = pack(grid.visible[..., None]).squeeze(-1)
        allowed = None
        if self.config.causal:
            allowed = times[:, None, :, 1] <= times[:, :, None, 1]
        state = SequenceState(
            pack(clean(tokens, grid.active)),
            valid,
            visible,
            times[..., 0],
            times,
            pack(expand(grid.context_intervals_ns), -1),
            pack(expand(grid.available_at_ns[..., None]), -1).squeeze(-1),
            allowed,
        )
        for block in self.blocks:
            state = block(state)
        context = bounding_context(unpack(state.context_intervals_ns), grid.active, 1)
        available = latest_available(
            unpack(state.available_at_ns[..., None]).squeeze(-1), grid.active, 1
        )
        return replace(
            grid,
            tokens=unpack(state.tokens),
            context_intervals_ns=context,
            available_at_ns=available,
        )
