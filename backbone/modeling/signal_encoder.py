"""One signal group's ordered patch, mask and sequence computation."""

import torch
from torch import nn

from ..contracts import SignalOutput
from .masking import MaskApplier


class SignalEncoder(nn.Module):
    def __init__(self, patchifier, patch_encoder, sequence_encoder):
        super().__init__()
        self.patchifier = patchifier
        self.masker = MaskApplier()
        self.patch_encoder = patch_encoder
        self.sequence_encoder = sequence_encoder

    def forward(self, group, batch, stage="none", visible=None, need_local=True):
        sample_visible = visible if stage == "waveform" else None
        if stage == "token" and visible is not None:
            b, e, c, s = group.values.shape
            p = s // self.patchifier.patch_samples
            if visible.shape != (b, c, e * p) or visible.dtype != torch.bool:
                raise ValueError("token visibility must be bool [B,C,N]")
            # Skip hidden raw values even during patch projection to avoid NaN
            # gradients. The public token mask is still applied after PatchEncoder.
            sample_visible = visible.reshape(b, c, e, p).permute(0, 2, 1, 3)
            sample_visible = sample_visible.repeat_interleave(
                self.patchifier.patch_samples, -1
            )
        patches = self.patchifier(group, batch, sample_visible)
        grid = self.patch_encoder(patches)
        grid = self.masker(grid, visible if stage == "token" else None)
        local = self.sequence_encoder(grid) if need_local else None
        return SignalOutput(grid, local)
