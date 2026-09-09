"""Nonoverlapping patches within epochs; no patch ever crosses a recording gap."""

import torch
from torch import nn

from dataloader.signals import SignalBatch, SignalGroup

from ..contracts import PatchBatch, PatchLayout


class Patchifier(nn.Module):
    def __init__(self, patch_samples: int, sample_rate_hz: float):
        super().__init__()
        self.patch_samples = patch_samples
        self.sample_rate_hz = sample_rate_hz

    def layout(self, group: SignalGroup, batch: SignalBatch) -> PatchLayout:
        b, e, _, s = group.values.shape
        if s % self.patch_samples:
            raise ValueError("epoch samples must be divisible by patch_samples")
        if not (group.sample_rate_hz == self.sample_rate_hz).all():
            raise ValueError("sample rate differs from the constructed backbone")
        p = s // self.patch_samples
        offsets = torch.arange(p, device=group.values.device) * self.patch_samples
        ns = torch.round(offsets.double() * (1e9 / self.sample_rate_hz)).long()
        start = batch.epoch_start_offset_ns[..., None] + ns
        duration_ns = round(self.patch_samples * 1e9 / self.sample_rate_hz)
        times = torch.stack((start, start + duration_ns), -1)
        times = torch.where(batch.epoch_mask[:, :, None, None], times, -1)
        return PatchLayout(
            self.sample_rate_hz,
            self.patch_samples,
            p,
            e,
            batch.epoch_in_record.repeat_interleave(p, 1),
            offsets.repeat(e),
            times.reshape(b, e * p, 2),
        )

    def forward(
        self, group: SignalGroup, batch: SignalBatch, sample_visible=None
    ) -> PatchBatch:
        layout = self.layout(group, batch)
        b, e, c, s = group.values.shape
        p, length = layout.patches_per_epoch, self.patch_samples
        quality = batch.epoch_mask[:, :, None] & group.valid
        active = (quality & group.channel_mask[:, None, :])[..., None].expand_as(
            group.values
        )
        if sample_visible is not None:
            if (
                sample_visible.shape != group.values.shape
                or sample_visible.dtype != torch.bool
            ):
                raise ValueError("waveform visibility must be bool [B,E,C,S]")
            active = active & sample_visible
        values = torch.where(active, group.values, 0.0)
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite values in visible, quality-valid signal")

        def patches(value):
            return (
                value.reshape(b, e, c, p, length)
                .permute(0, 2, 1, 3, 4)
                .reshape(b, c, e * p, length)
            )

        valid = quality.permute(0, 2, 1).repeat_interleave(p, -1)
        return PatchBatch(
            patches(values), patches(active), valid, layout, group.channel_ids
        )
