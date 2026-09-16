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
        p = s // self.patch_samples
        offsets = torch.arange(p, device=group.values.device) * self.patch_samples
        duration_ns = round(self.patch_samples * 1e9 / self.sample_rate_hz)
        ns = torch.arange(p, device=group.values.device) * duration_ns
        start = batch.epoch_start_offset_ns[..., None] + ns
        if batch.view_start_samples is not None:
            start = start + batch.view_start_samples[:, None, None] * round(
                1e9 / self.sample_rate_hz
            )
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
        self,
        group: SignalGroup,
        batch: SignalBatch,
        sample_visible=None,
        *,
        layout=None,
    ) -> PatchBatch:
        layout = self.layout(group, batch) if layout is None else layout
        b, e, c, s = group.values.shape
        p, length = layout.patches_per_epoch, self.patch_samples
        # The adapter owns QC; the view sampler owns visibility. Only apply them.
        data_valid, epoch_visible = group.data_valid, group.visible
        visible = epoch_visible.transpose(1, 2)[..., None].expand_as(group.values)
        active = (
            (data_valid & epoch_visible)
            .transpose(1, 2)[..., None]
            .expand_as(group.values)
        )
        if sample_visible is not None:
            if (
                sample_visible.shape != group.values.shape
                or sample_visible.dtype != torch.bool
            ):
                raise ValueError("waveform visibility must be bool [B,E,C,S]")
            visible = visible & sample_visible
            active = active & sample_visible
        values = torch.where(active, group.values, 0.0)

        def patches(value):
            return (
                value.reshape(b, e, c, p, length)
                .permute(0, 2, 1, 3, 4)
                .reshape(b, c, e * p, length)
            )

        valid = data_valid
        return PatchBatch(
            patches(values),
            patches(visible),
            valid,
            layout,
            group.channel_ids,
        )
