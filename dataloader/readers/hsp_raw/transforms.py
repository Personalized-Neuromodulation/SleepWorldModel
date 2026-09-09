from __future__ import annotations

from typing import Any, Sequence

import torch


class RandomChannelDropout:
    """Drop available channels while preserving the original availability mask."""

    def __init__(
        self,
        probability: float = 0.2,
        modalities: Sequence[str] = ("eeg",),
        min_remaining: int = 1,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        if min_remaining < 0:
            raise ValueError("min_remaining must be non-negative")
        self.probability = probability
        self.modalities = frozenset(modalities)
        self.min_remaining = min_remaining

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for modality in self.modalities:
            if modality not in sample["signals"]:
                continue
            mask = sample["channel_mask"][modality]
            available_indices = torch.where(mask)[0]
            if len(available_indices) <= self.min_remaining:
                continue
            proposed = torch.rand(len(available_indices)) < self.probability
            max_drop = len(available_indices) - self.min_remaining
            drop_indices = available_indices[torch.where(proposed)[0][:max_drop]]
            if len(drop_indices):
                sample["signals"][modality][:, drop_indices] = 0
                sample["channel_mask"][modality][drop_indices] = False
        return sample


__all__ = ["RandomChannelDropout"]
