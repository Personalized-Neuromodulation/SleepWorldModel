from __future__ import annotations

from typing import Any, Sequence

import torch

from .errors import HSPDataError


def hsp_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable nights and native sample rates without implicit resampling."""
    if not batch:
        raise ValueError("cannot collate an empty batch")

    modalities = tuple(batch[0]["signals"].keys())
    max_epochs = max(int(sample["epoch_mask"].numel()) for sample in batch)
    output_signals: dict[str, torch.Tensor] = {}
    output_sample_masks: dict[str, torch.Tensor] = {}
    output_available: dict[str, torch.Tensor] = {}
    output_channels: dict[str, torch.Tensor] = {}
    output_rates: dict[str, torch.Tensor] = {}

    for modality in modalities:
        channel_count = batch[0]["signals"][modality].shape[1]
        max_samples = max(sample["signals"][modality].shape[2] for sample in batch)
        values = torch.zeros(
            (len(batch), max_epochs, channel_count, max_samples), dtype=torch.float32
        )
        sample_mask = torch.zeros(
            (len(batch), max_epochs, max_samples), dtype=torch.bool
        )
        for batch_index, sample in enumerate(batch):
            tensor = sample["signals"][modality]
            epochs, channels, samples = tensor.shape
            if channels != channel_count:
                raise HSPDataError(
                    f"channel count changed within batch for modality {modality!r}"
                )
            values[batch_index, :epochs, :, :samples] = tensor
            sample_mask[batch_index, :epochs, :samples] = sample[
                "epoch_mask"
            ].unsqueeze(-1)
        output_signals[modality] = values
        output_sample_masks[modality] = sample_mask
        output_available[modality] = torch.stack(
            [sample["available_mask"][modality] for sample in batch]
        )
        output_channels[modality] = torch.stack(
            [sample["channel_mask"][modality] for sample in batch]
        )
        output_rates[modality] = torch.tensor(
            [sample["sample_rates"][modality] for sample in batch],
            dtype=torch.float32,
        )

    epoch_mask = torch.zeros((len(batch), max_epochs), dtype=torch.bool)
    for index, sample in enumerate(batch):
        epochs = sample["epoch_mask"].numel()
        epoch_mask[index, :epochs] = sample["epoch_mask"]

    result: dict[str, Any] = {
        "signals": output_signals,
        "sample_mask": output_sample_masks,
        "available_mask": output_available,
        "channel_mask": output_channels,
        "sample_rates": output_rates,
        "epoch_mask": epoch_mask,
        "channel_names": batch[0]["channel_names"],
        "source_channels": [sample["source_channels"] for sample in batch],
        "subject_id": [sample["subject_id"] for sample in batch],
        "session_id": [sample["session_id"] for sample in batch],
        "start_sec": torch.tensor([sample["start_sec"] for sample in batch]),
        "duration_sec": torch.tensor([sample["duration_sec"] for sample in batch]),
        "recording_duration_sec": torch.tensor(
            [sample["recording_duration_sec"] for sample in batch]
        ),
        "path": [sample["path"] for sample in batch],
    }

    if "annotations" in batch[0]:
        stage = torch.full((len(batch), max_epochs), -1, dtype=torch.int64)
        stage_mask = torch.zeros((len(batch), max_epochs), dtype=torch.bool)
        for index, sample in enumerate(batch):
            count = sample["annotations"]["stage"].numel()
            stage[index, :count] = sample["annotations"]["stage"]
            stage_mask[index, :count] = sample["annotations"]["stage_mask"]
        result["annotations"] = {"stage": stage, "stage_mask": stage_mask}

    return result


__all__ = ["hsp_collate_fn"]
