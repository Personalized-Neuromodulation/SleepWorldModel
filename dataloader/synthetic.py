"""Deterministic, small PSG fixtures for notebook exploration and smoke tests."""

import math

import torch

from .collate import collate_windows


def synthetic_windows(
    batch_size=2, epochs=2, sample_rate=200, epoch_seconds=30, seed=7
) -> dict:
    """Return actual collate schema, including missing channels and padded epochs."""
    channels = {
        "eeg": ("f3-m2", "f4-m1", "c3-m2", "c4-m1", "o1-m2", "o2-m1"),
        "eog": ("e1", "e2"),
        "ecg": ("ecg",),
        "emg": ("chin1-chin2", "lat", "rat"),
        "respiratory": ("airflow", "snore", "spo2"),
    }
    generator = torch.Generator().manual_seed(seed)
    s = round(sample_rate * epoch_seconds)
    time = torch.arange(s).float() / sample_rate
    samples = []
    for i in range(batch_size):
        e = max(1, epochs - (i % 2))
        sample = {
            "signals": {},
            "quality": {},
            "available_mask": {},
            "channel_mask": {},
            "sample_rates": {name: sample_rate for name in channels},
            "channel_names": channels,
            "units": {
                name: tuple("a.u." for _ in ids) for name, ids in channels.items()
            },
            "epoch_mask": torch.ones(e, dtype=torch.bool),
            "epoch_in_record": torch.arange(e),
            "epoch_start_offset_ns": torch.arange(e) * round(epoch_seconds * 1e9),
            "start_sec": 0.0,
            "duration_sec": e * epoch_seconds,
            "recording_duration_sec": e * epoch_seconds,
            "night_grade": 5,
            "subject_id": f"synthetic-{i}",
            "session_id": "1",
            "recording_id": f"synthetic-{i}-1",
            "path": "synthetic",
            "dataset": "synthetic",
            "version": "1",
            "tasks": {},
        }
        for name, ids in channels.items():
            c = len(ids)
            frequencies = torch.arange(1, c + 1).float()[:, None]
            wave = torch.sin(2 * math.pi * frequencies * time)
            values = wave[None].repeat(e, 1, 1)
            values += 0.05 * torch.randn(values.shape, generator=generator)
            available = torch.ones(c, dtype=torch.bool)
            if i % 2 and name == "eeg":
                available[-1] = False
                values[:, -1] = float("nan")
            good = torch.ones(e, c, dtype=torch.bool)
            sample["signals"][name] = values
            sample["quality"][name] = {
                "coverage_valid": good.clone(),
                "processing_valid": good.clone(),
                "artifact_valid": good.clone(),
                "valid": good & available[None],
                "hard_code": torch.zeros(e, c, dtype=torch.uint8),
            }
            sample["available_mask"][name] = available
            sample["channel_mask"][name] = available.clone()
        samples.append(sample)
    return collate_windows(samples)
