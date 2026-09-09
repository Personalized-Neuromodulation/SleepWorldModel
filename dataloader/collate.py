"""Pad windows while keeping QC, label validity and epoch identities explicit."""

import torch


def collate_windows(samples):
    if not samples:
        raise ValueError("cannot collate an empty batch")
    first = samples[0]
    for sample in samples[1:]:
        for key in ("dataset", "version", "channel_names", "sample_rates", "units"):
            if sample[key] != first[key]:
                raise ValueError(f"incompatible batch metadata: {key}")
        if set(sample["tasks"]) != set(first["tasks"]):
            raise ValueError("incompatible task sets")
    maximum = max(len(s["epoch_mask"]) for s in samples)

    def padded(values, fill=0):
        result = values[0].new_full((len(values), maximum, *values[0].shape[1:]), fill)
        for i, value in enumerate(values):
            result[i, : len(value)] = value
        return result

    batch = {
        "signals": {},
        "quality": {},
        "available_mask": {},
        "channel_mask": {},
        "sample_rates": {},
    }
    for name in first["signals"]:
        batch["signals"][name] = padded([s["signals"][name] for s in samples])
        batch["quality"][name] = {
            k: padded([s["quality"][name][k] for s in samples])
            for k in first["quality"][name]
        }
        for key in ("available_mask", "channel_mask"):
            batch[key][name] = torch.stack([s[key][name] for s in samples])
        batch["sample_rates"][name] = torch.tensor(
            [s["sample_rates"][name] for s in samples]
        )
    batch["tasks"] = {}
    for task in first["tasks"]:
        fields = first["tasks"][task]["field_names"]
        if any(s["tasks"][task]["field_names"] != fields for s in samples):
            raise ValueError("task field order mismatch")
        batch["tasks"][task] = {
            k: padded([s["tasks"][task][k] for s in samples])
            for k in first["tasks"][task]
            if k != "field_names"
        }
        batch["tasks"][task]["field_names"] = fields
    for key in ("epoch_mask", "epoch_in_record", "epoch_start_offset_ns"):
        batch[key] = padded(
            [s[key] for s in samples], False if key == "epoch_mask" else -1
        )
    for key in ("start_sec", "duration_sec", "recording_duration_sec"):
        batch[key] = torch.tensor([s[key] for s in samples], dtype=torch.float64)
    has_grade = ["night_grade" in sample for sample in samples]
    if any(has_grade) and not all(has_grade):
        raise ValueError("night_grade must be present for all samples or none")
    if all(has_grade):
        batch["night_grade"] = torch.tensor(
            [s["night_grade"] for s in samples], dtype=torch.uint8
        )
    for key in ("subject_id", "session_id", "recording_id", "path"):
        batch[key] = [s[key] for s in samples]
    for key in ("channel_names", "units", "dataset", "version"):
        batch[key] = first[key]
    return batch
