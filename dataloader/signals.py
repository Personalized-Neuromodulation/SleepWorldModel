"""Model-independent signal contracts and conversion from collated reader output."""

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class SignalGroup:
    values: Tensor  # [B,E,C,S]
    coverage_valid: Tensor  # [B,E,C]
    processing_valid: Tensor
    artifact_valid: Tensor
    valid: Tensor
    hard_code: Tensor
    available_mask: Tensor  # [B,C]
    channel_mask: Tensor
    sample_rate_hz: Tensor  # [B]
    channel_ids: tuple[str, ...]
    units: tuple[str, ...]

    def to(self, device) -> "SignalGroup":
        return replace(
            self,
            **{
                f.name: getattr(self, f.name).to(device)
                for f in fields(self)
                if isinstance(getattr(self, f.name), Tensor)
            },
        )


@dataclass(frozen=True)
class SignalBatch:
    groups: dict[str, SignalGroup]
    epoch_mask: Tensor
    epoch_in_record: Tensor
    epoch_start_offset_ns: Tensor
    start_sec: Tensor
    duration_sec: Tensor
    recording_duration_sec: Tensor
    recording_ids: tuple[str, ...]
    night_grade: Tensor | None = None

    def to(self, device) -> "SignalBatch":
        tensors = {
            f.name: getattr(self, f.name).to(device)
            for f in fields(self)
            if isinstance(getattr(self, f.name), Tensor)
        }
        return replace(
            self, groups={k: v.to(device) for k, v in self.groups.items()}, **tensors
        )

    def validate(self) -> None:
        if not self.groups or self.epoch_mask.ndim != 2:
            raise ValueError("nonempty groups and epoch_mask [B,E] required")
        b, e = self.epoch_mask.shape
        if b < 1 or e < 1:
            raise ValueError("SignalBatch must have nonempty B and E axes")
        for key in ("epoch_mask", "epoch_in_record", "epoch_start_offset_ns"):
            if getattr(self, key).shape != (b, e):
                raise ValueError(f"{key} must have shape [B,E]")
        if self.epoch_mask.dtype != torch.bool:
            raise ValueError("epoch_mask must be bool")
        for key in ("epoch_in_record", "epoch_start_offset_ns"):
            value = getattr(self, key)
            if value.dtype != torch.int64 or (value[self.epoch_mask] < 0).any():
                raise ValueError(f"{key} must contain nonnegative int64 real positions")
        if len(self.recording_ids) != b:
            raise ValueError("recording_ids must have length B")
        for key in ("start_sec", "duration_sec", "recording_duration_sec"):
            if getattr(self, key).shape != (b,):
                raise ValueError(f"{key} must have shape [B]")
        if self.night_grade is not None and self.night_grade.shape != (b,):
            raise ValueError("night_grade must have shape [B]")
        for name, group in self.groups.items():
            if group.values.ndim != 4 or group.values.shape[:2] != (b, e):
                raise ValueError(f"{name}: values must have shape [B,E,C,S]")
            c, s = group.values.shape[2:]
            if c == 0 or s == 0 or not group.values.is_floating_point():
                raise ValueError(f"{name}: empty or non-floating signals")
            if len(group.channel_ids) != c or len(set(group.channel_ids)) != c:
                raise ValueError(f"{name}: channel_ids must be unique and match C")
            if len(group.units) != c:
                raise ValueError(f"{name}: units must match C")
            for key in (
                "coverage_valid",
                "processing_valid",
                "artifact_valid",
                "valid",
            ):
                value = getattr(group, key)
                if value.shape != (b, e, c) or value.dtype != torch.bool:
                    raise ValueError(f"{name}.{key} must be bool [B,E,C]")
            if group.hard_code.shape != (b, e, c):
                raise ValueError(f"{name}: hard_code must be [B,E,C]")
            for key in ("available_mask", "channel_mask"):
                value = getattr(group, key)
                if value.shape != (b, c) or value.dtype != torch.bool:
                    raise ValueError(f"{name}.{key} must be bool [B,C]")
            rate = group.sample_rate_hz
            if (
                rate.shape != (b,)
                or not torch.isfinite(rate).all()
                or not (rate > 0).all()
                or not (rate == rate[0]).all()
            ):
                raise ValueError(f"{name}: one positive sample rate per batch required")
            expected = (
                group.available_mask[:, None]
                & group.coverage_valid
                & group.processing_valid
                & group.artifact_valid
            )
            if not torch.equal(group.valid, expected):
                raise ValueError(f"{name}: valid disagrees with quality masks")
            if (group.channel_mask & ~group.available_mask).any():
                raise ValueError(f"{name}: unavailable channels cannot be selected")
            epoch_ns = round(s * 1e9 / float(rate[0]))
            for row, exists in zip(self.epoch_start_offset_ns, self.epoch_mask):
                starts = row[exists]
                if ((starts[1:] - starts[:-1]) < epoch_ns).any():
                    raise ValueError("real epochs must be ordered and nonoverlapping")


def as_signal_batch(
    batch: Mapping,
    *,
    split_spo2: bool = True,
    scales: Mapping[str, float] | None = None,
) -> SignalBatch:
    """Adapt reader tensors; optional fixed scales divide values (no fitted statistics).

    Quality and identity metadata are sliced together. No source tensor is modified.
    Unrecognized groups are preserved, making this usable by additional datasets.
    """
    groups = {}
    for name, values in batch["signals"].items():
        ids = tuple(batch["channel_names"][name])
        selections = {name: tuple(range(len(ids)))}
        if split_spo2 and name == "respiratory" and "spo2" in ids:
            if "spo2" in batch["signals"]:
                raise ValueError("spo2 exists both independently and in respiratory")
            selections = {
                "respiratory": tuple(i for i, ch in enumerate(ids) if ch != "spo2"),
                "spo2": (ids.index("spo2"),),
            }
        for target, indices in selections.items():
            if not indices:
                continue
            # Preserve a view for the usual contiguous groups; only reordering copies.
            selection = (
                slice(indices[0], indices[-1] + 1)
                if indices == tuple(range(indices[0], indices[-1] + 1))
                else indices
            )
            selected = values[:, :, selection, :]
            scale = (scales or {}).get(target, 1.0)
            if not (0 < scale < float("inf")):
                raise ValueError(f"invalid fixed scale for {target}")
            groups[target] = SignalGroup(
                values=selected if scale == 1 else selected / scale,
                **{
                    key: batch["quality"][name][key][:, :, indices]
                    for key in (
                        "coverage_valid",
                        "processing_valid",
                        "artifact_valid",
                        "valid",
                        "hard_code",
                    )
                },
                available_mask=batch["available_mask"][name][:, indices],
                channel_mask=batch["channel_mask"][name][:, indices],
                sample_rate_hz=batch["sample_rates"][name],
                channel_ids=tuple(ids[i] for i in indices),
                units=tuple(
                    batch["units"][name][i]
                    if scale == 1
                    else f"{batch['units'][name][i]}/{scale:g}"
                    for i in indices
                ),
            )
    if scales and set(scales) - groups.keys():
        raise ValueError("scales contains unknown encoding groups")
    result = SignalBatch(
        groups=groups,
        recording_ids=tuple(batch["recording_id"]),
        night_grade=batch.get("night_grade"),
        **{
            key: batch[key]
            for key in (
                "epoch_mask",
                "epoch_in_record",
                "epoch_start_offset_ns",
                "start_sec",
                "duration_sec",
                "recording_duration_sec",
            )
        },
    )
    result.validate()
    return result
