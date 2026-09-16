"""Model-independent signal contracts and conversion from collated reader output."""

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from math import isfinite
from numbers import Real

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
    data_valid: Tensor  # [B,C,E], source QC & epoch existence, prepared by adapter

    @property
    def visible(self) -> Tensor:
        """View visibility [B,C,E]; broadcasting channel selection does not run QC."""
        return self.channel_mask[..., None].expand_as(self.data_valid)

    def to(self, device, non_blocking=False) -> "SignalGroup":
        return replace(
            self,
            **{
                f.name: getattr(self, f.name).to(device, non_blocking=non_blocking)
                for f in fields(self)
                if isinstance(getattr(self, f.name), Tensor)
            },
        )

    def pin_memory(self) -> "SignalGroup":
        return replace(
            self,
            **{
                f.name: getattr(self, f.name).pin_memory()
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
    view_start_samples: Tensor | None = None  # [B], crop inside source epoch

    @property
    def data_valid(self):
        """Epoch QC [B,C,E], excluding padding; visibility/dropout stay separate."""
        return {name: group.data_valid for name, group in self.groups.items()}

    def to(self, device, non_blocking=False) -> "SignalBatch":
        tensors = {
            f.name: getattr(self, f.name).to(device, non_blocking=non_blocking)
            for f in fields(self)
            if isinstance(getattr(self, f.name), Tensor)
        }
        return replace(
            self,
            groups={k: v.to(device, non_blocking) for k, v in self.groups.items()},
            **tensors,
        )

    def pin_memory(self) -> "SignalBatch":
        """DataLoader pins the already-adapted tensors, including derived QC."""
        return replace(
            self,
            groups={k: v.pin_memory() for k, v in self.groups.items()},
            **{
                f.name: getattr(self, f.name).pin_memory()
                for f in fields(self)
                if isinstance(getattr(self, f.name), Tensor)
            },
        )

    def validate(self, *, foundation=False, input_spec=None) -> None:
        """Validate once at the data boundary, before GPU transfer when possible."""
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
            if (
                group.data_valid.shape != (b, c, e)
                or group.data_valid.dtype != torch.bool
            ):
                raise ValueError(f"{name}: data_valid must be bool [B,C,E]")
            if not torch.equal(
                group.data_valid,
                (group.valid & self.epoch_mask[..., None]).transpose(1, 2),
            ):
                raise ValueError(
                    f"{name}: data_valid disagrees with source epoch/QC masks; rebuild at the data boundary"
                )
            active = (group.data_valid & group.visible).transpose(1, 2)
            if not torch.isfinite(
                torch.where(active[..., None], group.values, 0.0)
            ).all():
                raise ValueError(f"{name}: nonfinite values in valid signal")
            epoch_ns = round(s * 1e9 / float(rate[0]))
            starts = self.epoch_start_offset_ns
            previous = torch.where(self.epoch_mask, starts, -1).cummax(1).values
            compare = self.epoch_mask[:, 1:] & (previous[:, :-1] >= 0)
            if (compare & ((starts[:, 1:] - previous[:, :-1]) < epoch_ns)).any():
                raise ValueError("real epochs must be ordered and nonoverlapping")
        if foundation or input_spec is not None:
            from .validation import validate_input_spec

            validate_input_spec(self, input_spec, foundation=foundation)


def collate_signal_windows(samples, *, scales=None):
    """CPU worker boundary: collate and validate before DataLoader pinning.

    Task labels stay alongside the shared SignalBatch for an optional probe.
    """
    from .collate import collate_windows

    raw = collate_windows(samples)
    return {
        "signal_batch": as_signal_batch(raw, scales=scales, foundation=True),
        "tasks": raw["tasks"],
    }


def fixed_scale(spec):
    """Resolve fixed (offset, divisor); numeric specs retain division-only behavior."""
    if isinstance(spec, Mapping):
        if set(spec) != {"offset", "divisor"}:
            raise ValueError("fixed scale requires exactly offset and divisor")
        offset, divisor = spec["offset"], spec["divisor"]
    else:
        offset, divisor = 0.0, spec
    if any(isinstance(x, bool) or not isinstance(x, Real) for x in (offset, divisor)):
        raise ValueError("fixed scale offset/divisor must be numbers")
    if not isfinite(offset) or not isfinite(divisor) or divisor <= 0:
        raise ValueError("fixed scale requires finite offset and positive divisor")
    return float(offset), float(divisor)


def as_signal_batch(
    batch: Mapping,
    *,
    split_spo2: bool = True,
    scales: Mapping[str, float | Mapping[str, float]] | None = None,
    foundation: bool = False,
    input_spec: Mapping | None = None,
) -> SignalBatch:
    """Adapt reader tensors; optional fixed affine scales use no batch statistics.

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
            offset, scale = fixed_scale((scales or {}).get(target, 1.0))
            if offset:
                selected = (selected - offset).div_(scale)
            elif scale != 1:
                selected = selected / scale

            def scaled_unit(unit):
                if offset:
                    return f"({unit}-{offset:g})/{scale:g}"
                return unit if scale == 1 else f"{unit}/{scale:g}"

            groups[target] = SignalGroup(
                values=selected,
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
                data_valid=(
                    batch["quality"][name]["valid"][:, :, indices]
                    & batch["epoch_mask"][..., None]
                ).transpose(1, 2),
                units=tuple(scaled_unit(batch["units"][name][i]) for i in indices),
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
    result.validate(foundation=foundation, input_spec=input_spec)
    return result
