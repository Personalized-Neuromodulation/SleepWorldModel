from __future__ import annotations

import bisect
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .channels import (
    DEFAULT_SAMPLE_RATES,
    ChannelInput,
    resolve_channel_groups,
)
from .errors import HSPDataError
from .manifest import load_hsp_manifest
from .resampling import decimate


@dataclass
class _ResolvedRecord:
    record: dict[str, Any]
    path: Path
    selected: dict[str, tuple[str | None, ...]]
    source_sample_rates: dict[str, tuple[float | None, ...]]
    sample_rates: dict[str, float]
    duration_sec: float
    num_epochs: int
    num_samples: int


class HSPDataset(Dataset[dict[str, Any]]):
    """Lazy, time-aligned window loader for standardized HSP HDF5 sessions."""

    def __init__(
        self,
        root: str | Path,
        manifest_path: str | Path,
        *,
        split: str | None = "train",
        channel_profile: str = "hsp_full",
        channel_groups: Mapping[str, Sequence[ChannelInput]] | None = None,
        sampling_mode: str = "context",
        epoch_seconds: float = 30.0,
        context_epochs: int = 20,
        stride_epochs: int = 1,
        missing_channel: str = "mask",
        convert_to_physical: bool = True,
        normalization: str | Mapping[str, str] = "none",
        normalization_clip: float | None = 20.0,
        target_sample_rates: Mapping[str, float] | None = None,
        missing_modality_sample_rates: Mapping[str, float] | None = None,
        return_annotations: bool = False,
        annotation_expert: str = "expert_1",
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        max_open_files: int = 4,
    ) -> None:
        super().__init__()
        self.root = Path(root).resolve()
        self.manifest_path = Path(manifest_path)
        self.channel_groups = resolve_channel_groups(channel_profile, channel_groups)
        self.sampling_mode = sampling_mode
        self.epoch_seconds = float(epoch_seconds)
        self.context_epochs = int(context_epochs)
        self.stride_epochs = int(stride_epochs)
        self.missing_channel = missing_channel
        self.convert_to_physical = convert_to_physical
        self.normalization = normalization
        self.normalization_clip = normalization_clip
        self.target_sample_rates = {
            str(modality): float(rate)
            for modality, rate in (target_sample_rates or {}).items()
        }
        if any(rate <= 0 for rate in self.target_sample_rates.values()):
            raise ValueError("target sample rates must be positive")
        self.missing_modality_sample_rates = {
            **DEFAULT_SAMPLE_RATES,
            **(missing_modality_sample_rates or {}),
        }
        self.return_annotations = return_annotations
        self.annotation_expert = annotation_expert
        self.transform = transform
        self.max_open_files = int(max_open_files)
        self._file_cache: OrderedDict[str, Any] = OrderedDict()

        if sampling_mode not in {"epoch", "context", "night"}:
            raise ValueError("sampling_mode must be 'epoch', 'context', or 'night'")
        if self.epoch_seconds <= 0:
            raise ValueError("epoch_seconds must be positive")
        if self.context_epochs <= 0:
            raise ValueError("context_epochs must be positive")
        if self.stride_epochs <= 0:
            raise ValueError("stride_epochs must be positive")
        if missing_channel not in {"mask", "drop", "error"}:
            raise ValueError("missing_channel must be 'mask', 'drop', or 'error'")
        if self.max_open_files <= 0:
            raise ValueError("max_open_files must be positive")

        records = load_hsp_manifest(self.manifest_path)
        if split is not None:
            if split not in {"train", "validation", "test"}:
                raise ValueError("split must be train, validation, test, or None")
            records = [record for record in records if record.get("split") == split]
        records = [record for record in records if record.get("status") == "ok"]

        self.records: list[_ResolvedRecord] = []
        missing_examples: list[str] = []
        for record in records:
            resolved, missing = self._resolve_record(record)
            if missing and missing_channel == "error":
                missing_examples.append(
                    f"{record.get('subject_id')}/{record.get('session_id')}: "
                    f"{', '.join(missing)}"
                )
                continue
            if missing and missing_channel == "drop":
                continue
            if resolved is not None and resolved.num_samples > 0:
                self.records.append(resolved)

        if missing_examples:
            preview = "; ".join(missing_examples[:5])
            raise HSPDataError(f"requested channels are missing: {preview}")
        if not self.records:
            raise HSPDataError(
                "no manifest records satisfy the requested dataset configuration"
            )

        self._prefix: list[int] = []
        running = 0
        for record in self.records:
            running += record.num_samples
            self._prefix.append(running)

    def _record_path(self, record: Mapping[str, Any]) -> Path:
        relative = record.get("relative_path")
        if relative:
            return (self.root / str(relative)).resolve()
        return Path(str(record["path"])).resolve()

    def _resolve_record(
        self, record: dict[str, Any]
    ) -> tuple[_ResolvedRecord | None, list[str]]:
        signals: Mapping[str, Mapping[str, Any]] = record["signals"]
        selected: dict[str, tuple[str | None, ...]] = {}
        source_sample_rates: dict[str, tuple[float | None, ...]] = {}
        sample_rates: dict[str, float] = {}
        missing: list[str] = []
        available_durations: list[float] = [float(record["duration_sec"])]

        for modality, specs in self.channel_groups.items():
            names: list[str | None] = []
            channel_rates: list[float | None] = []
            for spec in specs:
                source = next(
                    (
                        candidate
                        for candidate in spec.candidates
                        if candidate in signals
                    ),
                    None,
                )
                names.append(source)
                if source is None:
                    channel_rates.append(None)
                    missing.append(f"{modality}/{spec.name}")
                    continue
                info = signals[source]
                fs = float(info.get("fs") or 0.0)
                length = int(info.get("length") or 0)
                if fs <= 0 or length <= 0:
                    channel_rates.append(None)
                    missing.append(f"{modality}/{spec.name}(invalid)")
                    names[-1] = None
                    continue
                channel_rates.append(fs)
                available_durations.append(length / fs)

            rates = tuple(rate for rate in channel_rates if rate is not None)
            requested_rate = self.target_sample_rates.get(modality)
            if not rates:
                output_rate = float(
                    requested_rate
                    or self.missing_modality_sample_rates.get(modality, 0.0)
                )
                if output_rate <= 0:
                    return None, missing
            else:
                output_rate = float(requested_rate or rates[0])
                if requested_rate is None and any(
                    not math.isclose(rate, output_rate, abs_tol=1e-6)
                    for rate in rates[1:]
                ):
                    raise HSPDataError(
                        f"modality {modality!r} has mixed channel sample rates in "
                        f"{record.get('path')}: {rates}. Set target_sample_rates "
                        "for this modality to produce a fixed tensor shape."
                    )

            for spec, source_rate in zip(specs, channel_rates):
                if source_rate is None:
                    continue
                ratio = source_rate / output_rate
                if ratio < 1.0 - 1e-6 or not math.isclose(
                    ratio, round(ratio), rel_tol=0.0, abs_tol=1e-6
                ):
                    raise HSPDataError(
                        f"unsupported resampling for {modality}/{spec.name} in "
                        f"{record.get('path')}: {source_rate:g} -> "
                        f"{output_rate:g} Hz. The built-in anti-aliased "
                        "resampler supports integer-factor downsampling only."
                    )

            selected[modality] = tuple(names)
            source_sample_rates[modality] = tuple(channel_rates)
            sample_rates[modality] = output_rate

        duration_sec = min(available_durations)
        num_epochs = int(math.floor((duration_sec + 1e-6) / self.epoch_seconds))
        required_epochs = 1 if self.sampling_mode == "epoch" else self.context_epochs
        if self.sampling_mode == "night":
            num_samples = 1 if num_epochs > 0 else 0
        elif num_epochs >= required_epochs:
            num_samples = 1 + (num_epochs - required_epochs) // self.stride_epochs
        else:
            num_samples = 0

        return (
            _ResolvedRecord(
                record=record,
                path=self._record_path(record),
                selected=selected,
                source_sample_rates=source_sample_rates,
                sample_rates=sample_rates,
                duration_sec=duration_sec,
                num_epochs=num_epochs,
                num_samples=num_samples,
            ),
            missing,
        )

    def __len__(self) -> int:
        return self._prefix[-1]

    def _locate(self, index: int) -> tuple[_ResolvedRecord, int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record_index = bisect.bisect_right(self._prefix, index)
        previous = self._prefix[record_index - 1] if record_index else 0
        local_index = index - previous
        record = self.records[record_index]
        if self.sampling_mode == "night":
            return record, 0, record.num_epochs
        epochs = 1 if self.sampling_mode == "epoch" else self.context_epochs
        return record, local_index * self.stride_epochs, epochs

    def _get_handle(self, path: Path) -> Any:
        import h5py

        key = str(path)
        handle = self._file_cache.pop(key, None)
        if handle is not None:
            self._file_cache[key] = handle
            return handle
        handle = h5py.File(path, "r")
        self._file_cache[key] = handle
        while len(self._file_cache) > self.max_open_files:
            _, oldest = self._file_cache.popitem(last=False)
            oldest.close()
        return handle

    def close(self) -> None:
        for handle in self._file_cache.values():
            try:
                handle.close()
            except Exception:
                pass
        self._file_cache.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file_cache"] = OrderedDict()
        return state

    def __del__(self) -> None:
        self.close()

    def _normalization_for(self, modality: str) -> str:
        if isinstance(self.normalization, str):
            method = self.normalization
        else:
            method = self.normalization.get(modality, "none")
        if method not in {"none", "zscore", "robust"}:
            raise ValueError(f"unsupported normalization method {method!r}")
        return method

    def _convert(self, values: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
        result = values.astype(np.float32, copy=False)
        if self.convert_to_physical:
            calibration = (
                info.get("dig_min"),
                info.get("dig_max"),
                info.get("phys_min"),
                info.get("phys_max"),
            )
            if all(value is not None for value in calibration):
                dig_min, dig_max, phys_min, phys_max = map(float, calibration)
                denominator = dig_max - dig_min
                if denominator > 0 and all(
                    math.isfinite(value)
                    for value in (dig_min, dig_max, phys_min, phys_max)
                ):
                    result = (result - dig_min) * (
                        (phys_max - phys_min) / denominator
                    ) + phys_min
        return result

    def _normalise(self, values: torch.Tensor, method: str) -> torch.Tensor:
        if method == "none":
            return values
        finite = torch.isfinite(values)
        valid = values[finite]
        if valid.numel() == 0:
            return torch.zeros_like(values)
        if method == "zscore":
            center = valid.mean()
            scale = valid.std(unbiased=False)
        else:
            center = valid.median()
            q25, q75 = torch.quantile(valid, torch.tensor([0.25, 0.75]))
            scale = (q75 - q25) / 1.349
        if not torch.isfinite(scale) or scale <= 1e-8:
            result = values - center
        else:
            result = (values - center) / scale
        result = torch.where(finite, result, torch.zeros_like(result))
        if self.normalization_clip is not None:
            result = result.clamp(-self.normalization_clip, self.normalization_clip)
        return result

    def _read_stage_annotations(
        self, handle: Any, start_epoch: int, epochs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = f"annotations/{self.annotation_expert}/stage"
        if base not in handle:
            return torch.full((epochs,), -1, dtype=torch.int64), torch.zeros(
                epochs, dtype=torch.bool
            )
        group = handle[base]
        if "starts" not in group or "codes" not in group:
            return torch.full((epochs,), -1, dtype=torch.int64), torch.zeros(
                epochs, dtype=torch.bool
            )
        starts = np.asarray(group["starts"][:], dtype=np.float64)
        codes = np.asarray(group["codes"][:], dtype=np.int64)
        durations = (
            np.asarray(group["durations"][:], dtype=np.float64)
            if "durations" in group
            else np.full_like(starts, self.epoch_seconds)
        )
        centers = (
            start_epoch * self.epoch_seconds
            + (np.arange(epochs, dtype=np.float64) + 0.5) * self.epoch_seconds
        )
        indices = np.searchsorted(starts, centers, side="right") - 1
        valid = indices >= 0
        clipped = np.clip(indices, 0, max(len(starts) - 1, 0))
        if len(starts):
            valid &= centers < starts[clipped] + durations[clipped] + 1e-6
            values = np.where(valid, codes[clipped], -1)
        else:
            values = np.full(epochs, -1, dtype=np.int64)
            valid = np.zeros(epochs, dtype=bool)
        return torch.from_numpy(values), torch.from_numpy(valid)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record, start_epoch, epochs = self._locate(index)
        handle = self._get_handle(record.path)
        start_sec = start_epoch * self.epoch_seconds
        duration_sec = epochs * self.epoch_seconds

        signals: dict[str, torch.Tensor] = {}
        available_masks: dict[str, torch.Tensor] = {}
        source_channels: dict[str, tuple[str | None, ...]] = {}

        for modality, specs in self.channel_groups.items():
            output_fs = record.sample_rates[modality]
            output_samples_per_epoch = int(round(self.epoch_seconds * output_fs))
            expected_output_samples = epochs * output_samples_per_epoch
            channels: list[torch.Tensor] = []
            available: list[bool] = []

            for spec, source, source_fs in zip(
                specs,
                record.selected[modality],
                record.source_sample_rates[modality],
            ):
                if source is None:
                    channel = torch.zeros(
                        (epochs, output_samples_per_epoch), dtype=torch.float32
                    )
                    available.append(False)
                else:
                    assert source_fs is not None
                    source_samples_per_epoch = int(
                        round(self.epoch_seconds * source_fs)
                    )
                    expected_source_samples = epochs * source_samples_per_epoch
                    start_sample = int(round(start_sec * source_fs))
                    stop_sample = start_sample + expected_source_samples
                    decimation_factor = int(round(source_fs / output_fs))
                    dataset = handle[f"signals/{source}"]
                    values = dataset[start_sample:stop_sample]
                    if len(values) != expected_source_samples:
                        raise HSPDataError(
                            f"short slice for {record.path}, signals/{source}: "
                            f"expected {expected_source_samples}, got {len(values)}"
                        )
                    info = record.record["signals"][source]
                    continuous = torch.from_numpy(self._convert(values, info))
                    continuous = decimate(continuous, decimation_factor)
                    if continuous.numel() != expected_output_samples:
                        raise HSPDataError(
                            f"resampling length mismatch for {record.path}, "
                            f"signals/{source}: expected {expected_output_samples}, "
                            f"got {continuous.numel()}"
                        )
                    channel = continuous.reshape(epochs, output_samples_per_epoch)
                    channel = self._normalise(
                        channel, self._normalization_for(modality)
                    )
                    available.append(True)
                channels.append(channel)

            signals[modality] = torch.stack(channels, dim=1)
            available_masks[modality] = torch.tensor(available, dtype=torch.bool)
            source_channels[modality] = record.selected[modality]

        sample: dict[str, Any] = {
            "signals": signals,
            "available_mask": available_masks,
            "channel_mask": {
                modality: mask.clone() for modality, mask in available_masks.items()
            },
            "sample_rates": dict(record.sample_rates),
            "channel_names": {
                modality: tuple(spec.name for spec in specs)
                for modality, specs in self.channel_groups.items()
            },
            "source_channels": source_channels,
            "epoch_mask": torch.ones(epochs, dtype=torch.bool),
            "subject_id": str(record.record["subject_id"]),
            "session_id": str(record.record["session_id"]),
            "start_sec": float(start_sec),
            "duration_sec": float(duration_sec),
            "recording_duration_sec": float(record.duration_sec),
            "path": str(record.path),
        }
        if self.return_annotations:
            stage, stage_mask = self._read_stage_annotations(
                handle, start_epoch, epochs
            )
            sample["annotations"] = {"stage": stage, "stage_mask": stage_mask}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


__all__ = ["HSPDataset"]
