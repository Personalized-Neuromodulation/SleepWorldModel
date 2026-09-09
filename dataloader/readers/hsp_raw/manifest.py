from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .errors import HSPDataError

MANIFEST_VERSION = 1


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def subject_split(
    subject_id: str,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> str:
    """Return a deterministic subject-level train/validation/test split."""
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios):
        raise ValueError("ratios must contain three non-negative values")
    total = sum(ratios)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"split ratios must sum to 1, got {total}")

    digest = hashlib.sha256(f"{seed}:{subject_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < ratios[0]:
        return "train"
    if value < ratios[0] + ratios[1]:
        return "validation"
    return "test"


def inspect_hsp_file(
    h5_path: str | Path,
    root: str | Path | None = None,
    split_seed: int = 42,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, Any]:
    """Read HDF5 metadata without loading complete signal arrays."""
    import h5py

    path = Path(h5_path).resolve()
    root_path = Path(root).resolve() if root is not None else None
    try:
        relative_path = str(path.relative_to(root_path)) if root_path else None
    except ValueError:
        relative_path = None

    record: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "path": str(path),
        "relative_path": relative_path,
        "status": "ok",
    }

    try:
        with h5py.File(path, "r") as handle:
            subject_id = str(
                _json_value(handle.attrs.get("subject_id", path.parents[2].name))
            )
            session_id = str(
                _json_value(handle.attrs.get("session_id", path.parents[1].name))
            )

            signals: dict[str, dict[str, Any]] = {}
            signal_group = handle.get("signals")
            if signal_group is None:
                raise HSPDataError("missing /signals group")

            inferred_durations: list[float] = []
            for name, dataset in signal_group.items():
                if dataset.ndim != 1:
                    continue
                attrs = {
                    key: _json_value(value) for key, value in dataset.attrs.items()
                }
                fs = float(attrs.get("fs") or 0.0)
                if fs > 0:
                    inferred_durations.append(len(dataset) / fs)
                signals[name.lower()] = {
                    "source_name": name,
                    "length": int(len(dataset)),
                    "dtype": str(dataset.dtype),
                    "fs": fs,
                    "unit": attrs.get("unit", ""),
                    "type": attrs.get("type", ""),
                    "dig_min": attrs.get("dig_min"),
                    "dig_max": attrs.get("dig_max"),
                    "phys_min": attrs.get("phys_min"),
                    "phys_max": attrs.get("phys_max"),
                }

            duration_attr = _json_value(handle.attrs.get("duration_sec"))
            duration_sec = float(duration_attr) if duration_attr is not None else 0.0
            if duration_sec <= 0 and inferred_durations:
                duration_sec = min(inferred_durations)
            if duration_sec <= 0:
                raise HSPDataError("recording duration is missing or non-positive")

            annotations: list[str] = []
            annotation_group = handle.get("annotations")
            if annotation_group is not None:

                def collect_annotation(name: str, obj: Any) -> None:
                    if (
                        isinstance(obj, h5py.Group)
                        and "codes" in obj
                        and "starts" in obj
                    ):
                        annotations.append(name)

                annotation_group.visititems(collect_annotation)

            record.update(
                subject_id=subject_id,
                session_id=session_id,
                split=subject_split(subject_id, split_seed, split_ratios),
                duration_sec=duration_sec,
                attrs={key: _json_value(value) for key, value in handle.attrs.items()},
                signals=signals,
                annotations=sorted(annotations),
                file_size=int(path.stat().st_size),
            )
    except Exception as error:
        record.update(status="error", error=f"{type(error).__name__}: {error}")

    return record


def iter_hsp_files(root: str | Path) -> Iterable[Path]:
    """Yield HSP HDF5 sessions in deterministic BIDS subject/session order."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"HSP root not found: {root_path}")
    for subject_path in sorted(root_path.glob("sub-*")):
        if not subject_path.is_dir():
            continue
        for session_path in sorted(subject_path.glob("ses-*")):
            eeg_path = session_path / "eeg"
            if not eeg_path.is_dir():
                continue
            yield from sorted(eeg_path.glob("*.h5"))


def build_hsp_manifest(
    root: str | Path,
    output_path: str | Path,
    *,
    split_seed: int = 42,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    limit: int | None = None,
    progress_callback: Callable[[int, Path, str], None] | None = None,
) -> dict[str, Any]:
    """Build a JSONL manifest atomically and return a compact scan summary."""
    root_path = Path(root).resolve()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")

    counts = {"files": 0, "ok": 0, "error": 0}
    channel_counts: dict[str, int] = {}
    sample_rate_counts: dict[str, dict[str, int]] = {}

    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for index, path in enumerate(iter_hsp_files(root_path)):
                if limit is not None and index >= limit:
                    break
                record = inspect_hsp_file(
                    path,
                    root=root_path,
                    split_seed=split_seed,
                    split_ratios=split_ratios,
                )
                stream.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
                stream.flush()
                counts["files"] += 1
                counts[record["status"]] += 1
                if progress_callback is not None:
                    progress_callback(counts["files"], path, record["status"])
                if record["status"] != "ok":
                    continue
                for channel_name, info in record["signals"].items():
                    channel_counts[channel_name] = (
                        channel_counts.get(channel_name, 0) + 1
                    )
                    fs_key = f"{float(info['fs']):g}"
                    per_channel = sample_rate_counts.setdefault(channel_name, {})
                    per_channel[fs_key] = per_channel.get(fs_key, 0) + 1
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return {
        **counts,
        "root": str(root_path),
        "manifest": str(output.resolve()),
        "channel_counts": dict(sorted(channel_counts.items())),
        "sample_rate_counts": dict(sorted(sample_rate_counts.items())),
    }


def load_hsp_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            version = record.get("manifest_version")
            if version != MANIFEST_VERSION:
                raise ValueError(
                    f"unsupported manifest version {version!r} at line {line_number}; "
                    f"expected {MANIFEST_VERSION}"
                )
            records.append(record)
    return records


__all__ = [
    "MANIFEST_VERSION",
    "build_hsp_manifest",
    "inspect_hsp_file",
    "iter_hsp_files",
    "load_hsp_manifest",
    "subject_split",
]
