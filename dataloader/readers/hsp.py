"""Read committed I0002 v1.0.0 artifacts without importing preprocessing code."""

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path, PureWindowsPath

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch

from ..schema import DatasetMetadata

CHANNELS = {
    "eeg": ("f3-m2", "f4-m1", "c3-m2", "c4-m1", "o1-m2", "o2-m1"),
    "eog": ("e1", "e2"),
    "ecg": ("ecg",),
    "emg": ("chin1-chin2", "lat", "rat"),
    "respiratory": ("airflow", "snore", "spo2"),
}
TASKS = ("sleep_stage", "heart_rate", "sao2")
BINDINGS = (
    "shard_id",
    "schema_sha256",
    "processing_config_sha256",
    "qc_config_sha256",
    "epoch_index_sha256",
    "channel_order_sha256",
)
QUALITY = ("coverage_valid", "processing_valid", "artifact_valid", "valid", "hard_code")
SCHEMAS = ("1.0.0-pilot.2", "1.0.0-full.1")


class HSPReleaseReader:
    def __init__(
        self,
        root: Path,
        *,
        version="v1.0.0",
        tasks=(),
        max_open_files=8,
        verify_checksums=False,
    ):
        self.root = root.resolve()
        self.tasks = tuple(tasks)
        if len(set(self.tasks)) != len(self.tasks) or set(self.tasks) - set(TASKS):
            raise ValueError(f"tasks must be a unique subset of {TASKS}")
        if version != "v1.0.0" or max_open_files < 1:
            raise ValueError(
                "HSP reader supports v1.0.0 and requires max_open_files >= 1"
            )
        release = json.loads(
            (self.root / "manifests/release.json").read_text(encoding="utf-8")
        )
        if release["version"] != version or release["status"] not in (
            "COMPLETE",
            "COMPLETE_WITH_FAILURES",
        ):
            raise ValueError("expected a completed, version-matched HSP release")
        self.metadata = DatasetMetadata(
            "hsp",
            version,
            CHANNELS.copy(),
            {k: 200.0 for k in CHANNELS},
            {
                k: tuple("%" if c == "spo2" else "uV" for c in v)
                for k, v in CHANNELS.items()
            },
        )
        self.max_open_files = max_open_files
        self.verify_checksums = verify_checksums
        self._handles = OrderedDict()
        self._pid = os.getpid()
        self._commits = {}
        self._artifacts = {}
        for row in pq.read_table(self.root / "manifests/shards.parquet").to_pylist():
            key = self._key(row["artifact"])
            self._path(key)
            if key in self._artifacts:
                raise ValueError(f"duplicate artifact: {key}")
            self._artifacts[key] = row
        self.records = [
            r
            for r in pq.read_table(self.root / "manifests/records.parquet").to_pylist()
            if r["status"] == "QC_PASSED"
        ]
        seen, ranges = set(), {}
        for row in self.records:
            identity = row["recording_id"]
            night_grade = int(row.get("night_grade", 0))
            if (
                identity in seen
                or int(row["first_epoch"]) < 0
                or int(row["n_epochs"]) <= 0
                or not 1 <= night_grade <= 5
            ):
                raise ValueError(
                    "duplicate recording or invalid epoch range/night grade in manifest"
                )
            seen.add(identity)
            shard = str(row["shard_id"])
            if not shard.isdigit():
                raise ValueError("invalid shard id")
            key = f"signals/shards/signal-{shard}.h5"
            artifact = self._artifacts.get(key)
            end = int(row["first_epoch"]) + int(row["n_epochs"])
            if artifact is None or end > int(artifact["epochs"]):
                raise ValueError(
                    "record range is not covered by a published signal artifact"
                )
            ranges.setdefault(shard, []).append((int(row["first_epoch"]), end))
        for spans in ranges.values():
            spans.sort()
            if any(b[0] < a[1] for a, b in zip(spans, spans[1:])):
                raise ValueError("overlapping recording ranges")

    @staticmethod
    def _key(path):
        return str(path).replace("\\", "/")

    def _path(self, key):
        path = (self.root / key).resolve()
        if (
            PureWindowsPath(key).is_absolute()
            or not path.is_relative_to(self.root)
            or path.suffix != ".h5"
        ):
            raise ValueError(f"unsafe or partial artifact path: {key}")
        return path

    def _open(self, key):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if key in self._handles:
            self._handles.move_to_end(key)
            return self._handles[key]
        row = self._artifacts.get(key)
        if row is None:
            raise ValueError(f"artifact is absent from manifest: {key}")
        shard = str(row["shard_id"])
        if not shard.isdigit():
            raise ValueError("invalid shard id")
        if shard not in self._commits:
            commit = json.loads(
                (self.root / f"manifests/commits/shard-{shard}.json").read_text(
                    encoding="utf-8"
                )
            )
            if commit["status"] != "QC_PASSED" or str(commit["shard_id"]) != shard:
                raise ValueError("shard is not committed")
            self._commits[shard] = {
                self._key(a["artifact"]): a for a in commit["artifacts"]
            }
        committed = self._commits[shard].get(key)
        if committed is None or any(
            committed[k] != row[k] for k in ("bytes", "sha256", "epochs", "shard_id")
        ):
            raise ValueError("manifest and commit disagree")
        path = self._path(key)
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"artifact size mismatch: {key}")
        if self.verify_checksums:
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                    raise ValueError(f"artifact checksum mismatch: {key}")
        handle = h5py.File(path, "r")
        if (
            str(handle.attrs.get("shard_id")) != shard
            or handle.attrs.get("version") != self.metadata.version
            or handle.attrs.get("schema_version") not in SCHEMAS
            or handle.attrs.get("release_status") != "QC_PASSED"
            or handle.attrs.get("epoch_count") != row["epochs"]
        ):
            handle.close()
            raise ValueError("unsupported or mismatched artifact header")
        while len(self._handles) >= self.max_open_files:
            self._handles.popitem(last=False)[1].close()
        self._handles[key] = handle
        return handle

    @staticmethod
    def _index(handle, record, sl, start, count):
        index = int(handle["segments/record_index"][sl.start])
        if (
            handle["records/recording_id"].asstr()[index] != record["recording_id"]
            or handle["records/subject_id"].asstr()[index] != record["subject_id"]
            or handle["records/session_id"].asstr()[index] != record["session_id"]
            or handle["records/first_epoch"][index] != record["first_epoch"]
            or handle["records/n_epochs"][index] != record["n_epochs"]
            or not np.all(handle["segments/record_index"][sl] == index)
            or not np.array_equal(
                handle["segments/epoch_in_record"][sl], np.arange(start, start + count)
            )
            or not np.array_equal(
                handle["segments/epoch_start_offset_ns"][sl],
                np.arange(start, start + count) * 30_000_000_000,
            )
        ):
            raise ValueError("recording identity or epoch alignment mismatch")
        return index

    def read_window(self, record, start, count):
        if start < 0 or count <= 0 or start + count > int(record["n_epochs"]):
            raise ValueError("window crosses recording boundary")
        shard = str(record["shard_id"])
        key = f"signals/shards/signal-{shard}.h5"
        sf = self._open(key)
        if (
            sf.attrs.get("artifact_kind") != "signal"
            or sf.attrs.get("sample_rate_hz") != 200
            or sf.attrs.get("epoch_seconds") != 30
            or sf.attrs.get("epoch_samples") != 6000
            or sf["channel_names"].asstr()[:].tolist()
            != [c for v in CHANNELS.values() for c in v]
            or sf["channel_units"].asstr()[:].tolist() != ["uV"] * 14 + ["%"]
        ):
            raise ValueError("unsupported signal layout")
        sl = slice(
            int(record["first_epoch"]) + start,
            int(record["first_epoch"]) + start + count,
        )
        idx = self._index(sf, record, sl, start, count)
        manifest_grade = int(record["night_grade"])
        if (
            "night_grade" not in sf["records"]
            or "grade_evaluated" not in sf["records"]
            or not bool(sf["records/grade_evaluated"][idx])
            or int(sf["records/night_grade"][idx]) != manifest_grade
        ):
            raise ValueError("manifest and signal shard night grade disagree")
        signals, quality, available = {}, {}, {}
        offset = 0
        for group, channels in CHANNELS.items():
            values = sf[f"signals/{group}"][sl]
            if (
                values.shape != (count, len(channels), 6000)
                or values.dtype != np.float32
                or not np.isfinite(values).all()
            ):
                raise ValueError(f"invalid signal array: {group}")
            signals[group] = torch.from_numpy(values)
            quality[group] = {
                k: torch.from_numpy(sf[f"quality/{group}/{k}"][sl]) for k in QUALITY
            }
            if any(v.shape != (count, len(channels)) for v in quality[group].values()):
                raise ValueError("invalid quality shape")
            available[group] = torch.from_numpy(
                sf["records/channel_available"][idx, offset : offset + len(channels)]
            )
            masks = quality[group]
            if available[group].dtype != torch.bool or any(
                masks[k].dtype != torch.bool for k in QUALITY[:-1]
            ):
                raise ValueError("quality and availability masks must be boolean")
            expected = (
                available[group][None]
                & masks["coverage_valid"]
                & masks["processing_valid"]
                & masks["artifact_valid"]
            )
            if not torch.equal(expected, masks["valid"]):
                raise ValueError("quality mask conjunction mismatch")
            offset += len(channels)
        bindings = {k: sf.attrs[k] for k in BINDINGS}
        task_values = {}
        for task in self.tasks:
            tf = self._open(f"tasks/{task}/shards/task-{shard}.h5")
            if (
                tf.attrs.get("task") != task
                or tf.attrs.get("artifact_kind") != f"task:{task}"
                or tf.attrs.get("signal_sha256") != self._artifacts[key]["sha256"]
                or any(tf.attrs.get(k) != v for k, v in bindings.items())
            ):
                raise ValueError("task/signal binding mismatch")
            self._index(tf, record, sl, start, count)
            task_values[task] = {
                k: torch.from_numpy(tf[k][sl])
                for k in ("labels", "valid", "qc_code", "field_valid", "field_qc_code")
            }
            task_values[task]["field_names"] = tuple(
                str(v) for v in tf.attrs["field_names"]
            )
            values = task_values[task]
            width = len(values["field_names"])
            shape = (count,) if task == "sleep_stage" else (count, width)
            if (
                values["labels"].shape != shape
                or values["valid"].shape != (count,)
                or values["qc_code"].shape != (count,)
                or values["field_valid"].shape != (count, width)
                or values["field_qc_code"].shape != (count, width)
                or values["valid"].dtype != torch.bool
                or values["field_valid"].dtype != torch.bool
                or not torch.equal(values["valid"], values["field_valid"].all(dim=1))
                or not torch.equal(values["field_valid"], values["field_qc_code"] == 0)
                or not torch.equal(values["valid"], values["qc_code"] == 0)
            ):
                raise ValueError("task mask or field shape mismatch")
        return {
            "signals": signals,
            "quality": quality,
            "available_mask": available,
            "channel_mask": {k: v.clone() for k, v in available.items()},
            "sample_rates": self.metadata.sample_rates.copy(),
            "channel_names": self.metadata.channels.copy(),
            "units": self.metadata.units.copy(),
            "tasks": task_values,
            "epoch_mask": torch.ones(count, dtype=torch.bool),
            "epoch_in_record": torch.arange(start, start + count),
            "epoch_start_offset_ns": torch.arange(start, start + count)
            * 30_000_000_000,
            "recording_id": record["recording_id"],
            "subject_id": record["subject_id"],
            "session_id": record["session_id"],
            "dataset": "hsp",
            "version": self.metadata.version,
            "night_grade": manifest_grade,
            "start_sec": start * 30.0,
            "duration_sec": count * 30.0,
            "recording_duration_sec": float(record["duration_sec"]),
            "path": str(self._path(key)),
        }

    def close(self):
        for handle in getattr(self, "_handles", {}).values():
            handle.close()
        self._handles = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["_commits"] = {}
        return state

    def __del__(self):
        self.close()
