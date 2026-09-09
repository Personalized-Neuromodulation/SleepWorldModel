from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import traceback
import uuid

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy

from .core import (
    CHANNELS, GROUPS, FS, EPOCH_SECONDS, EPOCH_SAMPLES, APPLICABLE, HARD_CODES,
    PRIORITY, CHANNEL_STATUS, FILTER_STATE, FILTER_RELATION, EFFECTIVE_FILTER,
    PROCESS_RESULT, COVERAGE_DETECTION, PSD_CONFIG, METRIC_SPECS, TARGETS,
    jsonable, native, profile_accepted, process_channel, primary_reason, hard_bit,
)
VERSION = "v1.0.0"
SCHEMA_VERSION = "1.0.0-full.1"
PILOT_SESSIONS = ("sub-I0002150057962/ses-1", "sub-I0002150052246/ses-1",
                  "sub-I0002150028664/ses-1")
EXCLUSIONS = ("sub-I0002150026681/ses-2", "sub-I0002150030931/ses-1",
              "sub-I0002150035292/ses-1")
TASKS = {"sleep_stage": ("Stage",), "heart_rate": ("Heart Rate Max", "Heart Rate Min", "Heart Rate Avg"),
         "sao2": ("SaO2 Max", "SaO2 Min")}
TASK_FIELDS = {"sleep_stage": ("stage",), "heart_rate": ("max_bpm", "min_bpm", "mean_bpm"),
               "sao2": ("max_percent", "min_percent")}
TASK_QC = {0: "TASK_INCLUDED", 1: "TASK_SOURCE_EPOCH_MISSING",
           2: "TASK_VALUE_MISSING_OR_NONFINITE", 3: "TASK_VALUE_OUT_OF_RANGE_OR_UNKNOWN_CLASS",
           4: "TASK_FIELD_ORDER_INVALID", 5: "TASK_SOURCE_TIME_UNVERIFIED",
           6: "TASK_H5_STAGE_CONFLICT", 7: "TASK_SOURCE_STAGE_MISMATCH"}
TASK_STRUCTURE = {0: "TASK_STRUCTURE_PASS", 1: "TASK_SOURCE_COVERAGE_MISMATCH",
                  2: "TASK_N_MISMATCH", 3: "TASK_RECORD_RANGE_MISMATCH",
                  4: "TASK_EPOCH_KEY_MISMATCH", 5: "TASK_DURATION_MISMATCH"}
STAGE_MAP = {"N3": 1, "3": 1, "N2": 2, "2": 2, "N1": 3, "1": 3,
             "REM": 4, "R": 4, "WAKE": 5, "W": 5}
MASK_KEYS = ("coverage_valid", "processing_valid", "artifact_valid", "valid")
BIND_KEYS = ("shard_id", "schema_sha256", "processing_config_sha256", "qc_config_sha256",
             "epoch_index_sha256", "channel_order_sha256")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def h5_strings(handle: h5py.File, name: str, values) -> None:
    handle.create_dataset(name, data=np.asarray(list(values), dtype=h5py.string_dtype("utf-8")))


def night_grade(valid: np.ndarray) -> tuple[int, dict[str, float]]:
    hours = valid.sum(axis=0) * EPOCH_SECONDS / 3600.0
    usable = {
        "eeg": float(hours[0:6].max()),
        "eog": float(hours[6:8].max()),
        "ecg": float(hours[8]),
        "emg": float(hours[9:12].max()),
        "airflow": float(hours[12]),
        "snore": float(hours[13]),
        "spo2": float(hours[14]),
    }
    if all(usable[key] > 5 for key in ("eeg", "eog", "emg", "ecg", "airflow", "spo2")):
        grade = 5
    elif all(usable[key] > 5 for key in ("eeg", "eog", "emg", "airflow", "spo2")):
        grade = 4
    elif all(usable[key] > 5 for key in ("eeg", "airflow", "spo2")):
        grade = 3
    elif all(usable[key] > 4 for key in ("eeg", "airflow", "spo2")):
        grade = 2
    else:
        grade = 1
    return grade, usable


def locate_sources(input_root: Path, session_key: str) -> tuple[Path, Path]:
    eeg = input_root.joinpath(*session_key.split("/"), "eeg")
    h5_candidates = sorted(eeg.glob("*.h5"))
    csv_candidates = sorted(path for path in eeg.glob("*sleep_annotations.csv") if "caisr" not in path.name.lower())
    if len(h5_candidates) != 1 or len(csv_candidates) != 1:
        raise ValueError(f"expected one H5 and one non-CAISR task CSV in {eeg}")
    return h5_candidates[0], csv_candidates[0]



def task_arrays(csv_path, count):
    indexed = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Epoch", *(f for fields in TASKS.values() for f in fields)}
        if required - set(reader.fieldnames or ()):
            raise ValueError("TASK_EPOCH_KEY_MISMATCH: missing source columns")
        for row in reader:
            try:
                index = int(row["Epoch"]) - 1
            except (TypeError, ValueError):
                raise ValueError("TASK_EPOCH_KEY_MISMATCH: invalid source epoch")
            if index < 0 or index in indexed:
                raise ValueError("TASK_EPOCH_KEY_MISMATCH: negative/duplicate source epoch")
            indexed[index] = row
    outputs = {}
    for task, fields in TASKS.items():
        width = len(fields)
        shape = (count,) if task == "sleep_stage" else (count, width)
        labels = np.zeros(shape, np.uint8) if task == "sleep_stage" else np.full(shape, np.nan, np.float32)
        codes = np.ones((count, width), np.uint8)
        for index, row in indexed.items():
            if index >= count:
                continue
            if task == "sleep_stage":
                text = (row[fields[0]] or "").strip().upper()
                labels[index] = STAGE_MAP.get(text, 0)
                codes[index, 0] = 0 if labels[index] else 2 if not text else 3
                continue
            for j, field in enumerate(fields):
                try:
                    value = float(row[field])
                except (ValueError, TypeError):
                    value = math.nan
                labels[index, j] = value if math.isfinite(value) else math.nan
                lower, upper = (20, 300) if task == "heart_rate" else (20, 100)
                codes[index, j] = 2 if not math.isfinite(value) or value == 0 else 3 if not lower <= value <= upper else 0
            if np.all(codes[index] == 0):
                vals = labels[index]
                ordered = vals[1] <= vals[2] <= vals[0] if width == 3 else vals[1] <= vals[0]
                if not ordered:
                    codes[index] = 4
        primary = np.min(np.where(codes == 0, 255, codes), axis=1).astype(np.uint8)
        primary[primary == 255] = 0
        outputs[task] = {"labels": labels, "field_qc_code": codes, "field_valid": codes == 0,
                         "qc_code": primary, "valid": np.all(codes == 0, axis=1)}
    missing = sum(i not in indexed for i in range(count))
    extra = sum(i >= count for i in indexed)
    outputs["_coverage"] = {"source_rows": len(indexed), "missing_signal_epochs": missing,
                            "extra_source_epochs": extra, "structure_code": int(bool(missing or extra)),
                            "status": TASK_STRUCTURE[int(bool(missing or extra))]}
    return outputs


def verify_source_alignment(events, tasks, count):
    """Cross-check source CSV epoch IDs against H5's explicit 30 s annotation grid."""
    if events is None:
        raise ValueError("TASK_EPOCH_KEY_MISMATCH: no H5 stage grid")
    mapping = json.loads(events["event_map"])
    if [mapping[str(k)].upper() for k in range(1, 6)] != ["N3", "N2", "N1", "REM", "WAKE"]:
        raise ValueError("TASK_EPOCH_KEY_MISMATCH: stage map")
    codes = np.full(count, -1, np.int16)
    conflict = np.zeros(count, bool)
    off_grid = nonpositive = 0
    for start, end, code in zip(events["starts"], events["ends"], events["codes"], strict=True):
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("TASK_EPOCH_KEY_MISMATCH: nonfinite event time")
        if start < 0 or not math.isclose(start / 30, round(start / 30), abs_tol=1e-6):
            off_grid += 1
            continue
        if end <= start:
            nonpositive += 1
            continue
        first, stop = round(start / 30), min(count, math.floor((end + 1e-6) / 30))
        if first >= stop:
            continue
        conflict[first:stop] |= (codes[first:stop] >= 0) & (codes[first:stop] != code)
        codes[first:stop] = code
    present = tasks["sleep_stage"]["field_qc_code"][:, 0] != 1
    stage = tasks["sleep_stage"]["labels"]
    known = tasks["sleep_stage"]["valid"].copy()
    unverified = present & (codes < 0)
    mismatch = known & (codes >= 0) & ~conflict & (codes != stage)
    # Source annotation defects do not change the signal grid or delete a night.
    # Missing time evidence affects all CSV tasks; conflicting stage values only
    # affect sleep-stage labels. Retain earlier value QC as the primary reason.
    for task in TASKS:
        item = tasks[task]
        fc = item["field_qc_code"]
        fc[unverified[:, None] & (fc == 0)] = 5
        if task == "sleep_stage":
            fc[conflict[:, None] & (fc == 0)] = 6
            fc[mismatch[:, None] & (fc == 0)] = 7
        item["field_valid"] = fc == 0
        item["valid"] = np.all(fc == 0, axis=1)
        primary = np.min(np.where(fc == 0, 255, fc), axis=1).astype(np.uint8)
        primary[primary == 255] = 0
        item["qc_code"] = primary
    # Explicit time coverage is independent of whether stage labels are scored.
    # An all-unscored CSV still has usable timestamps for the HR/SaO2 tasks.
    return {"method": "CSV one-based Epoch vs H5 explicit 30s stage grid", "offset_seconds": 0,
            "known_anchors": int((known & ~unverified & ~conflict & ~mismatch).sum()), "off_grid_events": off_grid,
            "nonpositive_events": nonpositive,
            "status": "PARTIAL" if (unverified | conflict | mismatch).any() else "PASS",
            "time_unverified_epochs": np.flatnonzero(unverified).tolist(),
            "h5_stage_conflict_epochs": np.flatnonzero(conflict).tolist(),
            "source_stage_mismatch_epochs": np.flatnonzero(mismatch).tolist(),
            "scope": "explicit time-grid consistency; known stages checked where present; not independent clinical validation"}


def process_session(input_root_text, session_key, stage_text, channel_threads):
    if session_key in EXCLUSIONS:
        raise ValueError("fixed excluded session")
    started = time.perf_counter()
    h5_path, csv_path = locate_sources(Path(input_root_text), session_key)
    with h5py.File(h5_path, "r") as source:
        root = {k: native(v) for k, v in source.attrs.items()}
        duration = float(root["duration_sec"])
        require(math.isfinite(duration) and 30 <= duration <= 72 * 3600, "invalid record duration")
        count = math.floor(duration / 30)
        require(str(root.get("subject_id")) == session_key.split("/")[0], "source subject identity")
        require(str(root.get("session_id")) == session_key.split("/")[1], "source session identity")
        inputs = []
        for name in CHANNELS:
            if name not in source["signals"]:
                inputs.append((None, {}))
                continue
            ds = source["signals"][name]
            attrs = {k: native(v) for k, v in ds.attrs.items()}
            attrs.update(_source_present=True, _source_samples=len(ds), _duration_sec=duration)
            try:
                # Profile gates run before waveform I/O.
                accepted = profile_accepted(name, attrs)
                data = ds[:] if accepted else None
            except Exception as error:
                data = None
                attrs["_read_error"] = repr(error)
            inputs.append((data, attrs))
        group = source.get("annotations/expert_1/stage")
        events = None if group is None else {key: group[key][:] for key in ("starts", "ends", "codes")}
        if events is not None:
            events["event_map"] = native(group.attrs["event_map"])
    results = [None] * 15
    with ThreadPoolExecutor(max_workers=channel_threads, thread_name_prefix="channel") as pool:
        futures = {pool.submit(process_channel, raw, attrs, name, count): i
                   for i, (name, (raw, attrs)) in enumerate(zip(CHANNELS, inputs, strict=True))}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    tasks = task_arrays(csv_path, count)
    alignment = verify_source_alignment(events, tasks, count)
    record_id = session_key.replace("/", "_")
    grade, usable = night_grade(np.column_stack([r["valid"] for r in results]))
    temp_path = Path(stage_text) / f"session-{record_id}.h5"
    with h5py.File(temp_path, "w") as target:
        target.create_dataset("signals", data=np.stack([r["data"] for r in results], 1))
        for key in (*MASK_KEYS, "hard_flags", "hard_code", "hard_evaluated_flags"):
            target.create_dataset(key, data=np.column_stack([r[key] for r in results]))
        for key in METRIC_SPECS:
            target.create_dataset(f"metrics/{key}", data=np.column_stack([r["metrics"][key] for r in results]))
        for task in TASKS:
            for key, value in tasks[task].items():
                target.create_dataset(f"tasks/{task}/{key}", data=value)
    metadata = {"session_key": session_key, "recording_id": record_id,
                "subject_id": root["subject_id"], "session_id": root["session_id"],
                "start_time_raw": str(root.get("meas_date", "")), "timezone_known": False,
                "duration_sec": duration, "epoch_count": count, "tail_seconds_dropped": duration - count * 30,
                "night_grade": grade, "usable_hours": usable, "task_coverage": tasks["_coverage"],
                "task_alignment": alignment, "source_h5": str(h5_path), "source_task_csv": str(csv_path),
                "source_h5_sha256": sha256_file(h5_path), "source_task_sha256": sha256_file(csv_path),
                "source_hsp_likert_scale": int(root.get("likert_scale", 0)),
                "channel_metadata": [r["metadata"] for r in results],
                "worker_seconds": time.perf_counter() - started, "worker_pid": os.getpid(),
                "channel_threads": channel_threads}
    return str(temp_path), metadata


def index_arrays(records):
    counts = np.asarray([r["epoch_count"] for r in records], np.int32)
    return {
        "records/first_epoch": np.cumsum(np.r_[np.int64(0), counts[:-1]], dtype=np.int64),
        "records/n_epochs": counts,
        "segments/record_index": np.repeat(np.arange(len(records), dtype=np.int32), counts),
        "segments/epoch_in_record": np.concatenate([np.arange(n, dtype=np.int32) for n in counts]),
        "segments/epoch_start_offset_ns": np.concatenate([np.arange(n, dtype=np.int64) * 30_000_000_000 for n in counts]),
    }


def common_index(handle, records):
    h5_strings(handle, "records/recording_id", [r["recording_id"] for r in records])
    for key in ("subject_id", "session_id", "start_time_raw"):
        h5_strings(handle, f"records/{key}", [r[key] for r in records])
    handle.create_dataset("records/timezone_known", data=np.zeros(len(records), bool))
    for key, values in index_arrays(records).items():
        handle.create_dataset(key, data=values)


def index_hash(records):
    digest = hashlib.sha256()
    digest.update(json.dumps([r["recording_id"] for r in records], ensure_ascii=False, separators=(",", ":")).encode())
    for key, values in sorted(index_arrays(records).items()):
        digest.update(key.encode())
        digest.update(values.dtype.str.encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def binding(stage, records):
    return {"shard_id": "00000", "schema_sha256": sha256_file(stage / "config/schema.json"),
            "processing_config_sha256": sha256_file(stage / "config/processing.json"),
            "qc_config_sha256": sha256_file(stage / "config/qc.json"),
            "epoch_index_sha256": index_hash(records),
            "channel_order_sha256": hashlib.sha256(json.dumps(CHANNELS, separators=(",", ":")).encode()).hexdigest()}


def artifact_attrs(kind, records, bindings):
    return {**bindings, "schema_version": SCHEMA_VERSION, "version": VERSION,
            "release_status": bindings.get("release_status", "PILOT_REVIEW_REQUIRED"), "artifact_kind": kind,
            "sample_rate_hz": FS, "epoch_seconds": 30, "epoch_samples": 6000,
            "epoch_count": sum(r["epoch_count"] for r in records), "created_at": utc_now(),
            "normalization": "none"}


def write_record_channel_qc(handle, records):
    metadata = [r["channel_metadata"] for r in records]
    for field, dtype, default in (
        ("channel_available", "bool", False), ("channel_status_code", "u1", 1),
        ("usable_hours", "f8", 0.0),
    ):
        path = "records/channel_usable_hours" if field == "usable_hours" else f"records/{field}"
        handle.create_dataset(path, data=np.asarray([[c.get(field, default) for c in r] for r in metadata], dtype=dtype))
    enum_fields = ("source_filter_state", "effective_filter_state", "filter_relation",
                   "filter_reapply_result", "resample_result", "coverage_detection")
    for field in enum_fields:
        handle.create_dataset(f"records/channel_processing/{field}",
                              data=np.asarray([[c.get(field, 0) for c in r] for r in metadata], np.uint8))
    for field in ("source_fs_hz", "filter_guard_seconds"):
        handle.create_dataset(f"records/channel_processing/{field}",
                              data=np.asarray([[c.get(field, math.nan) for c in r] for r in metadata], np.float64))
    handle.create_dataset("records/channel_processing/source_samples",
                          data=np.asarray([[c.get("source_samples", -1) for c in r] for r in metadata], np.int64))
    for pair_field, hp_key, lp_key in (
        ("parsed_source_hp_lp", "source_hp_hz", "source_lp_hz"),
        ("effective_hp_lp", "effective_hp_hz", "effective_lp_hz"),
    ):
        pairs = []
        for r in metadata:
            pairs.append([c.get("applied_hp_lp") or c.get("parsed_source_hp_lp") or (math.nan, math.nan)
                          if pair_field == "effective_hp_lp" else c.get(pair_field) or (math.nan, math.nan) for c in r])
        for j, field in enumerate((hp_key, lp_key)):
            handle.create_dataset(f"records/channel_processing/{field}", data=np.asarray(pairs, np.float64)[:, :, j])
    handle.create_dataset("records/channel_processing/notch_hz",
        data=np.asarray([[(c.get("parsed_source_notch_hz") or [math.nan])[0] for c in r] for r in metadata], np.float64))
    h5_strings(handle, "records/channel_metadata_json",
               [json.dumps(jsonable(r), ensure_ascii=False) for r in metadata])
    handle.create_dataset("records/night_grade", data=np.asarray([r["night_grade"] for r in records], np.uint8))
    handle.create_dataset("records/grade_evaluated", data=np.ones(len(records), bool))
    handle.create_dataset("records/source_hsp_likert_scale",
                          data=np.asarray([r["source_hsp_likert_scale"] for r in records], np.uint8))


def write_signal_shard(path, temp_paths, records, bindings):
    total = sum(r["epoch_count"] for r in records)
    with h5py.File(path, "w") as target:
        target.attrs.update(artifact_attrs("signal", records, bindings))
        h5_strings(target, "channel_names", CHANNELS)
        h5_strings(target, "channel_units", ["uV"] * 14 + ["%"])
        common_index(target, records)
        write_record_channel_qc(target, records)
        for group, (first, stop) in GROUPS.items():
            width = stop - first
            target.create_dataset(f"signals/{group}", (total, width, 6000), "f4",
                                  chunks=(min(4, total), width, 6000), compression="lzf")
            for key in (*MASK_KEYS, "hard_code"):
                target.create_dataset(f"quality/{group}/{key}", (total, width), "u1" if key == "hard_code" else "bool")
        offset = 0
        for temp_path, record in zip(temp_paths, records, strict=True):
            stop_epoch = offset + record["epoch_count"]
            with h5py.File(temp_path, "r") as source:
                for group, (a, b) in GROUPS.items():
                    target[f"signals/{group}"][offset:stop_epoch] = source["signals"][:, a:b]
                    for key in (*MASK_KEYS, "hard_code"):
                        target[f"quality/{group}/{key}"][offset:stop_epoch] = source[key][:, a:b]
            offset = stop_epoch


def write_qc_shard(path, temp_paths, records, bindings, signal_hash):
    with h5py.File(path, "w") as target:
        target.attrs.update({**artifact_attrs("qc_audit", records, bindings), "signal_sha256": signal_hash})
        for key in ("hard_flags", "hard_code", "hard_evaluated_flags", *[f"metrics/{k}" for k in METRIC_SPECS]):
            arrays = []
            for temp in temp_paths:
                with h5py.File(temp, "r") as source:
                    arrays.append(source[key][:])
            target.create_dataset(key, data=np.concatenate(arrays), compression="lzf")


def write_task_shard(path, task, temp_paths, records, bindings, signal_hash):
    with h5py.File(path, "w") as target:
        target.attrs.update({**artifact_attrs(f"task:{task}", records, bindings),
                             "signal_sha256": signal_hash, "task": task,
                             "field_names": np.asarray(TASK_FIELDS[task], dtype=h5py.string_dtype())})
        common_index(target, records)
        for key in ("labels", "valid", "qc_code", "field_valid", "field_qc_code"):
            arrays = []
            for temp in temp_paths:
                with h5py.File(temp, "r") as source:
                    arrays.append(source[f"tasks/{task}/{key}"][:])
            target.create_dataset(key, data=np.concatenate(arrays), compression="lzf")
        codes = np.asarray([r["task_coverage"]["structure_code"] for r in records], np.uint8)
        target.create_dataset("records/structure_qc_code", data=codes)
        target.create_dataset("records/structure_qc_flags", data=codes)  # only coverage mismatch survives publication
        for name, source_key in (("source_missing_epochs", "missing_signal_epochs"),
                                 ("source_extra_epochs", "extra_source_epochs")):
            target.create_dataset(f"records/{name}",
                                  data=np.asarray([r["task_coverage"][source_key] for r in records], np.int32))
        target.attrs["structure_qc_code"] = np.uint8(codes.max())
        target.attrs["structure_qc_flags"] = np.uint8(np.bitwise_or.reduce(codes))


def write_configs(stage, plan_sha):
    write_json(stage / "config/schema.json", {
        "schema_version": SCHEMA_VERSION, "version": VERSION, "sample_rate_hz": 200,
        "epoch_seconds": 30, "epoch_samples": 6000, "channel_order": CHANNELS, "groups": GROUPS,
        "task_fields": TASK_FIELDS, "task_qc_codes": TASK_QC, "task_structure_codes": TASK_STRUCTURE,
        "channel_status_codes": CHANNEL_STATUS, "metrics": {k: {"dtype": d, "not_evaluated": s}
                                                            for k, (d, s) in METRIC_SPECS.items()},
        "storage": {"chunk_epochs": 4, "compression": "lzf"}, "plan_sha256": plan_sha,
    })
    write_json(stage / "config/processing.json", {
        "version": VERSION, "normalization": "none", "source_filter_explicit": "record_only",
        "source_filter_unknown": "HP4 then LP4; Butterworth SOS; sosfiltfilt odd reflection",
        "unknown_targets": TARGETS, "emg_snore_source200_lp_hz": 95,
        "resample": {"method": "resample_poly", "window": ["kaiser", 5.0], "padtype": "line",
                     "500_to_200": [2, 5], "gap_phase": "global_record_origin"},
        "spo2_resample": "previous_sample_hold", "scope": "whole finite native runs",
        "source_filter_state": FILTER_STATE, "effective_filter_state": EFFECTIVE_FILTER,
        "filter_relation": FILTER_RELATION, "process_result": PROCESS_RESULT,
        "coverage_detection": COVERAGE_DETECTION,
        "plan_sha256": plan_sha,
    })
    write_json(stage / "config/qc.json", {
        "hard_codes": HARD_CODES, "priority": PRIORITY,
        "hard_applicable": ((APPLICABLE[None, :] & (1 << np.arange(7))[:, None]) != 0).tolist(),
        "psd": PSD_CONFIG, "high_amplitude": {"window_seconds": 1, "required_windows": 30,
            "threshold_uv": {"eeg": 1000, "eog": 2000, "ecg": 10000, "emg": 5000},
            "short_spike_per_window_limitation": "accepted_by_user"},
        "flat_std_uv": {"eeg": 0.5, "eog": 0.5, "ecg": 5}, "std_ddof": 0,
        "airflow_flat_codes": {"max_span": 2, "max_unique": 2}, "saturation_fraction": 0.05,
        "spo2_range": [50, 110], "spo2_min_in_range_fraction": 0.70,
        "spo2_availability": {"below_percent": 5, "max_nearzero_fraction": 0.90, "median_below_percent": 5},
        "scope": "30 seconds; filter state per record/channel", "plan_sha256": plan_sha,
    })


def validate_artifacts(signal_path, qc_path, task_paths, records, temp_paths=None):
    total = sum(r["epoch_count"] for r in records)
    signal_hash = sha256_file(signal_path)
    with h5py.File(signal_path) as sf, h5py.File(qc_path) as qf:
        expected_roots = {"hard_flags", "hard_code", "hard_evaluated_flags", "metrics"}
        require(set(qf) == expected_roots, "unexpected QC audit field")
        for key in BIND_KEYS:
            require(sf.attrs[key] == qf.attrs[key], f"QC binding {key}")
        require(qf.attrs["signal_sha256"] == signal_hash, "QC signal checksum")
        require(sf.attrs["epoch_index_sha256"] == index_hash(records), "index digest")
        require(sf["channel_names"].asstr()[:].tolist() == list(CHANNELS), "channel order")
        h, e, code = [qf[k][:] for k in ("hard_flags", "hard_evaluated_flags", "hard_code")]
        for key in ("hard_flags", "hard_evaluated_flags", "hard_code"):
            require(qf[key].shape == (total, 15) and qf[key].dtype == np.dtype("u1"), f"QC shape/dtype {key}")
        require(not np.any(h & (~e & 127)), "hard bit without evaluated bit")
        require(not np.any(e & (~APPLICABLE[None, :] & 127)), "inapplicable evaluated bit")
        require(not np.any((h | e) & 128), "reserved bit")
        require(np.array_equal(code, primary_reason(h)), "priority code")
        for key, (dtype, sentinel) in METRIC_SPECS.items():
            ds = qf[f"metrics/{key}"]
            require(ds.shape == (total, 15) and ds.dtype == np.dtype(dtype), f"metric schema {key}")
        masks = {key: np.concatenate([sf[f"quality/{g}/{key}"][:] for g in GROUPS], 1) for key in MASK_KEYS}
        available = sf["records/channel_available"][:]
        record_index = sf["segments/record_index"][:]
        expected_artifact = ((e & APPLICABLE) == APPLICABLE) & (h == 0)
        require(np.array_equal(masks["artifact_valid"], expected_artifact), "artifact completeness")
        require(np.array_equal(masks["valid"], available[record_index] & masks["coverage_valid"] &
                               masks["processing_valid"] & expected_artifact), "valid conjunction")
        require(np.array_equal(available, sf["records/channel_status_code"][:] == 0), "channel status")
        for key, expected in index_arrays(records).items():
            require(np.array_equal(sf[key][:], expected), f"index {key}")
        for group, (a, b) in GROUPS.items():
            ds = sf[f"signals/{group}"]
            require(ds.shape == (total, b-a, 6000) and ds.dtype == np.dtype("f4"), f"signal {group} shape/dtype")
            require(np.array_equal(sf[f"quality/{group}/hard_code"][:], code[:, a:b]), "signal hard code slice")
            for key in MASK_KEYS:
                require(sf[f"quality/{group}/{key}"].dtype == np.dtype("bool"), "bool mask dtype")
            for start in range(0, total, 64):
                require(np.isfinite(ds[start:start+64]).all(), "nonfinite stored signal")
        offsets = index_arrays(records)["records/first_epoch"]
        for i, r in enumerate(records):
            sl = slice(offsets[i], offsets[i] + r["epoch_count"])
            grade, _ = night_grade(masks["valid"][sl])
            require(grade == int(sf["records/night_grade"][i]), "night grade")
            require(np.allclose(sf["records/channel_usable_hours"][i], masks["valid"][sl].sum(0) / 120, rtol=0, atol=1e-12), "usable hours")
            if temp_paths:
                with h5py.File(temp_paths[i]) as tmp:
                    for group, (a, b) in GROUPS.items():
                        require(np.array_equal(sf[f"signals/{group}"][sl], tmp["signals"][:, a:b]), "packing changed waveforms")
        # Reconstruct flags from stored metrics at safe distance from rounded float32 thresholds.
        for bit_code, metric, threshold, operator in (
            (3, "saturation_fraction", .05, "ge"), (4, "high_amplitude_duration_seconds", 30, "eq"),
            (5, "power_line_fraction", .40, "gt"), (6, "high_frequency_fraction", .50, "gt"),
            (7, "spo2_in_range_fraction", .70, "lt"),
        ):
            checked = (e & hard_bit(bit_code)) != 0
            value = qf[f"metrics/{metric}"][:]
            exact = checked if operator == "eq" else checked & (np.abs(value.astype(float) - threshold) > 1e-7)
            trigger = value == threshold if operator == "eq" else value >= threshold if operator == "ge" else value > threshold if operator == "gt" else value < threshold
            require(np.array_equal((h[exact] & hard_bit(bit_code)) != 0, trigger[exact]), f"metric/flag {metric}")
        for task, path in task_paths.items():
            with h5py.File(path) as tf:
                for key in BIND_KEYS:
                    require(tf.attrs[key] == sf.attrs[key], f"task binding {task}/{key}")
                require(tf.attrs["signal_sha256"] == signal_hash, "task signal hash")
                for key, expected in index_arrays(records).items():
                    require(np.array_equal(tf[key][:], expected), "task epoch/record alignment")
                require(tf["records/recording_id"].asstr()[:].tolist() == [r["recording_id"] for r in records], "task identity")
                shape = (total,) if task == "sleep_stage" else (total, len(TASKS[task]))
                require(tf["labels"].shape == shape, "task label shape")
                fc, fv, tc, tv = [tf[k][:] for k in ("field_qc_code", "field_valid", "qc_code", "valid")]
                require(fc.shape == (total, len(TASKS[task])) and fc.dtype == np.dtype("u1"), "task field code")
                require(np.isin(fc, list(TASK_QC)).all(), "undefined task QC code")
                if task != "sleep_stage":
                    require(not np.isin(fc, [6, 7]).any(), "stage-only task QC on numeric task")
                require(fv.dtype == np.dtype("bool") and tv.shape == (total,), "task mask shape")
                require(np.array_equal(fv, fc == 0) and np.array_equal(tv, fv.all(1)), "task masks")
                expected_code = np.min(np.where(fc == 0, 255, fc), 1).astype(np.uint8)
                expected_code[expected_code == 255] = 0
                require(np.array_equal(tc, expected_code) and np.array_equal(tv, tc == 0), "task main code")
                require(not np.any(tf["records/structure_qc_flags"][:] & 254), "task structural failure")
    return {"status": "PASS", "total_epochs": total, "task_count": len(task_paths), "failures": [],
            "scope": "all published samples, shapes, masks, codes, keys, bindings, packing equality, metric consistency, grades"}


def task_report_lines(stage, records):
    """Explain published label QC separately from source coverage and time alignment."""
    reasons = {
        0: "所有必需标签字段有效", 1: "来源没有对应epoch行",
        2: "字段缺失/非有限或HR、SaO2为0占位", 3: "数值越界或分期未评分/未知类别",
        4: "min/mean/max顺序不成立", 5: "H5显式时间网格缺口，当前task epoch时间未核实",
        6: "H5同一epoch存在冲突分期，仅排除分期标签", 7: "CSV已知分期与H5不一致，仅排除分期标签",
    }
    source_rows = {}
    for record in records:
        path = Path(record["source_task_csv"])
        require("caisr" not in path.name.lower(), "CAISR source is out of scope")
        require(sha256_file(path) == record["source_task_sha256"], "task report source checksum changed")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            source_rows[record["recording_id"]] = {int(row["Epoch"])-1: row for row in csv.DictReader(handle)}

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    def ranges(indices):
        if not len(indices):
            return "—"
        groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1)
        parts = [f"[{int(g[0])*30},{(int(g[-1])+1)*30})" for g in groups[:8]]
        return "; ".join(parts) + (f"; ...共{len(groups)}段" if len(groups)>8 else "")

    lines = ["", "## Tasks", "",
             "标签有效性、来源覆盖和时间对齐分别统计。code 3的未分期标签不代表时间错位；结构code 1表示来源长度/覆盖差异，不能解释为已发布容器错位。",
             "", "| task | labels shape | field QC shape | valid epochs | excluded epochs |",
             "|---|---|---|---:|---:|"]
    code_rows, processing_rows, value_rows = [], [], []
    for task, fields in TASKS.items():
        with h5py.File(stage / f"tasks/{task}/shards/task-00000.h5") as tf:
            main = tf["qc_code"][:]
            lines.append(f"| {task} | {tf['labels'].shape} | {tf['field_qc_code'].shape} | {int((main==0).sum())} | {int((main!=0).sum())} |")
            for code, label in TASK_QC.items():
                n = int((main==code).sum())
                code_rows.append(f"| {task} | {code} | {label} | {n} | {n/len(main):.2%} | {reasons[code]} |")
            for ri, record in enumerate(records):
                first, n = int(tf['records/first_epoch'][ri]), int(tf['records/n_epochs'][ri])
                field_codes = tf['field_qc_code'][first:first+n]
                qc = main[first:first+n]
                missing = int(tf['records/source_missing_epochs'][ri])
                extra = int(tf['records/source_extra_epochs'][ri])
                structural = int(tf['records/structure_qc_code'][ri])
                cause = (f"源CSV {record['task_coverage']['source_rows']}行；超出signal的{extra}个epoch已丢弃；网格内缺失{missing}行"
                         if structural == 1 else "来源覆盖一致")
                counts = "; ".join(f"{TASK_QC[c]}={int((qc==c).sum())}" for c in TASK_QC if c and np.any(qc==c)) or "全部标签有效"
                alignment = record['task_alignment']
                for key, code in (("time_unverified_epochs", 5), ("h5_stage_conflict_epochs", 6), ("source_stage_mismatch_epochs", 7)):
                    ids = np.asarray(alignment.get(key, []), dtype=int)
                    if len(ids) and (task == "sleep_stage" or code == 5):
                        cause += f"；{TASK_QC[code]}: {len(ids)} epochs, {ranges(ids)} s"
                processing_rows.append(f"| {record['session_key']} | {task} | {n}/{record['epoch_count']} | {n*30}/{record['epoch_count']*30} | {alignment['status']} / {alignment['offset_seconds']} s | {alignment['known_anchors']} | {structural}: {TASK_STRUCTURE[structural]} | {cell(cause)}；{counts} |")
                indexed = source_rows[record['recording_id']]
                for j, field in enumerate(fields):
                    for code in (c for c in TASK_QC if c):
                        ids = np.flatnonzero(field_codes[:,j] == code)
                        if not len(ids):
                            continue
                        tokens = Counter("<epoch missing>" if int(k) not in indexed else indexed[int(k)].get(field) or "<empty>" for k in ids)
                        values = "; ".join(f"{token!r} × {cnt}" for token,cnt in tokens.most_common(8))
                        if len(tokens)>8:
                            values += f"; ...共{len(tokens)}种值"
                        value_rows.append(f"| {record['session_key']} | {task} | {TASK_FIELDS[task][j]} | {code}: {TASK_QC[code]} | {len(ids)} | {cell(values)} | {ranges(ids)} |")
    lines += ["", "### Task QC codes", "",
              "按epoch主码计数，每个task各行合计等于N；字段级原因见下表。",
              "", "| task | code | name | epochs | fraction | explanation |", "|---|---:|---|---:|---:|---|", *code_rows,
              "", "### Task processing / alignment", "",
              "N和时长为task/signal，时长单位秒。known anchors统计已知分期比较数，为0但显式网格完整时仍可通过时间结构校验。",
              "", "| session | task | N task/signal | seconds task/signal | grid / offset | known anchors | structure QC | concrete reason |",
              "|---|---|---|---|---|---:|---|---|", *processing_rows,
              "", "### Task invalid fields / source values", "",
              "原始CSV经SHA256验证。区间使用记录起点为0的半开区间[start,end)，单位秒。字段排除数可在同一epoch重叠，不能直接求和为task排除epoch数。",
              "", "| session | task | field | QC code | epochs | raw CSV values | relative seconds |",
              "|---|---|---|---|---:|---|---|", *value_rows,
              "", "分期L被映射为0（未评分/未知类别），field_qc_code=3；不会填成Wake，也不会移动后续epoch。HR/SaO2原始0按缺失占位处理，使用code 2。", ""]
    return lines


def write_reports(stage, records, validation, elapsed, processes, threads):
    signal_path, qc_path = stage / "signals/shards/signal-00000.h5", stage / "signals/qc/qc-00000.h5"
    rows, epoch_rows = [], []
    with h5py.File(signal_path) as sf, h5py.File(qc_path) as qf:
        h, e = qf["hard_flags"][:], qf["hard_evaluated_flags"][:]
        ri, ei = sf["segments/record_index"][:], sf["segments/epoch_in_record"][:]
        for c, channel in enumerate(CHANNELS):
            for code in range(1, 8):
                hit = (h[:, c] & hard_bit(code)) != 0
                evaluated = (e[:, c] & hard_bit(code)) != 0
                denom = int(evaluated.sum())
                rows.append({"channel": channel, "code": code, "name": HARD_CODES[code],
                             "applicable": bool(APPLICABLE[c] & hard_bit(code)),
                             "evaluated_epochs": denom, "hard_epochs": int(hit.sum()),
                             "hit_rate": float(hit.sum()/denom) if denom else None})
                for k in np.flatnonzero(hit):
                    epoch_rows.append({"session_key": records[ri[k]]["session_key"], "channel": channel,
                                       "epoch_in_record": int(ei[k]), "start_seconds": int(ei[k]*30),
                                       "end_seconds": int((ei[k]+1)*30), "hard_code": code, "name": HARD_CODES[code]})
    pq.write_table(pa.Table.from_pylist(rows), stage / "reports/qc_summary.parquet")
    if epoch_rows:
        pq.write_table(pa.Table.from_pylist(epoch_rows), stage / "reports/epoch_hard.parquet")
        with (stage / "reports/epoch_hard.csv").open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(epoch_rows[0]))
            writer.writeheader()
            writer.writerows(epoch_rows)
    lines = ["# I0002 v1.0.0 pilot QC", "",
             f"- Automated validation: **{validation['status']}**; release: **PILOT_REVIEW_REQUIRED**",
             f"- Records: {len(records)}; epochs: {sum(r['epoch_count'] for r in records)}; elapsed: {elapsed:.2f} s",
             f"- Concurrency: {processes} processes x {threads} channel threads; BLAS threads=1",
             "- PSD: Welch, periodic Hann 800, overlap 400, FFT 800, 0.25 Hz, float64, 14 segments, mean.",
             "- High amplitude: all 30 one-second windows exceed threshold; repeated brief-spike limitation accepted.",
             "- No EMG high-frequency-noise test. No warning codes or processing-boundary artifact.",
             "", "## Records", "", "| session | epochs | grade | seconds | source alignment | missing task | extra task |",
             "|---|---:|---:|---:|---|---:|---:|"]
    for r in records:
        lines.append(f"| {r['session_key']} | {r['epoch_count']} | {r['night_grade']} | {r['worker_seconds']:.2f} | {r['task_alignment']['status']} | {r['task_coverage']['missing_signal_epochs']} | {r['task_coverage']['extra_source_epochs']} |")
    lines += ["", "## Hard codes", "", "| channel | code | name | evaluated | hard |", "|---|---:|---|---:|---:|"]
    for row in rows:
        if row["applicable"]:
            lines.append(f"| {row['channel']} | {row['code']} | {row['name']} | {row['evaluated_epochs']} | {row['hard_epochs']} |")
    lines += ["", "## Channel processing", "", "| session | channel | source Hz | effective HP/LP | status | usable h |",
              "|---|---|---:|---|---|---:|"]
    for r in records:
        for c in r["channel_metadata"]:
            lines.append(f"| {r['session_key']} | {c['canonical_channel']} | {c.get('source_fs_hz')} | {c.get('applied_hp_lp') or c.get('parsed_source_hp_lp')} | {c['channel_status_code']} | {c['usable_hours']:.4f} |")
    lines += task_report_lines(stage, records)
    lines += ["", "## Validation and limits", "",
              validation["scope"],
              "- Same-source numerical checks do not independently prove acquisition fidelity.",
              "- These three short real recordings cover explicit 200/500 Hz and unknown 200 Hz filters; full-night throughput is not inferred from them.",
              "- Missing channels, rejected profiles, NaN, clipping, flat, 50/60/80 Hz and task failures are covered by synthetic tests.",
              "- Snore has only the NAN rule. This is not comprehensive snore artifact detection.",
              "- Manual waveform review remains required before full processing.",
              "- epoch_hard.csv/parquet gives recording-relative [start_seconds,end_seconds) for each hard code.", ""]
    (stage / "reports/qc.md").write_text("\n".join(lines), encoding="utf-8")
    # A bounded reproducible read measurement, not a full-night throughput estimate.
    rng = np.random.default_rng(20260905)
    with h5py.File(signal_path) as sf:
        n = sf["signals/eeg"].shape[0]
        order = rng.integers(0, n, size=min(n, 128))
        start = time.perf_counter()
        byte_count = 0
        for i in order:
            for g in GROUPS:
                byte_count += sf[f"signals/{g}"][int(i):int(i)+1].nbytes
        read_seconds = time.perf_counter() - start
    (stage / "reports/io.md").write_text(
        f"# Pilot I/O\n\n- Chunk: 4 epochs, LZF; provisional pending representative full-night benchmark.\n"
        f"- Shard bytes: {signal_path.stat().st_size}\n- Random-read seed: 20260905\n"
        f"- Cached random read: {byte_count/2**20/read_seconds:.2f} MiB/s over {len(order)} epochs.\n"
        "- This short-record cached benchmark does not predict cold-cache full dataset throughput.\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build one current-schema, review-only pilot shard.")
    parser.add_argument("--input-root", type=Path, default=Path(r"I:\HSP\I0002"))
    parser.add_argument("--output-root", type=Path, default=Path(r"I:\HSP\I0002-preprocess\processed\v1.0.0"))
    parser.add_argument("--plan", type=Path, default=Path(r"E:\Code\SleepWorldModel\docs\I0002_PREPROCESS_PLAN.md"))
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--run-name", default=datetime.now().strftime("pilot-%Y%m%d-%H%M%S"))
    args = parser.parse_args(argv)
    if not 1 <= args.processes <= 3 or not 1 <= args.threads <= 15:
        parser.error("processes 1..3, threads 1..15")
    if Path(args.run_name).name != args.run_name or not args.run_name.startswith("pilot-"):
        parser.error("run-name must be a single pilot-* directory name")
    args.output_root.mkdir(parents=True, exist_ok=True)
    destination = args.output_root / args.run_name
    if destination.exists():
        raise FileExistsError(destination)
    stage = args.output_root / f".{args.run_name}-staging-{uuid.uuid4().hex[:8]}"
    for relative in ("signals/shards", "signals/qc", "manifests", "reports", "logs", "config", "work"):
        (stage / relative).mkdir(parents=True, exist_ok=True)
    (stage / "logs/errors.jsonl").touch(exist_ok=False)
    started = time.perf_counter()

    def log(message):
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with (stage / "logs/run.log").open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")

    def progress(status, completed, phase, eta=None):
        write_json(stage / "logs/progress.json", {"status": status, "completed": completed, "total": 3,
            "phase": phase, "elapsed_seconds": time.perf_counter()-started, "eta_seconds": eta})

    try:
        log(f"START processes={args.processes} threads={args.threads} destination={destination}")
        progress("RUNNING", 0, "config", None)
        write_configs(stage, sha256_file(args.plan))
        # Preserve the reviewed inputs to this build; runtime output files are generated by this program.
        shutil.copy2(args.plan, stage / "config/preprocess.md")
        for name in ("core.py", "pipeline.py"):
            shutil.copy2(Path(__file__).with_name(name), stage / "config" / name)
        write_json(stage / "config/runtime.json", {"numpy": np.__version__, "scipy": scipy.__version__,
            "h5py": h5py.__version__, "processes": args.processes, "channel_threads": args.threads,
            "code_sha256": {n: sha256_file(Path(__file__).with_name(n)) for n in ("core.py", "pipeline.py")}})
        outputs = {}
        progress("RUNNING", 0, "session_workers", None)
        with ProcessPoolExecutor(max_workers=args.processes) as pool:
            futures = {pool.submit(process_session, str(args.input_root), s, str(stage/"work"), args.threads): s for s in PILOT_SESSIONS}
            for future in as_completed(futures):
                s = futures[future]
                temp, record = future.result()
                outputs[s] = (Path(temp), record)
                elapsed = time.perf_counter() - started
                eta = elapsed / len(outputs) * (3-len(outputs))
                log(f"SESSION_DONE {s} epochs={record['epoch_count']} worker_seconds={record['worker_seconds']:.2f} pid={record['worker_pid']} ETA_workers_seconds={eta:.1f}")
                for c in record["channel_metadata"]:
                    for error in c.get("errors", []):
                        with (stage/"logs/errors.jsonl").open("a", encoding="utf-8") as fp:
                            fp.write(json.dumps({"session":s,"channel":c["canonical_channel"],"error":error})+"\n")
                progress("RUNNING", len(outputs), "session_workers", eta)
        temps, records = zip(*(outputs[s] for s in PILOT_SESSIONS))
        temps, records = list(temps), list(records)
        bindings = binding(stage, records)
        progress("RUNNING", 3, "packing_and_validation", None)
        log("PACKING single shard")
        sp, qp = stage/"signals/shards/signal-00000.h5", stage/"signals/qc/qc-00000.h5"
        write_signal_shard(sp, temps, records, bindings)
        signal_hash = sha256_file(sp)
        write_qc_shard(qp, temps, records, bindings, signal_hash)
        tasks = {}
        for task in TASKS:
            path = stage/f"tasks/{task}/shards/task-00000.h5"
            path.parent.mkdir(parents=True)
            write_task_shard(path, task, temps, records, bindings, signal_hash)
            tasks[task] = path
        validation = validate_artifacts(sp, qp, tasks, records, temps)
        log(f"VALIDATION_PASS epochs={validation['total_epochs']}")
        write_reports(stage, records, validation, time.perf_counter()-started, args.processes, args.threads)
        rows = [{"recording_id":r["recording_id"],"subject_id":r["subject_id"],"session_id":r["session_id"],
                 "session_key":r["session_key"],"duration_sec":r["duration_sec"],"n_epochs":r["epoch_count"],
                 "shard_id":"00000","first_epoch":int(index_arrays(records)["records/first_epoch"][i]),
                 "night_grade":r["night_grade"],"source_h5":r["source_h5"],
                 "source_h5_sha256":r["source_h5_sha256"],"source_task_sha256":r["source_task_sha256"]}
                for i,r in enumerate(records)]
        pq.write_table(pa.Table.from_pylist(rows),stage/"manifests/records.parquet")
        artifacts = [{"artifact":str(p.relative_to(stage)), "bytes":p.stat().st_size,
                      "sha256":sha256_file(p),"epochs":validation["total_epochs"]} for p in (sp,qp,*tasks.values())]
        pq.write_table(pa.Table.from_pylist(artifacts),stage/"manifests/shards.parquet")
        for task, p in tasks.items():
            pq.write_table(pa.Table.from_pylist([{"shard_id":"00000","path":str(p.relative_to(stage)),
                "signal_sha256":signal_hash,"epoch_index_sha256":bindings["epoch_index_sha256"]}]),
                stage/f"tasks/{task}/manifest.parquet")
            write_json(stage/f"tasks/{task}/dataset.json",{"status":"PILOT_REVIEW_REQUIRED","task":task,"version":VERSION,"shards":1})
        write_json(stage/"signals/dataset.json",{"status":"PILOT_REVIEW_REQUIRED","version":VERSION,"shards":1,"epochs":validation["total_epochs"]})
        elapsed = time.perf_counter()-started
        write_json(stage/"manifests/release.json",{"version":VERSION,"schema_version":SCHEMA_VERSION,
            "status":"PILOT_REVIEW_REQUIRED","validation":validation,"records":3,"epochs":validation["total_epochs"],
            "sessions":PILOT_SESSIONS,"process_workers":args.processes,"channel_threads_per_process":args.threads,
            "plan_sha256":sha256_file(args.plan),"artifacts":artifacts,"elapsed_seconds":elapsed,
            "automated_qc":"PASS","manual_review":"PENDING","source_records":records})
        # Only remove this run's own temporary session files after verified publication.
        for p in temps:
            require(p.resolve().parent == (stage/"work").resolve(), "temporary path safety")
            p.unlink()
        (stage/"work").rmdir()
        progress("PILOT_REVIEW_REQUIRED",3,"complete",0)
        log(f"COMPLETE elapsed_seconds={elapsed:.2f}")
        stage.rename(destination)
        print(f"PILOT_PATH={destination}",flush=True)
        return 0
    except Exception as error:
        log(f"FAILED {error!r}")
        with (stage/"logs/errors.jsonl").open("a",encoding="utf-8") as fp:
            fp.write(json.dumps({"error":repr(error),"traceback":traceback.format_exc()})+"\n")
        progress("FAILED",len(locals().get("outputs",{})),"failed",None)
        print(traceback.format_exc(),flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
