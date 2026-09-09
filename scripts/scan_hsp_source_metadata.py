"""Read-only inventory of HSP H5/EDF/BIDS metadata.

This scanner never reads signal samples.  It inventories the source files and
extracts per-channel header metadata so preprocessing rules can be designed
from the actual cohort rather than from a handful of recordings.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import statistics
import time

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CHANNELS = (
    "f3-m2", "f4-m1", "c3-m2", "c4-m1", "o1-m2", "o2-m1",
    "e1", "e2", "ecg", "chin1-chin2", "lat", "rat", "airflow",
    "snore", "spo2",
)
ALIASES = {
    "f3-m2": ("f3-m2",), "f4-m1": ("f4-m1",),
    "c3-m2": ("c3-m2",), "c4-m1": ("c4-m1",),
    "o1-m2": ("o1-m2",), "o2-m1": ("o2-m1",),
    "e1": ("e1",), "e2": ("e2",), "ecg": ("ecg", "ekg"),
    "chin1-chin2": ("chin1-chin2", "chin"), "lat": ("lat",), "rat": ("rat",),
    "airflow": ("airflow", "npt", "ptaf"), "snore": ("snore",),
    "spo2": ("spo2", "sao2"),
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace").strip()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def text(value) -> str:
    value = scalar(value)
    return "" if value is None else str(value).strip()


def number(value):
    try:
        result = float(text(value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def normalise_channel(value: str) -> str:
    value = value.strip().lower().replace("–", "-").replace("—", "-")
    value = re.sub(r"\s+", "", value)
    return value.replace("_", "-")


ALIAS_TO_CANONICAL = {
    normalise_channel(alias): canonical
    for canonical, aliases in ALIASES.items()
    for alias in aliases
}


FILTER_PATTERNS = {
    "hp_hz": re.compile(r"\bHP\s*:?\s*([-+]?\d+(?:\.\d+)?)\s*Hz", re.I),
    "lp_hz": re.compile(r"\bLP\s*:?\s*([-+]?\d+(?:\.\d+)?)\s*Hz", re.I),
    "notch_hz": re.compile(r"\bN\s*:?\s*([-+]?\d+(?:\.\d+)?)\s*Hz", re.I),
}


def parse_prefilter(value: str) -> dict:
    result = {}
    for key, pattern in FILTER_PATTERNS.items():
        match = pattern.search(value or "")
        result[key] = float(match.group(1)) if match else None
    result["filter_parse_ok"] = bool(value) and any(result[k] is not None for k in FILTER_PATTERNS)
    return result


def read_edf_header(path: Path) -> tuple[dict, list[dict]]:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError("EDF fixed header is shorter than 256 bytes")

        def field(start: int, size: int) -> str:
            return fixed[start:start + size].decode("latin-1", "replace").strip()

        header_bytes = int(field(184, 8))
        record_count = int(field(236, 8))
        record_seconds = float(field(244, 8))
        signal_count = int(field(252, 4))
        if signal_count <= 0 or header_bytes < 256 + signal_count * 256:
            raise ValueError("invalid EDF channel count or header size")
        variable = handle.read(header_bytes - 256)
        if len(variable) != header_bytes - 256:
            raise ValueError("EDF variable header is truncated")

    widths = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32)
    names = ("label", "transducer", "unit", "phys_min", "phys_max",
             "dig_min", "dig_max", "prefilter", "samples_per_record", "reserved")
    columns: dict[str, list[str]] = {}
    offset = 0
    for name, width in zip(names, widths):
        block = variable[offset:offset + width * signal_count]
        columns[name] = [
            block[i * width:(i + 1) * width].decode("latin-1", "replace").strip()
            for i in range(signal_count)
        ]
        offset += width * signal_count
    channels = []
    for i in range(signal_count):
        samples_per_record = number(columns["samples_per_record"][i])
        fs = samples_per_record / record_seconds if samples_per_record is not None and record_seconds > 0 else None
        sample_count = int(samples_per_record * record_count) if samples_per_record is not None and record_count >= 0 else None
        prefilter = columns["prefilter"][i]
        channels.append({
            "raw_channel": columns["label"][i],
            "unit": columns["unit"][i],
            "dtype": "int16_edf",
            "sample_count": sample_count,
            "fs_hz": fs,
            "duration_seconds": sample_count / fs if sample_count is not None and fs else None,
            "dig_min": number(columns["dig_min"][i]),
            "dig_max": number(columns["dig_max"][i]),
            "phys_min": number(columns["phys_min"][i]),
            "phys_max": number(columns["phys_max"][i]),
            "prefilter_raw": prefilter,
            "transducer": columns["transducer"][i],
            **parse_prefilter(prefilter),
        })
    duration = record_count * record_seconds if record_count >= 0 else None
    return ({
        "edf_version": field(0, 8), "edf_start_date": field(168, 8),
        "edf_start_time": field(176, 8), "edf_header_bytes": header_bytes,
        "edf_reserved": field(192, 44), "edf_record_count": record_count,
        "edf_record_seconds": record_seconds, "edf_signal_count": signal_count,
        "edf_duration_seconds": duration,
    }, channels)


def read_tsv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    result = []
    for row in rows:
        label = text(row.get("name"))
        if not label:
            continue
        hp = number(row.get("lowcutoff"))
        lp = number(row.get("high_cutoff"))
        result.append({
            "raw_channel": label, "channel_type": text(row.get("type")),
            "unit": text(row.get("units")), "fs_hz": number(row.get("sampling_frequency")),
            "hp_hz": hp, "lp_hz": lp, "notch_hz": None,
            "filter_parse_ok": hp is not None or lp is not None,
            "prefilter_raw": f"HP:{hp} LP:{lp}", "status": text(row.get("status")),
            "status_description": text(row.get("status_description")),
            "description": text(row.get("description")),
        })
    return result


def h5_headers(path: Path) -> tuple[dict, list[dict]]:
    with h5py.File(path, "r") as handle:
        root = {k: scalar(v) for k, v in handle.attrs.items()}
        if "signals" not in handle:
            raise ValueError("H5 has no /signals group")
        channels = []
        for name, dataset in handle["signals"].items():
            attrs = {k: scalar(v) for k, v in dataset.attrs.items()}
            fs = number(attrs.get("fs"))
            count = int(dataset.shape[0]) if dataset.ndim == 1 else None
            prefilter = text(attrs.get("prefilter"))
            channels.append({
                "raw_channel": name, "unit": text(attrs.get("unit")),
                "dtype": str(dataset.dtype), "ndim": dataset.ndim,
                "sample_count": count, "fs_hz": fs,
                "duration_seconds": count / fs if count is not None and fs else None,
                "dig_min": number(attrs.get("dig_min")), "dig_max": number(attrs.get("dig_max")),
                "phys_min": number(attrs.get("phys_min")), "phys_max": number(attrs.get("phys_max")),
                "prefilter_raw": prefilter, "signal_type": text(attrs.get("type")),
                "value_domain": text(attrs.get("value_domain")),
                **parse_prefilter(prefilter),
            })
    return root, channels


def add_channel_identity(rows: list[dict], source: str, session_key: str) -> list[dict]:
    result = []
    for row in rows:
        normalised = normalise_channel(row["raw_channel"])
        result.append({
            "session_key": session_key, "source": source,
            "normalised_channel": normalised,
            "canonical_channel": ALIAS_TO_CANONICAL.get(normalised, ""),
            "selected_for_model": False,
            **row,
        })
    # A source may expose both a preferred channel and one of its fallbacks
    # (for example H5 airflow + ptaf). Select exactly one according to ALIASES
    # order while retaining every candidate in the inventory.
    for canonical, aliases in ALIASES.items():
        ranks = {normalise_channel(alias): i for i, alias in enumerate(aliases)}
        candidates = [r for r in result if r["canonical_channel"] == canonical]
        if candidates:
            min(candidates, key=lambda r: (ranks.get(r["normalised_channel"], 10_000), r["normalised_channel"]))[
                "selected_for_model"
            ] = True
    return result


def scan_session(task: dict) -> dict:
    started = time.perf_counter()
    row, local_root, edf_root = task["row"], Path(task["local_root"]), Path(task["edf_root"])
    subject = text(row.get("BIDSFolder"))
    session_value = text(row.get("SessionID"))
    session = session_value if session_value.startswith("ses-") else f"ses-{session_value}"
    session_key = f"{subject}/{session}"
    local_eeg = local_root / subject / session / "eeg"
    remote_eeg = edf_root / subject / session / "eeg"
    stem = f"{subject}_{session}"
    h5_path = local_eeg / f"{stem}.h5"
    edf_path = remote_eeg / f"{stem}_task-PSG_eeg.edf"
    tsv_path = local_eeg / f"{stem}_task-PSG_channels.tsv"
    json_path = local_eeg / f"{stem}_task-PSG_eeg.json"
    manual_csv = local_eeg / f"{stem}_task-psg_sleep_annotations.csv"
    issues: list[dict] = []
    channels: list[dict] = []
    session_row = {
        "annotation_index": task["index"], "patient_id": text(row.get("BDSPPatientID")),
        "subject_id": subject, "session_id": session, "session_key": session_key,
        "metadata_duration_seconds": number(row.get("RecordingDuration")),
        "metadata_start": text(row.get("StartDateTime")), "metadata_end": text(row.get("EndDateTime")),
        "local_eeg_exists": local_eeg.is_dir(), "remote_eeg_exists": remote_eeg.is_dir(),
        "h5_exists": h5_path.is_file(), "edf_exists": edf_path.is_file(),
        "channels_tsv_exists": tsv_path.is_file(), "bids_json_exists": json_path.is_file(),
        "manual_sleep_csv_exists": manual_csv.is_file(),
        "h5_path": str(h5_path), "edf_path": str(edf_path),
        "h5_bytes": h5_path.stat().st_size if h5_path.is_file() else None,
        "edf_bytes": edf_path.stat().st_size if edf_path.is_file() else None,
    }

    def issue(code: str, detail: str = ""):
        issues.append({"session_key": session_key, "code": code, "detail": detail})

    if h5_path.is_file():
        try:
            h5_root, h5_channels = h5_headers(h5_path)
            session_row.update({
                "h5_subject_id": text(h5_root.get("subject_id")),
                "h5_session_id": text(h5_root.get("session_id")),
                "h5_duration_seconds": number(h5_root.get("duration_sec")),
                "h5_channel_count": len(h5_channels),
            })
            channels.extend(add_channel_identity(h5_channels, "h5", session_key))
            if session_row["h5_subject_id"] != subject or session_row["h5_session_id"] != session:
                issue("H5_IDENTITY_MISMATCH", f"{session_row['h5_subject_id']}/{session_row['h5_session_id']}")
        except Exception as error:
            session_row["h5_error"] = f"{type(error).__name__}: {error}"
            issue("H5_HEADER_ERROR", session_row["h5_error"])
    else:
        issue("H5_MISSING")

    if edf_path.is_file():
        try:
            edf_header, edf_channels = read_edf_header(edf_path)
            session_row.update(edf_header)
            channels.extend(add_channel_identity(edf_channels, "edf", session_key))
        except Exception as error:
            session_row["edf_error"] = f"{type(error).__name__}: {error}"
            issue("EDF_HEADER_ERROR", session_row["edf_error"])
    else:
        issue("EDF_MISSING")

    if tsv_path.is_file():
        try:
            tsv_channels = read_tsv(tsv_path)
            session_row["tsv_channel_count"] = len(tsv_channels)
            channels.extend(add_channel_identity(tsv_channels, "tsv", session_key))
        except Exception as error:
            session_row["tsv_error"] = f"{type(error).__name__}: {error}"
            issue("TSV_READ_ERROR", session_row["tsv_error"])
    else:
        issue("CHANNELS_TSV_MISSING")

    if json_path.is_file():
        try:
            document = json.loads(json_path.read_text(encoding="utf-8-sig"))
            session_row.update({
                "manufacturer": text(document.get("Manufacturer")),
                "power_line_frequency": text(document.get("PowerLineFrequency")),
                "bids_sampling_frequency": number(document.get("SamplingFrequency")),
                "bids_software_filters": text(document.get("SoftwareFilters")),
                "recording_type": text(document.get("RecordingType")),
                "eeg_reference": text(document.get("EEGReference")),
            })
        except Exception as error:
            session_row["bids_json_error"] = f"{type(error).__name__}: {error}"
            issue("BIDS_JSON_ERROR", session_row["bids_json_error"])

    if not manual_csv.is_file():
        issue("MANUAL_SLEEP_CSV_MISSING")

    present_by_source = {
        source: {r["canonical_channel"] for r in channels if r["source"] == source and r["selected_for_model"]}
        for source in ("h5", "edf", "tsv")
    }
    for source in ("h5", "edf", "tsv"):
        for canonical in CHANNELS:
            if canonical not in present_by_source[source]:
                issue(f"{source.upper()}_SELECTED_CHANNEL_MISSING", canonical)

    # Exact metadata comparisons; numeric tolerances prevent string-format false positives.
    by_key = {
        (r["source"], r["canonical_channel"]): r
        for r in channels if r["canonical_channel"] and r["selected_for_model"]
    }
    for canonical in CHANNELS:
        h5_row, edf_row, tsv_row = (by_key.get((source, canonical)) for source in ("h5", "edf", "tsv"))
        if h5_row and edf_row:
            for field, tolerance in (("fs_hz", 1e-6), ("hp_hz", 5e-4), ("lp_hz", 5e-4),
                                     ("dig_min", 1e-6), ("dig_max", 1e-6),
                                     ("phys_min", 1e-6), ("phys_max", 1e-6)):
                left, right = h5_row.get(field), edf_row.get(field)
                if left is None or right is None:
                    if left != right:
                        issue("H5_EDF_METADATA_UNKNOWN", f"{canonical}:{field}:{left}!={right}")
                elif not math.isclose(float(left), float(right), rel_tol=1e-8, abs_tol=tolerance):
                    issue("H5_EDF_METADATA_MISMATCH", f"{canonical}:{field}:{left}!={right}")
            if h5_row.get("unit", "").lower() != edf_row.get("unit", "").lower():
                issue("H5_EDF_UNIT_MISMATCH", f"{canonical}:{h5_row.get('unit')}!={edf_row.get('unit')}")
        if h5_row and tsv_row:
            for field, tolerance in (("fs_hz", 1e-6), ("hp_hz", 5e-4), ("lp_hz", 5e-4)):
                left, right = h5_row.get(field), tsv_row.get(field)
                if left is not None and right is not None and not math.isclose(float(left), float(right), rel_tol=1e-8, abs_tol=tolerance):
                    issue("H5_TSV_METADATA_MISMATCH", f"{canonical}:{field}:{left}!={right}")

    md = session_row.get("metadata_duration_seconds")
    for source_field in ("h5_duration_seconds", "edf_duration_seconds"):
        duration = session_row.get(source_field)
        if md is not None and duration is not None:
            session_row[f"{source_field}_minus_metadata"] = duration - md
            if abs(duration - md) > 30:
                issue("DURATION_MISMATCH_GT30S", f"{source_field}:{duration-md:+.3f}")
    session_row["scan_seconds"] = time.perf_counter() - started
    return {"session": session_row, "channels": channels, "issues": issues}


def scan_batch(payload: tuple[list[dict], int]) -> list[dict]:
    """Scan one process batch using a small thread pool for independent sessions."""
    tasks, thread_count = payload
    if thread_count == 1:
        return [scan_session(task) for task in tasks]
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        return list(executor.map(scan_session, tasks))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")


def quantiles(values: list[float]) -> dict:
    array = np.asarray([v for v in values if v is not None and math.isfinite(v)], np.float64)
    if not len(array):
        return {"count": 0}
    points = np.quantile(array, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    return dict(zip(("min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"), points.tolist())) | {"count": len(array), "mean": float(array.mean())}


def aggregate(sessions: list[dict], channels: list[dict], issues: list[dict], elapsed: float) -> tuple[dict, list[dict], list[dict]]:
    issue_counts: dict[str, int] = {}
    for row in issues:
        issue_counts[row["code"]] = issue_counts.get(row["code"], 0) + 1
    filter_counts: dict[tuple, set[str]] = {}
    for row in channels:
        key = (row["source"], row["canonical_channel"] or row["normalised_channel"],
               row["raw_channel"], row["selected_for_model"], row.get("prefilter_raw", ""),
               row.get("hp_hz"), row.get("lp_hz"), row.get("notch_hz"),
               row.get("fs_hz"), row.get("unit", ""))
        filter_counts.setdefault(key, set()).add(row["session_key"])
    filter_rows = []
    total_sessions = len(sessions)
    for key, session_keys in filter_counts.items():
        source, channel, raw_channel, selected, raw, hp, lp, notch, fs, unit = key
        filter_rows.append({
            "source": source, "channel": channel, "raw_channel": raw_channel,
            "selected_for_model": selected, "prefilter_raw": raw,
            "hp_hz": hp, "lp_hz": lp, "notch_hz": notch, "fs_hz": fs, "unit": unit,
            "recording_count": len(session_keys),
            "percent_of_all_recordings": 100.0 * len(session_keys) / total_sessions if total_sessions else 0,
        })
    filter_rows.sort(key=lambda r: (r["source"], r["channel"], -r["recording_count"], r["prefilter_raw"]))

    coverage_rows = []
    for source in ("h5", "edf", "tsv"):
        for canonical in CHANNELS:
            found = {r["session_key"] for r in channels if r["source"] == source and r["canonical_channel"] == canonical and r["selected_for_model"]}
            coverage_rows.append({
                "source": source, "channel": canonical, "recording_count": len(found),
                "percent": 100.0 * len(found) / total_sessions if total_sessions else 0,
            })
    summary = {
        "generated_utc": utc(), "recording_count": total_sessions,
        "subject_count": len({r["subject_id"] for r in sessions}),
        "patient_count": len({r["patient_id"] for r in sessions}),
        "h5_found": sum(bool(r["h5_exists"]) for r in sessions),
        "edf_found": sum(bool(r["edf_exists"]) for r in sessions),
        "channels_tsv_found": sum(bool(r["channels_tsv_exists"]) for r in sessions),
        "manual_sleep_csv_found": sum(bool(r["manual_sleep_csv_exists"]) for r in sessions),
        "metadata_duration_seconds": quantiles([r.get("metadata_duration_seconds") for r in sessions]),
        "h5_duration_seconds": quantiles([r.get("h5_duration_seconds") for r in sessions]),
        "edf_duration_seconds": quantiles([r.get("edf_duration_seconds") for r in sessions]),
        "scan_seconds": elapsed, "issue_counts": dict(sorted(issue_counts.items())),
    }
    return summary, filter_rows, coverage_rows


def report_markdown(summary: dict, filters: list[dict], coverage: list[dict]) -> str:
    lines = [
        "# HSP I0002 源数据头信息扫描", "",
        "> 本报告只读取 metadata、H5/EDF 头、channels.tsv 和 BIDS JSON；未读取整夜信号样本。", "",
        "## 总览", "",
        f"- 记录：{summary['recording_count']:,}",
        f"- 被试：{summary['subject_count']:,}",
        f"- 本地 H5：{summary['h5_found']:,}",
        f"- 网络 EDF：{summary['edf_found']:,}",
        f"- channels.tsv：{summary['channels_tsv_found']:,}",
        f"- 人工睡眠分期 CSV：{summary['manual_sleep_csv_found']:,}",
        f"- 扫描耗时：{summary['scan_seconds']:.1f} 秒", "",
        "## 记录时长（秒）", "",
        "| 来源 | N | min | p05 | median | p95 | max |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("metadata", "metadata_duration_seconds"), ("H5", "h5_duration_seconds"), ("EDF", "edf_duration_seconds")):
        q = summary[key]
        lines.append(f"| {label} | {q.get('count',0):,} | {q.get('min',0):.1f} | {q.get('p05',0):.1f} | {q.get('median',0):.1f} | {q.get('p95',0):.1f} | {q.get('max',0):.1f} |")
    lines += ["", "## 15路通道覆盖率", "", "| 来源 | 通道 | 夜数 | 占比 |", "|---|---|---:|---:|"]
    for row in coverage:
        lines.append(f"| {row['source']} | {row['channel']} | {row['recording_count']:,} | {row['percent']:.2f}% |")
    lines += ["", "## Filter profile（前120项）", "", "| 来源 | 通道 | fs | HP | LP | Notch | 夜数 | 总夜占比 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    selected_filters = [r for r in filters if r["selected_for_model"]]
    for row in selected_filters[:120]:
        def shown(value):
            return "" if value is None else value
        lines.append(f"| {row['source']} | {row['channel']} | {shown(row['fs_hz'])} | {shown(row['hp_hz'])} | {shown(row['lp_hz'])} | {shown(row['notch_hz'])} | {row['recording_count']:,} | {row['percent_of_all_recordings']:.2f}% |")
    lines += ["", "完整分布见 `tables/filter_distribution.parquet`。", "", "## 问题计数", "", "| 问题 | 数量 |", "|---|---:|"]
    for code, count in summary["issue_counts"].items():
        lines.append(f"| {code} | {count:,} |")
    return "\n".join(lines) + "\n"


def load_metadata(path: Path, limit: int) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: (text(r.get("BIDSFolder")), text(r.get("SessionID"))))
    return rows[:limit] if limit else rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path(r"I:\HSP\I0002"))
    parser.add_argument("--edf-root", type=Path, default=Path(r"\\172.16.6.5\sleep\HSP\I0002"))
    parser.add_argument("--metadata", type=Path, default=Path(r"I:\HSP\psg-metadata\I0002_psg_metadata_2025-09-08.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processes", "--workers", dest="processes", type=int, default=4,
                        help="worker process count (--workers is retained as an alias)")
    parser.add_argument("--threads-per-process", type=int, default=1,
                        help="session-scanning threads inside each worker process")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.processes < 1 or args.threads_per_process < 1 or args.limit < 0:
        parser.error("processes and threads-per-process must be positive; limit must be non-negative")
    for path in (args.local_root, args.edf_root):
        if not path.is_dir():
            parser.error(f"source root not found: {path}")
    if not args.metadata.is_file():
        parser.error(f"metadata not found: {args.metadata}")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = load_metadata(args.metadata, args.limit)
    tasks = [
        {"index": i, "row": row, "local_root": str(args.local_root), "edf_root": str(args.edf_root)}
        for i, row in enumerate(rows)
    ]
    batch_size = args.threads_per_process
    batches = [
        (tasks[offset:offset + batch_size], args.threads_per_process)
        for offset in range(0, len(tasks), batch_size)
    ]
    started = time.perf_counter()
    sessions, channels, issues = [], [], []
    progress_path = args.output / "progress.json"
    log_path = args.output / "scan.jsonl"
    concurrency = {"processes": args.processes, "threads_per_process": args.threads_per_process,
                   "max_concurrent_sessions": args.processes * args.threads_per_process}
    atomic_json(progress_path, {"status": "running", "started_utc": utc(), "total": len(tasks),
                                "completed": 0, **concurrency})
    completed = 0
    with log_path.open("w", encoding="utf-8", buffering=1) as log, ProcessPoolExecutor(max_workers=args.processes) as executor:
        for batch_results in executor.map(scan_batch, batches, chunksize=1):
            for result in batch_results:
                completed += 1
                sessions.append(result["session"])
                channels.extend(result["channels"])
                issues.extend(result["issues"])
                log.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                if completed == 1 or completed % 25 == 0 or completed == len(tasks):
                    elapsed = time.perf_counter() - started
                    rate = completed / elapsed if elapsed else 0
                    state = {"status": "running", "total": len(tasks), "completed": completed,
                             "elapsed_seconds": elapsed, "records_per_second": rate,
                             "eta_seconds": (len(tasks) - completed) / rate if rate else None,
                             "current_session": result["session"]["session_key"], "updated_utc": utc(),
                             **concurrency}
                    atomic_json(progress_path, state)
                    print(json.dumps(state, ensure_ascii=False), flush=True)
    elapsed = time.perf_counter() - started
    summary, filters, coverage = aggregate(sessions, channels, issues, elapsed)
    summary["concurrency"] = concurrency
    table_root = args.output / "tables"
    write_parquet(table_root / "sessions.parquet", sessions)
    write_parquet(table_root / "channels.parquet", channels)
    write_parquet(table_root / "issues.parquet", issues)
    write_parquet(table_root / "filter_distribution.parquet", filters)
    write_parquet(table_root / "channel_coverage.parquet", coverage)
    atomic_json(args.output / "summary.json", summary)
    from generate_hsp_source_report import build_report
    (args.output / "report.md").write_text(build_report(args.output), encoding="utf-8")
    atomic_json(progress_path, {"status": "completed", **summary})
    print(json.dumps({"status": "completed", **summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
