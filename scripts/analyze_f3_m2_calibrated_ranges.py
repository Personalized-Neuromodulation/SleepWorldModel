"""Persist a reproducible sample audit of calibrated H5 F3-M2 ranges."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


QUANTILES = (0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0)


def quantile_code(histogram: np.ndarray, quantile: float) -> int:
    count = int(histogram.sum())
    target = int(np.floor(quantile * (count - 1))) + 1
    return int(np.searchsorted(np.cumsum(histogram), target)) - 32768


def physical_value(code: int, row: dict) -> float:
    return float(
        row["phys_min"]
        + (code - row["dig_min"])
        * (row["phys_max"] - row["phys_min"])
        / (row["dig_max"] - row["dig_min"])
    )


def atomic_parquet(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(output)


def decode_edf_text(value: bytes) -> str:
    return value.decode("latin-1", errors="replace").strip()


def read_edf_header(path: Path) -> tuple[int, int, float, list[dict]]:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        header_bytes = int(decode_edf_text(fixed[184:192]))
        record_count = int(decode_edf_text(fixed[236:244]))
        record_seconds = float(decode_edf_text(fixed[244:252]))
        signal_count = int(decode_edf_text(fixed[252:256]))
        signal_header = handle.read(header_bytes - 256)
    fields = (
        ("label", 16), ("transducer", 80), ("unit", 8),
        ("phys_min", 8), ("phys_max", 8), ("dig_min", 8), ("dig_max", 8),
        ("prefilter", 80), ("samples_per_record", 8), ("reserved", 32),
    )
    signals = [{} for _ in range(signal_count)]
    position = 0
    for field, width in fields:
        for index in range(signal_count):
            value = decode_edf_text(signal_header[position : position + width])
            position += width
            if field in {"phys_min", "phys_max", "dig_min", "dig_max"}:
                value = float(value)
            elif field == "samples_per_record":
                value = int(value)
            signals[index][field] = value
    return header_bytes, record_count, record_seconds, signals


def compare_representative_with_edf(row: dict, session: dict) -> dict:
    edf_path = Path(session["edf_path"])
    header_bytes, record_count, record_seconds, signals = read_edf_header(edf_path)
    signal_index = next(
        index for index, signal in enumerate(signals)
        if signal["label"].casefold() == "f3-m2"
    )
    edf_signal = signals[signal_index]
    offsets = np.cumsum(
        [0] + [signal["samples_per_record"] for signal in signals[:-1]]
    )
    samples_per_record = sum(signal["samples_per_record"] for signal in signals)
    edf_data = np.memmap(
        edf_path,
        dtype="<i2",
        mode="r",
        offset=header_bytes,
        shape=(record_count, samples_per_record),
    )
    starts = [
        max(0, min(record_count - 61, int(record_count * fraction)))
        for fraction in (0.1, 0.3, 0.5, 0.7, 0.9)
    ]
    h5_fs = float(row["source_fs_hz"])
    with h5py.File(row["h5_path"], "r") as handle:
        dataset = handle["signals"]["f3-m2"]
        h5_values = np.concatenate([
            np.asarray(
                dataset[
                    int(start * record_seconds * h5_fs) :
                    int((start + 60) * record_seconds * h5_fs)
                ],
                dtype=np.int16,
            )
            for start in starts
        ])
    edf_values = np.concatenate([
        np.asarray(
            edf_data[
                start : start + 60,
                offsets[signal_index] :
                offsets[signal_index] + edf_signal["samples_per_record"],
            ]
        ).reshape(-1)
        for start in starts
    ])
    differences = h5_values.astype(np.int32) - edf_values.astype(np.int32)
    edf_fs = edf_signal["samples_per_record"] / record_seconds
    metadata_equal = (
        h5_fs == edf_fs
        and row["unit"] == edf_signal["unit"]
        and row["dig_min"] == edf_signal["dig_min"]
        and row["dig_max"] == edf_signal["dig_max"]
        and row["phys_min"] == edf_signal["phys_min"]
        and row["phys_max"] == edf_signal["phys_max"]
    )
    physical_scale = row["lsb_physical"]
    return {
        "session_key": row["session_key"],
        "phys_min": row["phys_min"],
        "phys_max": row["phys_max"],
        "unit": row["unit"],
        "representative_rank": row["representative_rank"],
        "h5_fs_hz": h5_fs,
        "edf_fs_hz": edf_fs,
        "h5_edf_metadata_equal": metadata_equal,
        "comparison_windows": len(starts),
        "seconds_per_window": 60,
        "comparison_sample_count": len(h5_values),
        "digital_equal_fraction": float(np.mean(differences == 0)),
        "max_abs_digital_code_difference": int(np.max(np.abs(differences))),
        "max_abs_physical_difference": float(np.max(np.abs(differences))) * physical_scale,
        "edf_path": str(edf_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--sample-per-profile", type=int, default=100)
    args = parser.parse_args()

    table_root = args.scan_root / "tables"
    channel_dataset = ds.dataset(table_root / "channels.parquet", format="parquet")
    channels = channel_dataset.to_table(
        columns=[
            "session_key", "source", "canonical_channel", "selected_for_model",
            "raw_channel", "unit", "fs_hz", "dig_min", "dig_max", "phys_min",
            "phys_max",
        ],
        filter=(ds.field("source") == "h5")
        & (ds.field("canonical_channel") == "f3-m2")
        & (ds.field("selected_for_model") == True),
    ).to_pylist()
    sessions = {
        row["session_key"]: row
        for row in pq.read_table(
            table_root / "sessions.parquet",
            columns=["session_key", "h5_path", "edf_path"],
        ).to_pylist()
    }

    profiles: dict[tuple, list[dict]] = {}
    for row in channels:
        profile = (
            row["dig_min"], row["dig_max"], row["phys_min"],
            row["phys_max"], row["unit"],
        )
        profiles.setdefault(profile, []).append(row)

    detail_rows: list[dict] = []
    started = time.perf_counter()
    completed = 0
    selected_total = sum(min(len(rows), args.sample_per_profile) for rows in profiles.values())
    for profile, rows in sorted(profiles.items(), key=lambda item: item[0][2]):
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"f3-m2-calibrated-range-v1:{row['session_key']}".encode("utf-8")
            ).digest(),
        )[: args.sample_per_profile]
        for sample_rank, row in enumerate(ranked, 1):
            path = Path(sessions[row["session_key"]]["h5_path"])
            with h5py.File(path, "r") as handle:
                dataset = handle["signals"][row["raw_channel"]]
                histogram = np.zeros(65536, dtype=np.int64)
                for start in range(0, dataset.shape[0], 1_000_000):
                    values = np.asarray(dataset[start : start + 1_000_000], dtype=np.int32)
                    histogram += np.bincount(values + 32768, minlength=65536)

            codes = {quantile: quantile_code(histogram, quantile) for quantile in QUANTILES}
            values = {quantile: physical_value(code, row) for quantile, code in codes.items()}
            total_samples = int(histogram.sum())
            detail_rows.append({
                "session_key": row["session_key"],
                "h5_path": str(path),
                "sample_rank": sample_rank,
                "profile_total_nights": len(rows),
                "source_fs_hz": row["fs_hz"],
                "dig_min": row["dig_min"],
                "dig_max": row["dig_max"],
                "phys_min": row["phys_min"],
                "phys_max": row["phys_max"],
                "unit": row["unit"],
                "lsb_physical": (row["phys_max"] - row["phys_min"])
                / (row["dig_max"] - row["dig_min"]),
                "observed_min": values[0.0],
                "p001": values[0.001],
                "p01": values[0.01],
                "median": values[0.5],
                "p99": values[0.99],
                "p999": values[0.999],
                "observed_max": values[1.0],
                "span_p001_p999": values[0.999] - values[0.001],
                "rail_fraction": float(
                    (histogram[:2].sum() + histogram[-2:].sum()) / total_samples
                ),
                "unique_digital_codes": int(np.count_nonzero(histogram)),
                "sample_count": total_samples,
                "selection_method": "SHA256(f3-m2-calibrated-range-v1:session_key)",
            })
            completed += 1
            if completed % 25 == 0 or completed == selected_total:
                print(
                    f"progress={completed}/{selected_total} "
                    f"elapsed_seconds={time.perf_counter() - started:.1f}",
                    flush=True,
                )

    summary_rows: list[dict] = []
    for profile in sorted(profiles, key=lambda item: item[2]):
        rows = [
            row for row in detail_rows
            if (
                row["dig_min"], row["dig_max"], row["phys_min"],
                row["phys_max"], row["unit"],
            ) == profile
        ]
        spans = np.asarray([row["span_p001_p999"] for row in rows], dtype=np.float64)
        median_span = float(np.median(spans))
        ordered = sorted(
            rows,
            key=lambda row: (abs(row["span_p001_p999"] - median_span), row["session_key"]),
        )
        representative_rank = {row["session_key"]: rank for rank, row in enumerate(ordered, 1)}
        for row in rows:
            row["representative_rank"] = representative_rank[row["session_key"]]

        def median(field: str) -> float:
            return float(np.median([row[field] for row in rows]))

        summary_rows.append({
            "dig_min": profile[0],
            "dig_max": profile[1],
            "phys_min": profile[2],
            "phys_max": profile[3],
            "unit": profile[4],
            "profile_total_nights": len(profiles[profile]),
            "sample_nights": len(rows),
            "lsb_physical": median("lsb_physical"),
            "night_median_p001": median("p001"),
            "night_median_p01": median("p01"),
            "night_median_p50": median("median"),
            "night_median_p99": median("p99"),
            "night_median_p999": median("p999"),
            "night_median_span_p001_p999": median_span,
            "night_p05_span_p001_p999": float(np.quantile(spans, 0.05)),
            "night_p95_span_p001_p999": float(np.quantile(spans, 0.95)),
            "median_rail_fraction": median("rail_fraction"),
            "max_rail_fraction": max(row["rail_fraction"] for row in rows),
            "median_unique_digital_codes": median("unique_digital_codes"),
            "selection_method": "SHA256(f3-m2-calibrated-range-v1:session_key)",
        })

    edf_comparison_rows = [
        compare_representative_with_edf(row, sessions[row["session_key"]])
        for row in detail_rows
        if row["representative_rank"] <= 5
    ]
    atomic_parquet(detail_rows, table_root / "f3_m2_calibrated_range_sample.parquet")
    atomic_parquet(summary_rows, table_root / "f3_m2_calibrated_range_summary.parquet")
    atomic_parquet(
        edf_comparison_rows,
        table_root / "f3_m2_h5_edf_representative_comparison.parquet",
    )
    print(f"detail_rows={len(detail_rows)}")
    print(f"summary_rows={len(summary_rows)}")
    print(f"edf_comparison_rows={len(edf_comparison_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
