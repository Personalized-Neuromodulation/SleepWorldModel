"""Generate a statistics-only Markdown report from an HSP source scan."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq


CHANNELS = (
    "f3-m2", "f4-m1", "c3-m2", "c4-m1", "o1-m2", "o2-m1",
    "e1", "e2", "ecg", "chin1-chin2", "lat", "rat", "airflow",
    "snore", "spo2",
)
SOURCES = ("h5", "edf")


def pct(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.2f}%" if denominator else "—"


def shown(value, digits: int = 3) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "unknown"
        if value.is_integer():
            return str(int(value))
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    value = str(value)
    return value.replace("|", "\\|").replace("\n", " ") if value else "(blank)"


def add_table(lines: list[str], headers: list[str], rows: list[list]) -> None:
    lines.extend([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---:" if index else "---" for index in range(len(headers))) + "|",
    ])
    lines.extend("| " + " | ".join(shown(value) for value in row) + " |" for row in rows)
    lines.append("")


def quantile_row(values: list[float], divisor: float = 1.0) -> list:
    array = np.asarray([value / divisor for value in values if value is not None], dtype=np.float64)
    points = np.quantile(array, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    return [
        len(array), *points[:5].tolist(), float(array.mean()), *points[5:].tolist()
    ]


def unique_selected_rows(channel_rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    selected = {}
    for row in channel_rows:
        selected[(row["session_key"], row["source"], row["canonical_channel"])] = row
    return selected


def build_report(scan_root: Path) -> str:
    table_root = scan_root / "tables"
    summary_path = scan_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    sessions = pq.read_table(table_root / "sessions.parquet").to_pylist()
    issues = pq.read_table(table_root / "issues.parquet").to_pylist()
    channel_dataset = ds.dataset(table_root / "channels.parquet", format="parquet")
    all_h5_rows = channel_dataset.to_table(
        columns=[
            "session_key", "source", "canonical_channel", "raw_channel",
            "selected_for_model", "unit", "fs_hz", "dig_min", "dig_max",
            "phys_min", "phys_max", "hp_hz", "lp_hz", "notch_hz",
            "prefilter_raw",
        ],
        filter=ds.field("source") == "h5",
    ).to_pylist()
    selected_filter = (
        (ds.field("selected_for_model") == True)
        & ds.field("source").isin(["h5", "edf", "tsv"])
    )
    selected_columns = [
        "session_key", "source", "canonical_channel", "raw_channel", "unit",
        "dtype", "sample_count", "fs_hz", "duration_seconds", "dig_min",
        "dig_max", "phys_min", "phys_max", "prefilter_raw", "hp_hz",
        "lp_hz", "notch_hz", "filter_parse_ok", "value_domain",
    ]
    channel_rows = channel_dataset.to_table(
        columns=selected_columns, filter=selected_filter
    ).to_pylist()
    selected = unique_selected_rows(channel_rows)
    total = len(sessions)
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = ["# HSP I0002 源数据统计报告", ""]

    lines.extend(["## 1. 报告与扫描规模", ""])
    patients = Counter(row["patient_id"] for row in sessions)
    subjects = {row["subject_id"] for row in sessions}
    task_seconds = sum(float(row.get("scan_seconds") or 0) for row in sessions)
    wall_seconds = float(summary.get("scan_seconds") or 0)
    concurrency = summary.get("concurrency") or {}
    add_table(lines, ["统计项", "数值"], [
        ["报告生成时间", now],
        ["记录数", total],
        ["patient数", len(patients)],
        ["subject数", len(subjects)],
        ["多session patient数", sum(count > 1 for count in patients.values())],
        ["最大session数/patient", max(patients.values())],
        ["全量扫描墙钟秒", wall_seconds],
        ["平均记录/秒", total / wall_seconds if wall_seconds else None],
        ["session任务累计秒", task_seconds],
        ["进程数", concurrency.get("processes")],
        ["每进程线程数", concurrency.get("threads_per_process")],
        ["最大并发session数", concurrency.get("max_concurrent_sessions")],
    ])

    session_count_distribution = Counter(patients.values())
    add_table(lines, ["session数/patient", "patient数", "patient占比"], [
        [count, patient_count, pct(patient_count, len(patients))]
        for count, patient_count in sorted(session_count_distribution.items())
    ])

    years = Counter()
    for row in sessions:
        match = re.search(r"(?:19|20)\d{2}", str(row.get("metadata_start") or ""))
        years[match.group(0) if match else "unknown"] += 1
    add_table(lines, ["记录年份", "夜数", "占比"], [
        [year, count, pct(count, total)] for year, count in sorted(years.items())
    ])

    lines.extend(["## 2. 文件存在性与大小", ""])
    presence_fields = [
        ("本地H5", "h5_exists"), ("网络EDF", "edf_exists"),
        ("channels.tsv", "channels_tsv_exists"), ("BIDS EEG JSON", "bids_json_exists"),
        ("人工睡眠分期CSV", "manual_sleep_csv_exists"),
    ]
    add_table(lines, ["文件", "存在夜数", "缺失夜数", "存在占比"], [
        [label, sum(bool(row.get(field)) for row in sessions),
         total - sum(bool(row.get(field)) for row in sessions),
         pct(sum(bool(row.get(field)) for row in sessions), total)]
        for label, field in presence_fields
    ])

    file_size_rows = []
    for label, field in (("H5", "h5_bytes"), ("EDF", "edf_bytes")):
        values = [float(row[field]) for row in sessions if row.get(field) is not None]
        q = quantile_row(values, 1024 ** 2)
        file_size_rows.append([
            label, len(values), sum(values) / 1024 ** 3, sum(values) / 1024 ** 4,
            q[1], q[3], q[5], q[8], q[9], q[10],
        ])
    add_table(lines, ["来源", "N", "总GiB", "总TiB", "min MiB", "P5 MiB", "median MiB", "P95 MiB", "P99 MiB", "max MiB"], file_size_rows)

    size_ratios = [
        row["h5_bytes"] / row["edf_bytes"]
        for row in sessions
        if row.get("h5_bytes") is not None and row.get("edf_bytes")
    ]
    ratio_q = quantile_row(size_ratios)
    add_table(lines, ["比值", "N", "min", "P1", "P5", "P25", "median", "mean", "P75", "P95", "P99", "max"], [
        ["H5 bytes / EDF bytes", *ratio_q]
    ])

    lines.extend(["## 3. 整夜记录时长", ""])
    duration_rows = []
    for label, field in (
        ("metadata", "metadata_duration_seconds"),
        ("H5", "h5_duration_seconds"),
        ("EDF", "edf_duration_seconds"),
    ):
        q = quantile_row([row.get(field) for row in sessions], 3600)
        duration_rows.append([label, *q])
    add_table(lines, ["来源", "N", "min h", "P1 h", "P5 h", "P25 h", "median h", "mean h", "P75 h", "P95 h", "P99 h", "max h"], duration_rows)

    duration_bins = (
        ("<1h", lambda value: value < 3600),
        ("1–<4h", lambda value: 3600 <= value < 14400),
        ("4–<5h", lambda value: 14400 <= value < 18000),
        ("5–<8h", lambda value: 18000 <= value < 28800),
        ("8–12h", lambda value: 28800 <= value <= 43200),
        (">12h", lambda value: value > 43200),
    )
    bin_rows = []
    for label, field in (("H5", "h5_duration_seconds"), ("EDF", "edf_duration_seconds")):
        values = [float(row[field]) for row in sessions if row.get(field) is not None]
        for bin_label, predicate in duration_bins:
            count = sum(predicate(value) for value in values)
            bin_rows.append([label, bin_label, count, pct(count, len(values))])
    add_table(lines, ["来源", "时长区间", "夜数", "占比"], bin_rows)

    threshold_rows = []
    for label, field in (("H5", "h5_duration_seconds"), ("EDF", "edf_duration_seconds")):
        values = [float(row[field]) for row in sessions if row.get(field) is not None]
        for threshold_label, threshold in (("<30s", 30), ("<5min", 300), ("<30min", 1800), ("<1h", 3600), (">4h", 14400), (">5h", 18000)):
            count = sum(value < threshold for value in values) if threshold_label.startswith("<") else sum(value > threshold for value in values)
            threshold_rows.append([label, threshold_label, count, pct(count, len(values))])
    add_table(lines, ["来源", "阈值", "夜数", "占比"], threshold_rows)

    paired_durations = []
    duration_outliers = []
    for row in sessions:
        h5_duration, edf_duration = row.get("h5_duration_seconds"), row.get("edf_duration_seconds")
        if h5_duration is None or edf_duration is None:
            continue
        difference = float(edf_duration - h5_duration)
        paired_durations.append(difference)
        if abs(difference) > 0.000001:
            duration_outliers.append([row["session_key"], h5_duration, edf_duration, difference])
    duration_array = np.asarray(paired_durations)
    add_table(lines, ["H5/EDF时长比较", "夜数", "占比"], [
        ["完全一致", int(np.sum(np.abs(duration_array) <= 0.000001)), pct(int(np.sum(np.abs(duration_array) <= 0.000001)), len(duration_array))],
        ["不一致", int(np.sum(np.abs(duration_array) > 0.000001)), pct(int(np.sum(np.abs(duration_array) > 0.000001)), len(duration_array))],
        ["绝对差>1s", int(np.sum(np.abs(duration_array) > 1)), pct(int(np.sum(np.abs(duration_array) > 1)), len(duration_array))],
        ["绝对差>30s", int(np.sum(np.abs(duration_array) > 30)), pct(int(np.sum(np.abs(duration_array) > 30)), len(duration_array))],
        ["EDF更长", int(np.sum(duration_array > 0.000001)), pct(int(np.sum(duration_array > 0.000001)), len(duration_array))],
        ["H5更长", int(np.sum(duration_array < -0.000001)), pct(int(np.sum(duration_array < -0.000001)), len(duration_array))],
    ])
    add_table(lines, ["session", "H5 s", "EDF s", "EDF−H5 s"], sorted(duration_outliers, key=lambda row: abs(row[3]), reverse=True))

    lines.extend(["## 4. 目标通道覆盖率与组合完整度", ""])
    coverage_rows = []
    presence = defaultdict(set)
    for session_key, source, channel in selected:
        presence[(source, channel)].add(session_key)
    for channel in CHANNELS:
        h5_sessions = presence[("h5", channel)]
        edf_sessions = presence[("edf", channel)]
        tsv_sessions = presence[("tsv", channel)]
        coverage_rows.append([
            channel, len(h5_sessions), pct(len(h5_sessions), total),
            len(edf_sessions), pct(len(edf_sessions), total),
            len(tsv_sessions), pct(len(tsv_sessions), total),
            len(h5_sessions - edf_sessions), len(edf_sessions - h5_sessions),
        ])
    add_table(lines, ["通道", "H5夜数", "H5占比", "EDF夜数", "EDF占比", "TSV夜数", "TSV占比", "仅H5", "仅EDF"], coverage_rows)

    session_keys = [row["session_key"] for row in sessions]
    modality_sets = {
        "6 EEG全部": set(CHANNELS[:6]),
        "2 EOG全部": {"e1", "e2"},
        "ECG": {"ecg"},
        "3 EMG全部": {"chin1-chin2", "lat", "rat"},
        "Airflow+Snore+SpO2全部": {"airflow", "snore", "spo2"},
        "15通道全部": set(CHANNELS),
    }
    completeness_rows = []
    for source in ("h5", "edf", "tsv"):
        available_by_session = defaultdict(set)
        for (session_key, row_source, channel) in selected:
            if row_source == source:
                available_by_session[session_key].add(channel)
        for label, required in modality_sets.items():
            count = sum(required <= available_by_session[session] for session in session_keys)
            completeness_rows.append([source, label, count, pct(count, total)])
    add_table(lines, ["来源", "完整组合", "夜数", "占比"], completeness_rows)

    lines.extend(["## 5. 逐通道H5/EDF记录时长一致性", ""])
    channel_duration_rows = []
    for channel in CHANNELS:
        h5_sessions = presence[("h5", channel)]
        edf_sessions = presence[("edf", channel)]
        common = h5_sessions & edf_sessions
        differences = []
        for session in common:
            left = selected[(session, "h5", channel)].get("duration_seconds")
            right = selected[(session, "edf", channel)].get("duration_seconds")
            if left is not None and right is not None:
                differences.append(float(right - left))
        array = np.asarray(differences)
        channel_duration_rows.append([
            channel, len(common), int(np.sum(np.abs(array) <= 0.000001)),
            int(np.sum(np.abs(array) > 1)), int(np.sum(np.abs(array) > 30)),
            len(h5_sessions - edf_sessions), len(edf_sessions - h5_sessions),
            float(np.max(np.abs(array))) if len(array) else None,
        ])
    add_table(lines, ["通道", "配对夜数", "完全一致", "差>1s", "差>30s", "仅H5", "仅EDF", "最大绝对差 s"], channel_duration_rows)

    lines.extend(["## 6. 采样率分布", ""])
    sampling_rows = []
    for channel in CHANNELS:
        for source in SOURCES:
            rows = [row for (session, row_source, row_channel), row in selected.items() if row_source == source and row_channel == channel]
            counts = Counter(row.get("fs_hz") for row in rows)
            for fs_hz, count in counts.most_common():
                sampling_rows.append([source, channel, fs_hz, count, pct(count, len(rows)), pct(count, total)])
    add_table(lines, ["来源", "通道", "采样率 Hz", "夜数", "通道内占比", "全体占比"], sampling_rows)

    lines.extend(["## 7. 采样率与滤波联合profile（每通道前3）", ""])
    joint_filter_rows = []
    filter_summary_rows = []
    for channel in CHANNELS:
        for source in SOURCES:
            rows = [row for (session, row_source, row_channel), row in selected.items() if row_source == source and row_channel == channel]
            profiles = Counter((row.get("fs_hz"), row.get("hp_hz"), row.get("lp_hz"), row.get("notch_hz")) for row in rows)
            parsed = sum(bool(row.get("filter_parse_ok")) for row in rows)
            filter_summary_rows.append([source, channel, len(rows), parsed, pct(parsed, len(rows)), len(profiles)])
            for rank, (profile, count) in enumerate(profiles.most_common(3), 1):
                joint_filter_rows.append([source, channel, rank, *profile, count, pct(count, len(rows)), pct(count, total)])
    add_table(lines, ["来源", "通道", "rank", "fs Hz", "HP Hz", "LP Hz", "Notch Hz", "夜数", "通道内占比", "全体占比"], joint_filter_rows)
    add_table(lines, ["来源", "通道", "存在夜数", "解析成功夜数", "解析成功率", "联合profile数"], filter_summary_rows)

    lines.extend(["## 8. 单位分布", ""])
    unit_rows = []
    for channel in CHANNELS:
        for source in SOURCES:
            rows = [row for (session, row_source, row_channel), row in selected.items() if row_source == source and row_channel == channel]
            counts = Counter(row.get("unit") or "(blank)" for row in rows)
            for unit, count in counts.most_common():
                unit_rows.append([source, channel, unit, count, pct(count, len(rows)), pct(count, total)])
    add_table(lines, ["来源", "通道", "单位", "夜数", "通道内占比", "全体占比"], unit_rows)

    lines.extend(["## 9. 原始通道名映射分布（每通道前5）", ""])
    raw_name_rows = []
    for channel in CHANNELS:
        for source in SOURCES:
            rows = [row for (session, row_source, row_channel), row in selected.items() if row_source == source and row_channel == channel]
            counts = Counter(row.get("raw_channel") or "(blank)" for row in rows)
            for rank, (raw_name, count) in enumerate(counts.most_common(5), 1):
                raw_name_rows.append([source, channel, rank, raw_name, count, pct(count, len(rows))])
    add_table(lines, ["来源", "目标通道", "rank", "原始通道名", "夜数", "通道内占比"], raw_name_rows)

    lines.extend(["## 10. 数值存储与校准profile", ""])
    storage_rows = []
    calibration_rows = []
    for channel in CHANNELS:
        for source in SOURCES:
            rows = [row for (session, row_source, row_channel), row in selected.items() if row_source == source and row_channel == channel]
            storage = Counter((row.get("dtype") or "(blank)", row.get("value_domain") or "(blank)") for row in rows)
            for rank, (profile, count) in enumerate(storage.most_common(3), 1):
                storage_rows.append([source, channel, rank, profile[0], profile[1], count, pct(count, len(rows))])
            calibration = Counter((row.get("dig_min"), row.get("dig_max"), row.get("phys_min"), row.get("phys_max"), row.get("unit") or "(blank)") for row in rows)
            for rank, (profile, count) in enumerate(calibration.most_common(3), 1):
                calibration_rows.append([source, channel, rank, *profile, count, pct(count, len(rows))])
    add_table(lines, ["来源", "通道", "rank", "dtype", "value domain", "夜数", "通道内占比"], storage_rows)
    add_table(lines, ["来源", "通道", "rank", "dig min", "dig max", "phys min", "phys max", "单位", "夜数", "通道内占比"], calibration_rows)

    lines.extend(["## 11. H5标定属性完整性", ""])
    calibration_fields = ("dig_min", "dig_max", "phys_min", "phys_max")

    def calibration_presence(rows: list[dict]) -> list:
        counts = [sum(row.get(field) is not None for row in rows) for field in calibration_fields]
        all_four = sum(all(row.get(field) is not None for field in calibration_fields) for row in rows)
        return [
            len(rows), *counts, all_four, len(rows) - all_four,
            pct(all_four, len(rows)),
        ]

    selected_h5_rows = [
        row for row in all_h5_rows
        if row.get("selected_for_model") and row.get("canonical_channel") in CHANNELS
    ]
    add_table(
        lines,
        ["范围", "H5通道实例数", "dig_min非空", "dig_max非空", "phys_min非空",
         "phys_max非空", "四项均非空", "至少一项缺失", "四项完整率"],
        [
            ["全部H5通道数据集", *calibration_presence(all_h5_rows)],
            ["计划15路的已选H5通道", *calibration_presence(selected_h5_rows)],
        ],
    )

    calibration_channel_rows = []
    for channel in CHANNELS:
        rows = [row for row in selected_h5_rows if row.get("canonical_channel") == channel]
        calibration_channel_rows.append([channel, *calibration_presence(rows)])
    add_table(
        lines,
        ["通道", "H5通道实例数", "dig_min非空", "dig_max非空", "phys_min非空",
         "phys_max非空", "四项均非空", "至少一项缺失", "四项完整率"],
        calibration_channel_rows,
    )

    missing_combinations = Counter(
        tuple(field for field in calibration_fields if row.get(field) is None)
        for row in all_h5_rows
        if any(row.get(field) is None for field in calibration_fields)
    )
    add_table(lines, ["全部H5缺失字段组合", "通道实例数"], [
        ["+".join(fields), count] for fields, count in missing_combinations.most_common()
    ] or [["无", 0]])

    lines.extend(["## 12. H5/EDF元数据差异（按通道）", ""])
    mismatch_counts = Counter()
    mismatch_sessions = defaultdict(set)
    mismatch_channel_counts = Counter()
    mismatch_channel_sessions = defaultdict(set)
    for row in issues:
        code = row["code"]
        detail = row.get("detail") or ""
        if code == "H5_EDF_METADATA_MISMATCH":
            parts = detail.split(":", 2)
            if len(parts) >= 2:
                key = (parts[0], parts[1])
                mismatch_counts[key] += 1
                mismatch_sessions[key].add(row["session_key"])
                mismatch_channel_counts[parts[0]] += 1
                mismatch_channel_sessions[parts[0]].add(row["session_key"])
        elif code == "H5_EDF_UNIT_MISMATCH":
            channel = detail.split(":", 1)[0]
            key = (channel, "unit")
            mismatch_counts[key] += 1
            mismatch_sessions[key].add(row["session_key"])
    add_table(lines, ["通道", "H5/EDF配对夜数", "metadata mismatch finding数", "影响夜数", "配对夜占比"], [
        [
            channel,
            len(presence[("h5", channel)] & presence[("edf", channel)]),
            mismatch_channel_counts[channel],
            len(mismatch_channel_sessions[channel]),
            pct(
                len(mismatch_channel_sessions[channel]),
                len(presence[("h5", channel)] & presence[("edf", channel)]),
            ),
        ]
        for channel in CHANNELS
    ])
    add_table(lines, ["通道", "字段", "finding数", "影响夜数", "全体夜占比"], [
        [channel, field, count, len(mismatch_sessions[(channel, field)]), pct(len(mismatch_sessions[(channel, field)]), total)]
        for (channel, field), count in sorted(mismatch_counts.items())
    ])

    lines.extend(["## 13. H5/channels.tsv元数据差异（按通道）", ""])
    tsv_mismatch_counts = Counter()
    tsv_mismatch_sessions = defaultdict(set)
    tsv_field_counts = Counter()
    tsv_field_sessions = defaultdict(set)
    for row in issues:
        if row["code"] != "H5_TSV_METADATA_MISMATCH":
            continue
        parts = (row.get("detail") or "").split(":", 2)
        if len(parts) < 2:
            continue
        channel, field = parts[:2]
        tsv_mismatch_counts[channel] += 1
        tsv_mismatch_sessions[channel].add(row["session_key"])
        tsv_field_counts[(channel, field)] += 1
        tsv_field_sessions[(channel, field)].add(row["session_key"])
    add_table(lines, ["通道", "H5/TSV配对夜数", "metadata mismatch finding数", "影响夜数", "配对夜占比"], [
        [
            channel,
            len(presence[("h5", channel)] & presence[("tsv", channel)]),
            tsv_mismatch_counts[channel],
            len(tsv_mismatch_sessions[channel]),
            pct(
                len(tsv_mismatch_sessions[channel]),
                len(presence[("h5", channel)] & presence[("tsv", channel)]),
            ),
        ]
        for channel in CHANNELS
    ])
    add_table(lines, ["通道", "字段", "finding数", "影响夜数", "全体夜占比"], [
        [channel, field, count, len(tsv_field_sessions[(channel, field)]), pct(len(tsv_field_sessions[(channel, field)]), total)]
        for (channel, field), count in sorted(tsv_field_counts.items())
    ])

    lines.extend(["## 14. EDF计划通道缺失（按通道）", ""])
    edf_missing_counts = Counter()
    edf_missing_sessions = defaultdict(set)
    for row in issues:
        if row["code"] != "EDF_SELECTED_CHANNEL_MISSING":
            continue
        channel = row.get("detail") or "unknown"
        edf_missing_counts[channel] += 1
        edf_missing_sessions[channel].add(row["session_key"])
    add_table(lines, ["通道", "缺失夜数", "全体夜占比", "EDF存在夜数", "EDF存在占比"], [
        [
            channel,
            edf_missing_counts[channel],
            pct(edf_missing_counts[channel], total),
            len(presence[("edf", channel)]),
            pct(len(presence[("edf", channel)]), total),
        ]
        for channel in sorted(CHANNELS, key=lambda name: (-edf_missing_counts[name], CHANNELS.index(name)))
    ])

    lines.extend(["## 15. BIDS与EDF头字段分布", ""])
    header_fields = [
        ("manufacturer", "manufacturer"),
        ("power_line_frequency", "power_line_frequency"),
        ("bids_sampling_frequency", "bids_sampling_frequency"),
        ("recording_type", "recording_type"),
        ("eeg_reference", "eeg_reference"),
        ("edf_version", "edf_version"),
        ("edf_reserved", "edf_reserved"),
        ("edf_record_seconds", "edf_record_seconds"),
        ("edf_signal_count", "edf_signal_count"),
        ("h5_channel_count", "h5_channel_count"),
        ("tsv_channel_count", "tsv_channel_count"),
    ]
    header_rows = []
    for label, field in header_fields:
        counts = Counter(shown(row.get(field)) for row in sessions)
        top = counts.most_common(20)
        for rank, (value, count) in enumerate(top, 1):
            header_rows.append([label, rank, value, count, pct(count, total)])
        other = total - sum(count for _, count in top)
        if other:
            header_rows.append([label, ">20", "other", other, pct(other, total)])
    add_table(lines, ["字段", "rank", "值", "夜数", "占比"], header_rows)

    lines.extend(["## 16. Finding统计", ""])
    issue_counts = Counter(row["code"] for row in issues)
    affected = defaultdict(set)
    for row in issues:
        affected[row["code"]].add(row["session_key"])
    add_table(lines, ["finding code", "finding数", "影响夜数", "全体夜占比"], [
        [code, count, len(affected[code]), pct(len(affected[code]), total)]
        for code, count in sorted(issue_counts.items())
    ])

    lines.extend(["## 17. H5严格Airflow（仅原始dataset名airflow）", ""])
    strict_airflow = [
        row for row in all_h5_rows
        if str(row.get("raw_channel") or "").strip().casefold() == "airflow"
    ]
    strict_sessions = {row["session_key"] for row in strict_airflow}
    add_table(lines, ["统计项", "夜数", "全体夜占比"], [
        ["H5 /signals/airflow存在", len(strict_sessions), pct(len(strict_sessions), total)],
        ["H5 /signals/airflow缺失", total - len(strict_sessions), pct(total - len(strict_sessions), total)],
        ["旧canonical Airflow中由其他名称fallback的夜数",
         len(presence[("h5", "airflow")] - strict_sessions),
         pct(len(presence[("h5", "airflow")] - strict_sessions), total)],
    ])
    strict_fs = Counter(row.get("fs_hz") for row in strict_airflow)
    add_table(lines, ["H5严格Airflow原生采样率Hz", "夜数", "严格Airflow占比", "全体夜占比"], [
        [fs, count, pct(count, len(strict_airflow)), pct(count, total)]
        for fs, count in sorted(strict_fs.items(), key=lambda item: (item[0] is None, item[0] or 0))
    ])
    strict_calibration = Counter(
        (
            row.get("dig_min"), row.get("dig_max"), row.get("phys_min"),
            row.get("phys_max"), row.get("unit") or "(blank)",
        )
        for row in strict_airflow
    )
    add_table(lines, ["dig_min", "dig_max", "phys_min", "phys_max", "unit", "夜数", "严格Airflow占比"], [
        [*profile, count, pct(count, len(strict_airflow))]
        for profile, count in strict_calibration.most_common()
    ])
    strict_filters = Counter(
        (row.get("fs_hz"), row.get("hp_hz"), row.get("lp_hz"), row.get("notch_hz"))
        for row in strict_airflow
    )
    add_table(lines, ["fs_hz", "hp_hz", "lp_hz", "notch_hz", "夜数", "严格Airflow占比"], [
        [*profile, count, pct(count, len(strict_airflow))]
        for profile, count in strict_filters.most_common()
    ])

    lines.extend(["## 18. 计划15路H5量程与单位缺失（Airflow严格同名）", ""])
    strict_model_rows = [
        row for row in all_h5_rows
        if row.get("selected_for_model")
        and row.get("canonical_channel") in CHANNELS
        and (
            row.get("canonical_channel") != "airflow"
            or str(row.get("raw_channel") or "").strip().casefold() == "airflow"
        )
    ]
    strict_by_channel = defaultdict(list)
    for row in strict_model_rows:
        strict_by_channel[row["canonical_channel"]].append(row)

    summary_rows = []
    calibration_rows = []
    for channel in CHANNELS:
        rows = strict_by_channel[channel]
        present_sessions = {row["session_key"] for row in rows}
        missing_units = sum(not str(row.get("unit") or "").strip() for row in rows)
        missing_range_attrs = sum(
            any(row.get(field) is None for field in ("dig_min", "dig_max", "phys_min", "phys_max"))
            for row in rows
        )
        units = sorted({str(row.get("unit") or "").strip() or "(blank)" for row in rows})
        profiles = Counter(
            (
                row.get("dig_min"), row.get("dig_max"), row.get("phys_min"),
                row.get("phys_max"), str(row.get("unit") or "").strip() or "(blank)",
            )
            for row in rows
        )
        summary_rows.append([
            channel, len(present_sessions), total - len(present_sessions),
            pct(len(present_sessions), total), len(profiles), ", ".join(units),
            missing_units, pct(missing_units, len(rows)), missing_range_attrs,
        ])
        for rank, (profile, count) in enumerate(profiles.most_common(), 1):
            calibration_rows.append([
                channel, rank, *profile, "是" if profile[-1] == "(blank)" else "否",
                count, pct(count, len(rows)),
            ])

    add_table(lines, [
        "通道", "存在夜数", "缺失夜数", "全体存在占比", "量程profile数",
        "单位取值", "单位缺失夜数", "单位缺失/存在", "标定四属性缺失夜数",
    ], summary_rows)
    add_table(lines, [
        "通道", "rank", "dig_min", "dig_max", "phys_min", "phys_max", "unit",
        "单位缺失", "夜数", "通道存在夜占比",
    ], calibration_rows)

    f3_summary_path = table_root / "f3_m2_calibrated_range_summary.parquet"
    f3_detail_path = table_root / "f3_m2_calibrated_range_sample.parquet"
    if f3_summary_path.is_file() and f3_detail_path.is_file():
        lines.extend(["## 19. F3-M2两种量程的标定后实际值抽样", ""])
        f3_summary = pq.read_table(f3_summary_path).to_pylist()
        f3_detail = pq.read_table(f3_detail_path).to_pylist()
        add_table(lines, [
            "phys_min", "phys_max", "unit", "profile总夜数", "确定性抽样夜数",
            "LSB", "每夜p1中位", "每夜p99中位", "每夜p0.1中位",
            "每夜p99.9中位", "p0.1-p99.9宽度中位", "宽度p05", "宽度p95",
            "rail占比中位", "最大rail占比", "唯一digital code数中位",
        ], [
            [
                row["phys_min"], row["phys_max"], row["unit"],
                row["profile_total_nights"], row["sample_nights"], row["lsb_physical"],
                row["night_median_p01"], row["night_median_p99"],
                row["night_median_p001"], row["night_median_p999"],
                row["night_median_span_p001_p999"],
                row["night_p05_span_p001_p999"], row["night_p95_span_p001_p999"],
                row["median_rail_fraction"], row["max_rail_fraction"],
                row["median_unique_digital_codes"],
            ]
            for row in sorted(f3_summary, key=lambda row: row["phys_min"])
        ])
        representatives = sorted(
            (row for row in f3_detail if row["representative_rank"] <= 5),
            key=lambda row: (row["phys_min"], row["representative_rank"]),
        )
        add_table(lines, [
            "phys_min", "phys_max", "代表rank", "session", "fs_hz", "LSB",
            "observed_min", "p0.1", "p1", "median", "p99", "p99.9",
            "observed_max", "rail占比", "唯一digital code数",
        ], [
            [
                row["phys_min"], row["phys_max"], row["representative_rank"],
                row["session_key"], row["source_fs_hz"], row["lsb_physical"],
                row["observed_min"], row["p001"], row["p01"], row["median"],
                row["p99"], row["p999"], row["observed_max"],
                row["rail_fraction"], row["unique_digital_codes"],
            ]
            for row in representatives
        ])
        comparison_path = table_root / "f3_m2_h5_edf_representative_comparison.parquet"
        if comparison_path.is_file():
            comparisons = pq.read_table(comparison_path).to_pylist()
            add_table(lines, [
                "phys_min", "phys_max", "代表rank", "session", "H5 fs", "EDF fs",
                "元数据一致", "比较窗口", "比较样本数", "digital一致率",
                "最大code差", "最大物理值差",
            ], [
                [
                    row["phys_min"], row["phys_max"], row["representative_rank"],
                    row["session_key"], row["h5_fs_hz"], row["edf_fs_hz"],
                    "是" if row["h5_edf_metadata_equal"] else "否",
                    f"{row['comparison_windows']}x{row['seconds_per_window']}s",
                    row["comparison_sample_count"], row["digital_equal_fraction"],
                    row["max_abs_digital_code_difference"],
                    row["max_abs_physical_difference"],
                ]
                for row in sorted(
                    comparisons,
                    key=lambda row: (row["phys_min"], row["representative_rank"]),
                )
            ])

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.scan_root / "report.md"
    report = build_report(args.scan_root)
    output.write_text(report, encoding="utf-8")
    print(f"wrote={output}")
    print(f"bytes={output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
