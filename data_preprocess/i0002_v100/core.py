from __future__ import annotations

from fractions import Fraction
import math
import re

import numpy as np
from scipy import signal

FS = 200
EPOCH_SECONDS = 30
EPOCH_SAMPLES = 6000
CHANNELS = (
    "f3-m2", "f4-m1", "c3-m2", "c4-m1", "o1-m2", "o2-m1",
    "e1", "e2", "ecg", "chin1-chin2", "lat", "rat", "airflow", "snore", "spo2",
)
GROUPS = {"eeg": (0, 6), "eog": (6, 8), "ecg": (8, 9),
          "emg": (9, 12), "respiratory": (12, 15)}
HARD_CODES = {0: "NO_HARD_ARTIFACT_DETECTED", 1: "NAN", 2: "FLAT_LINE",
              3: "SATURATION", 4: "HIGH_AMPLITUDE", 5: "POWER_LINE_INTERFERENCE",
              6: "HIGH_FREQUENCY_NOISE", 7: "SPO2_OUT_OF_RANGE"}
PRIORITY = (1, 3, 4, 2, 5, 6, 7)
APPLICABLE = np.asarray([63] * 9 + [29] * 3 + [3, 1, 65], np.uint8)
CHANNEL_STATUS = {"AVAILABLE": 0, "CHANNEL_MISSING": 1, "CHANNEL_UNREADABLE": 2,
                  "CALIBRATION_OR_UNIT_INVALID": 3, "PROFILE_EXCLUDED": 4,
                  "SPO2_NEAR_ZERO": 5}
TARGETS = {"eeg": (0.3, 35.0), "eog": (0.3, 35.0), "ecg": (0.3, 70.0),
           "emg": (10.0, 100.0), "airflow": (0.159, 15.0), "snore": (10.0, 100.0)}
FILTER_STATE = {"NOT_EVALUATED": 0, "SOURCE_FILTER_EXPLICIT": 1, "SOURCE_FILTER_UNKNOWN": 2}
FILTER_RELATION = {"NOT_EVALUATED": 0, "MATCH": 1, "SOURCE_WIDER": 2,
                   "SOURCE_NARROWER": 3, "MIXED": 4, "UNKNOWN": 5, "NOT_REQUIRED": 6}
EFFECTIVE_FILTER = {"NOT_EVALUATED": 0, "SOURCE_FILTER_AS_RECORDED": 1,
                    "SOURCE_FILTER_REAPPLY": 2, "FILTER_NOT_REQUIRED": 3,
                    "REAPPLY_FAILED": 4, "PROCESSING_SKIPPED_PROFILE_EXCLUDED": 5}
PROCESS_RESULT = {"NOT_EVALUATED": 0, "NOT_REQUIRED": 1, "SUCCESS": 2, "FAILED": 3,
                  "UNSUPPORTED": 4, "NOT_RUN_PROFILE_EXCLUDED": 5, "PARTIAL": 6}
COVERAGE_DETECTION = {"NOT_EVALUATED": 0, "LENGTH_ONLY": 1, "TIMESTAMP_CHECKED": 2}
PSD_CONFIG = {
    "method": "welch", "fs": 200, "window": "periodic_hann", "nperseg": 800,
    "noverlap": 400, "nfft": 800, "detrend": "constant", "return_onesided": True,
    "scaling": "density", "average": "mean", "compute_dtype": "float64",
    "frequency_spacing_hz": 0.25, "segments_per_epoch": 14,
    "denominator_hz": "[0.5,100]", "line_hz": "[49,51] union [59,61]",
    "high_frequency_hz": "(70,100]", "integration": "sum(selected_bins)*0.25",
    "zero_denominator_ratio": 0.0, "line_threshold": 0.40, "hf_threshold": 0.50,
    "comparison": "strict_greater_than", "epoch_batch_size": 64,
}
METRIC_SPECS = {
    "nonfinite_output_count": ("u2", 65535),
    "nonfinite_source_count": ("u4", 4294967295),
    "std_physical": ("f4", np.nan),
    "digital_peak_to_peak_codes": ("f8", np.nan),
    "digital_unique_count_capped3": ("u1", 255),
    "saturation_fraction": ("f4", np.nan),
    "high_amplitude_duration_seconds": ("u1", 255),
    "power_line_fraction": ("f4", np.nan),
    "high_frequency_fraction": ("f4", np.nan),
    "spo2_in_range_fraction": ("f4", np.nan),
}


def hard_bit(code):
    if not 1 <= code <= 7:
        raise ValueError(code)
    return np.uint8(1 << (code - 1))


def primary_reason(flags):
    result = np.zeros(np.shape(flags), np.uint8)
    for code in reversed(PRIORITY):
        result[(flags & hard_bit(code)) != 0] = code
    return result


def kind(name: str) -> str:
    index = CHANNELS.index(name)
    if index < 6:
        return "eeg"
    if index < 8:
        return "eog"
    if index == 8:
        return "ecg"
    if index < 12:
        return "emg"
    return name


def native(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def intervals(mask: np.ndarray):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return zip(starts, stops, strict=True)


def physical_values(raw: np.ndarray, attrs: dict, name: str):
    keys = ("dig_min", "dig_max", "phys_min", "phys_max")
    if not np.issubdtype(raw.dtype, np.number) or not all(k in attrs for k in keys):
        raise ValueError("source is not integer-coded with complete calibration")
    dmin, dmax, pmin, pmax = (float(attrs[k]) for k in keys)
    if not all(math.isfinite(v) for v in (dmin, dmax, pmin, pmax)):
        raise ValueError("nonfinite calibration")
    if dmax <= dmin or pmax <= pmin:
        raise ValueError("invalid calibration ordering")
    unit = str(attrs.get("unit", "")).strip().lower().replace("µ", "u").replace("μ", "u")
    values = pmin + (raw.astype(np.float64) - dmin) * ((pmax - pmin) / (dmax - dmin))
    if name == "spo2":
        if unit not in ("%", "percent", "percentage"):
            raise ValueError("SpO2 unit is not percent")
        output_unit = "%"
    else:
        scales = {"uv": 1.0, "microvolt": 1.0, "microvolts": 1.0, "mv": 1e3, "v": 1e6}
        if unit not in scales:
            raise ValueError("channel unit is missing or unsupported")
        values *= scales[unit]
        output_unit = "uV"
    return values, output_unit, (dmin, dmax, pmin, pmax)


def profile_accepted(name: str, attrs: dict) -> bool:
    unit = str(attrs.get("unit", "")).strip().lower().replace("µ", "u").replace("μ", "u")
    pmin = float(attrs.get("phys_min", math.nan))
    pmax = float(attrs.get("phys_max", math.nan))
    if name == "airflow":
        return pmin == -3200 and pmax == 3200 and unit == "uv"
    if name == "spo2":
        return pmin == 0 and pmax == 100 and unit in ("%", "percent", "percentage")
    return True


FILTER_RE = re.compile(r"\b(HP|LP)\s*:?\s*([-+]?\d+(?:\.\d+)?)\s*Hz", re.I)
NOTCH_RE = re.compile(r"\b(?:N|NOTCH)\s*:?\s*([-+]?\d+(?:\.\d+)?)\s*Hz", re.I)


def parse_prefilter(value: str):
    found: dict[str, float] = {}
    for key, text in FILTER_RE.findall(str(value)):
        key = key.upper()
        number = float(text)
        if key in found and found[key] != number:
            return None
        found[key] = number
    if set(found) != {"HP", "LP"}:
        return None
    hp, lp = found["HP"], found["LP"]
    if not all(math.isfinite(v) for v in (hp, lp)) or hp < 0 or lp < 0:
        return None
    return hp, lp


def filter_relation(name: str, parsed):
    if name == "spo2":
        return FILTER_RELATION["NOT_REQUIRED"]
    if parsed is None or parsed == (0.0, 0.0):
        return FILTER_RELATION["UNKNOWN"]
    target = TARGETS[kind(name)]
    narrower = parsed[0] > target[0] or parsed[1] < target[1]
    wider = parsed[0] < target[0] or parsed[1] > target[1]
    if narrower and wider:
        label = "MIXED"
    elif narrower:
        label = "SOURCE_NARROWER"
    elif wider:
        label = "SOURCE_WIDER"
    else:
        label = "MATCH"
    return FILTER_RELATION[label]


def design_supplement(name: str, fs: float):
    channel_kind = kind(name)
    low, high = TARGETS[channel_kind]
    if fs == 200 and (channel_kind == "emg" or name == "snore"):
        high = 95.0
    if high >= fs / 2:
        raise ValueError("filter cutoff reaches Nyquist")
    stages = [
        signal.butter(4, low, btype="highpass", fs=fs, output="sos"),
        signal.butter(4, high, btype="lowpass", fs=fs, output="sos"),
    ]
    radius = max(float(np.max(np.abs(signal.sos2zpk(sos)[1]))) for sos in stages)
    guard_seconds = math.ceil(math.log(1e-4) / math.log(radius) / fs)
    return stages, guard_seconds, (low, high)


def resample_run(values: np.ndarray, fs: float):
    ratio = Fraction(FS / fs).limit_denominator(10000)
    if not math.isclose(ratio.numerator / ratio.denominator, FS / fs, rel_tol=1e-12):
        raise ValueError("unsupported rational sample rate")
    if fs == FS:
        return values
    return signal.resample_poly(values, ratio.numerator, ratio.denominator, window=("kaiser", 5.0), padtype="line")



def spectral_metrics(x):
    """Batched, frozen Welch PSD on finite 200 Hz float32 publication candidates."""
    x = np.asarray(x, np.float64)
    if x.shape[-1] != EPOCH_SAMPLES or not np.isfinite(x).all():
        raise ValueError("PSD requires complete finite 6000-point epochs")
    frequencies, power = signal.welch(
        x, fs=FS, window=signal.windows.hann(800, sym=False),
        nperseg=800, noverlap=400, nfft=800, detrend="constant",
        return_onesided=True, scaling="density", axis=-1, average="mean",
    )
    total = power[..., (frequencies >= 0.5) & (frequencies <= 100)].sum(-1) * 0.25
    line_mask = ((frequencies >= 49) & (frequencies <= 51)) | ((frequencies >= 59) & (frequencies <= 61))
    line = power[..., line_mask].sum(-1) * 0.25
    hf = power[..., (frequencies > 70) & (frequencies <= 100)].sum(-1) * 0.25
    if not np.isfinite(total).all() or np.any(total < 0):
        raise FloatingPointError("invalid PSD denominator")
    return (np.divide(line, total, out=np.zeros_like(total), where=total > 0),
            np.divide(hf, total, out=np.zeros_like(total), where=total > 0))


def empty_channel(count, status, metadata, coverage=None):
    return {
        "data": np.zeros((count, EPOCH_SAMPLES), np.float32),
        "coverage_valid": np.zeros(count, bool) if coverage is None else coverage,
        "processing_valid": np.zeros(count, bool),
        "artifact_valid": np.zeros(count, bool),
        "valid": np.zeros(count, bool),
        "hard_code": np.zeros(count, np.uint8),
        "hard_flags": np.zeros(count, np.uint8),
        "hard_evaluated_flags": np.zeros(count, np.uint8),
        "metrics": {key: np.full(count, sentinel, dtype=dtype)
                    for key, (dtype, sentinel) in METRIC_SPECS.items()},
        "metadata": {**metadata, "channel_available": status == 0,
                     "channel_status_code": status, "usable_hours": 0.0},
    }


def audit_epochs(candidate, raw, attrs, name, coverage, source_fs):
    count = len(candidate)
    hard = np.zeros(count, np.uint8)
    evaluated = np.zeros(count, np.uint8)
    metrics = {key: np.full(count, sentinel, dtype=dtype)
               for key, (dtype, sentinel) in METRIC_SPECS.items()}
    epoch_size = round(source_fs * EPOCH_SECONDS)
    full = min(count, len(raw) // epoch_size)
    source_nonfinite = np.zeros(count, np.uint32)
    if full:
        source_nonfinite[:full] = (~np.isfinite(raw[:full * epoch_size].reshape(full, epoch_size))).sum(1)
        metrics["nonfinite_source_count"][:full] = source_nonfinite[:full]
    # An all-NaN placeholder from failed processing is not evidence of source NaN.
    has_candidate = np.isfinite(candidate).any(1)
    out_nf = (~np.isfinite(candidate)).sum(1)
    metrics["nonfinite_output_count"][has_candidate] = out_nf[has_candidate]
    nan_check = coverage & (has_candidate | (source_nonfinite > 0))
    evaluated[nan_check] |= hard_bit(1)
    bad_nan = nan_check & ((source_nonfinite > 0) | (has_candidate & (out_nf > 0)))
    hard[bad_nan] |= hard_bit(1)
    good = coverage & has_candidate & (out_nf == 0) & (source_nonfinite == 0)
    ids = np.flatnonzero(good)
    channel_kind = kind(name)

    def mark(indices, code, bad):
        evaluated[indices] |= hard_bit(code)
        hard[indices[np.asarray(bad, bool)]] |= hard_bit(code)

    for offset in range(0, len(ids), 64):
        ii = ids[offset:offset + 64]
        x = candidate[ii].astype(np.float64)
        if channel_kind in ("eeg", "eog", "ecg"):
            std = x.std(1, ddof=0)
            metrics["std_physical"][ii] = std
            mark(ii, 2, std < (5.0 if name == "ecg" else 0.5))
        if channel_kind in ("eeg", "eog", "ecg", "emg"):
            over = np.ptp(x.reshape(-1, 30, 200), axis=-1) > {
                "eeg": 1000, "eog": 2000, "ecg": 10000, "emg": 5000,
            }[channel_kind]
            duration = over.sum(1)
            metrics["high_amplitude_duration_seconds"][ii] = duration
            mark(ii, 4, duration == 30)
            line, hf = spectral_metrics(x)
            metrics["power_line_fraction"][ii] = line
            mark(ii, 5, line > 0.40)
            if channel_kind != "emg":
                metrics["high_frequency_fraction"][ii] = hf
                mark(ii, 6, hf > 0.50)
            # Native digital data, not resampled values.
            digital = raw[:full * epoch_size].reshape(full, epoch_size)[ii]
            fraction = ((digital <= float(attrs["dig_min"]) + 1) |
                        (digital >= float(attrs["dig_max"]) - 1)).mean(1)
            metrics["saturation_fraction"][ii] = fraction
            mark(ii, 3, fraction >= 0.05)
        if name == "airflow":
            digital = raw[:full * epoch_size].reshape(full, epoch_size)[ii].astype(np.float64)
            span = np.ptp(digital, axis=1)
            # Only <=2-code-span rows can pass; count unique codes capped at 3.
            unique = np.full(len(ii), 3, np.uint8)
            for j in np.flatnonzero(span <= 2):
                unique[j] = min(3, len(np.unique(digital[j])))
            metrics["digital_peak_to_peak_codes"][ii] = span
            metrics["digital_unique_count_capped3"][ii] = unique
            mark(ii, 2, (span <= 2) & (unique <= 2))
        if name == "spo2":
            inside = ((x >= 50) & (x <= 110)).mean(1)
            metrics["spo2_in_range_fraction"][ii] = inside
            mark(ii, 7, inside < 0.70)
    return hard, evaluated, metrics


def process_channel(raw, attrs, name, count):
    meta = {"canonical_channel": name, "source_present": raw is not None or bool(attrs.get("_source_present")),
            "source_attrs": jsonable(attrs), "coverage_detection": COVERAGE_DETECTION["NOT_EVALUATED"],
            "errors": []}
    if attrs.get("_read_error"):
        return empty_channel(count, 2, {**meta, "errors": [attrs["_read_error"]]})
    if raw is None and not attrs.get("_source_present"):
        return empty_channel(count, 1, meta)
    try:
        fs = float(attrs["fs"])
        if not math.isfinite(fs) or fs <= 0 or not math.isclose(fs * 30, round(fs * 30), abs_tol=1e-8):
            raise ValueError("invalid source sampling rate")
        source_samples = len(raw) if raw is not None else int(attrs["_source_samples"])
        coverage = (np.arange(1, count + 1) * round(30 * fs)) <= source_samples
        meta.update(source_fs_hz=fs, source_samples=source_samples, coverage_detection=COVERAGE_DETECTION["LENGTH_ONLY"])
    except (ValueError, KeyError, TypeError) as error:
        return empty_channel(count, 3, {**meta, "errors": [str(error)]})
    if not profile_accepted(name, attrs):
        return empty_channel(count, 4, {
            **meta, "effective_filter_state": EFFECTIVE_FILTER["PROCESSING_SKIPPED_PROFILE_EXCLUDED"],
            "filter_reapply_result": PROCESS_RESULT["NOT_RUN_PROFILE_EXCLUDED"],
            "resample_result": PROCESS_RESULT["NOT_RUN_PROFILE_EXCLUDED"],
            "calibration_profile_accepted": False, "waveform_read": False,
        }, coverage)
    try:
        values, unit, calibration = physical_values(raw, attrs, name)
    except (ValueError, KeyError, TypeError) as error:
        return empty_channel(count, 3, {**meta, "errors": [str(error)]}, coverage)
    meta.update(output_unit=unit, calibration=calibration, calibration_arithmetic_valid=True,
                calibration_profile_accepted=True, waveform_read=True)
    # Respect the root recording boundary, retaining its fractional final epoch for filtering context.
    source_stop = min(len(values), math.ceil(float(attrs.get("_duration_sec", count * 30)) * fs))
    values = values[:source_stop]
    raw = raw[:source_stop]
    parsed = parse_prefilter(str(attrs.get("prefilter", "")))
    unknown = name != "spo2" and (parsed is None or parsed == (0, 0))
    applied = None
    stages = []
    guard = 0
    meta.update(
        source_filter_state=FILTER_STATE["NOT_EVALUATED"] if name == "spo2" else
            FILTER_STATE["SOURCE_FILTER_UNKNOWN" if unknown else "SOURCE_FILTER_EXPLICIT"],
        filter_relation=filter_relation(name, parsed), source_prefilter=str(attrs.get("prefilter", "")),
        parsed_source_hp_lp=parsed, parsed_source_notch_hz=[float(v) for v in NOTCH_RE.findall(str(attrs.get("prefilter", "")))],
        filter_reapply_result=PROCESS_RESULT["NOT_REQUIRED"],
        effective_filter_state=EFFECTIVE_FILTER["FILTER_NOT_REQUIRED" if name == "spo2" else "SOURCE_FILTER_AS_RECORDED"],
    )
    output = np.full(count * EPOCH_SAMPLES, np.nan, np.float64)
    runs = 0
    failures = 0
    try:
        if unknown:
            stages, guard, applied = design_supplement(name, fs)
    except ValueError as error:
        result = empty_channel(count, 0, {**meta, "errors": [str(error)],
            "effective_filter_state": EFFECTIVE_FILTER["REAPPLY_FAILED"],
            "filter_reapply_result": PROCESS_RESULT["UNSUPPORTED"]}, coverage)
        return result
    # Only true nonfinite samples split the filtering run. Artifacts never split it.
    for first, stop in intervals(np.isfinite(values)):
        first, stop = int(first), int(stop)
        run = values[first:stop]
        try:
            for sos in stages:
                # SciPy's documented odd-reflection padding; no guard-based epoch rejection.
                run = signal.sosfiltfilt(sos, run, padtype="odd")
            left = max(0, math.ceil(first * FS / fs))
            right = min(len(output), math.ceil(stop * FS / fs))
            if right <= left:
                continue
            if name == "spo2":
                indices = np.floor(np.arange(left, right, dtype=np.float64) * fs / FS + 1e-10).astype(np.int64) - first
                if np.any(indices < 0) or np.any(indices >= len(run)):
                    raise ValueError("SpO2 interpolation crossed native coverage")
                transformed = run[indices]
            elif fs == FS:
                transformed = run[left - first:right - first]
            else:
                # Align the rational resampler's phase to recording t=0 after a true gap.
                ratio = Fraction(FS / fs).limit_denominator(10000)
                prefix = first % ratio.denominator
                aligned = np.pad(run, (prefix, 0), mode="reflect" if len(run) > 1 else "edge") if prefix else run
                sampled = resample_run(aligned, fs)
                start_index = round((first - prefix) * FS / fs)
                transformed = sampled[left - start_index:right - start_index]
            if len(transformed) != right - left:
                raise ValueError("resampled output length mismatch")
            output[left:right] = transformed
            runs += 1
        except (ValueError, FloatingPointError) as error:
            failures += 1
            meta["errors"].append(f"native_run[{first}:{stop}]: {error}")
    if unknown:
        meta["effective_filter_state"] = EFFECTIVE_FILTER["SOURCE_FILTER_REAPPLY" if runs else "REAPPLY_FAILED"]
        meta["filter_reapply_result"] = PROCESS_RESULT["PARTIAL" if runs and failures else "SUCCESS" if runs else "FAILED"]
    meta.update(applied_hp_lp=applied, filter_guard_seconds=guard, processed_runs=runs,
                failed_runs=failures, resample_result=PROCESS_RESULT[
                    "PARTIAL" if runs and failures else "FAILED" if not runs else
                    "NOT_REQUIRED" if fs == FS else "SUCCESS"])
    # QC sees the exact float32 values that will be stored, before replacing nonfinite positions.
    with np.errstate(over="ignore", invalid="ignore"):
        candidate = output.reshape(count, EPOCH_SAMPLES).astype(np.float32)
    hard, evaluated, metrics = audit_epochs(candidate, raw, attrs, name, coverage, fs)
    processing = coverage & np.isfinite(candidate).all(1)
    available = True
    status = 0
    if name == "spo2":
        finite_values = values[np.isfinite(values)]
        if len(finite_values) and ((finite_values < 5).mean() >= 0.90 or np.median(finite_values) < 5):
            available, status = False, CHANNEL_STATUS["SPO2_NEAR_ZERO"]
    applicability = APPLICABLE[CHANNELS.index(name)]
    artifact = ((evaluated & applicability) == applicability) & (hard == 0)
    valid = available & coverage & processing & artifact
    meta.update(channel_available=available, channel_status_code=status, usable_hours=float(valid.sum() / 120))
    return {"data": np.where(np.isfinite(candidate), candidate, 0).astype(np.float32),
            "coverage_valid": coverage, "processing_valid": processing, "artifact_valid": artifact,
            "valid": valid, "hard_code": primary_reason(hard), "hard_flags": hard,
            "hard_evaluated_flags": evaluated, "metrics": metrics, "metadata": meta}
