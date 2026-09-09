from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ChannelSpec:
    """One canonical model input and the HSP signal names that can provide it."""

    name: str
    candidates: tuple[str, ...]


def channel(name: str, *fallbacks: str) -> ChannelSpec:
    return ChannelSpec(name=name, candidates=(name, *fallbacks))


DEFAULT_SAMPLE_RATES: dict[str, float] = {
    "eeg": 100.0,
    "eog": 100.0,
    "emg": 100.0,
    "ecg": 100.0,
    "resp": 25.0,
    "spo2": 25.0,
}


CHANNEL_PROFILES: dict[str, dict[str, tuple[ChannelSpec, ...]]] = {
    "hsp_full": {
        "eeg": (
            channel("f3-m2"),
            channel("f4-m1"),
            channel("c3-m2"),
            channel("c4-m1"),
            channel("o1-m2"),
            channel("o2-m1"),
        ),
        "eog": (channel("e1"), channel("e2")),
        "emg": (channel("chin1-chin2"),),
        "ecg": (channel("ecg"),),
        "resp": (
            channel("abd"),
            channel("chest"),
            # PTAF is the pressure-flow fallback when thermistor airflow is absent.
            # Its 25/200/500 Hz source rate is normalized per channel by the loader.
            channel("airflow", "ptaf"),
        ),
        "spo2": (channel("spo2"),),
    },
    "hsp_neuro": {
        "eeg": (
            channel("f3-m2"),
            channel("f4-m1"),
            channel("c3-m2"),
            channel("c4-m1"),
            channel("o1-m2"),
            channel("o2-m1"),
        ),
        "eog": (channel("e1"), channel("e2")),
        "emg": (channel("chin1-chin2"),),
    },
    "hsp_jepa7": {
        "eeg": (ChannelSpec("central-eeg", ("c4-m1", "c3-m2")),),
        "eog": (ChannelSpec("eog", ("e1", "e2")),),
        "emg": (channel("chin1-chin2"),),
        "ecg": (channel("ecg"),),
        "resp": (channel("abd"), channel("chest")),
        "spo2": (channel("spo2"),),
    },
}


ChannelInput = str | ChannelSpec | Sequence[str]


def resolve_channel_groups(
    profile: str,
    channel_groups: Mapping[str, Sequence[ChannelInput]] | None,
) -> dict[str, tuple[ChannelSpec, ...]]:
    if channel_groups is None:
        if profile not in CHANNEL_PROFILES:
            raise ValueError(
                f"unknown channel profile {profile!r}; available: {sorted(CHANNEL_PROFILES)}"
            )
        return CHANNEL_PROFILES[profile]

    result: dict[str, tuple[ChannelSpec, ...]] = {}
    for modality, entries in channel_groups.items():
        specs: list[ChannelSpec] = []
        for entry in entries:
            if isinstance(entry, ChannelSpec):
                specs.append(entry)
            elif isinstance(entry, str):
                specs.append(channel(entry.lower()))
            else:
                candidates = tuple(str(value).lower() for value in entry)
                if not candidates:
                    raise ValueError(f"empty candidate list for modality {modality!r}")
                specs.append(ChannelSpec(candidates[0], candidates))
        if not specs:
            raise ValueError(f"modality {modality!r} has no channels")
        result[str(modality)] = tuple(specs)
    if not result:
        raise ValueError("channel_groups cannot be empty")
    return result


__all__ = [
    "CHANNEL_PROFILES",
    "DEFAULT_SAMPLE_RATES",
    "ChannelInput",
    "ChannelSpec",
    "resolve_channel_groups",
]
