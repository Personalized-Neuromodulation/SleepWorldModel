"""Settings for the minimal SSL baseline."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SSLConfig:
    channels: dict[str, int]
    sample_rates: dict[str, float]
    epoch_seconds: float = 30.0
    hidden_dim: int = 32
    embedding_dim: int = 128
    projection_dim: int = 64
    crop_fraction: float = 0.8

    def __post_init__(self):
        if not self.channels or self.channels.keys() != self.sample_rates.keys():
            raise ValueError(
                "channels and sample_rates must describe the same modalities"
            )
        if any(not name or "." in name for name in self.channels):
            raise ValueError("modality names must be nonempty and cannot contain dots")
        if any(not isinstance(n, int) or n < 1 for n in self.channels.values()):
            raise ValueError("channel counts must be positive integers")
        if any(not math.isfinite(fs) or fs <= 0 for fs in self.sample_rates.values()):
            raise ValueError("sample rates must be finite and positive")
        if not math.isfinite(self.epoch_seconds) or self.epoch_seconds <= 0:
            raise ValueError("epoch_seconds must be finite and positive")
        if min(self.hidden_dim, self.embedding_dim, self.projection_dim) < 1:
            raise ValueError("model dimensions must be positive")
        if not 0 < self.crop_fraction < 1:
            raise ValueError("crop_fraction must be between 0 and 1")
        for fs in self.sample_rates.values():
            samples = fs * self.epoch_seconds
            if (
                not math.isclose(samples, round(samples))
                or int(samples * self.crop_fraction) < 2
            ):
                raise ValueError(
                    "each epoch/crop must have a valid integer sample length"
                )
