"""Shape-preserving CNN configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CNNConfig:
    kernel_size: int = 3

    def __post_init__(self):
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
