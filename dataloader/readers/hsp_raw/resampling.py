from __future__ import annotations

import torch
import torch.nn.functional as functional


def decimate(values: torch.Tensor, factor: int) -> torch.Tensor:
    """Integer-factor FIR decimation with a windowed-sinc anti-alias filter."""
    if factor == 1:
        return values
    if factor <= 0:
        raise ValueError("decimation factor must be positive")

    taps = 20 * factor + 1
    half = taps // 2
    positions = torch.arange(-half, half + 1, dtype=values.dtype)
    cutoff = 0.45 / factor
    kernel = 2.0 * cutoff * torch.sinc(2.0 * cutoff * positions)
    kernel *= torch.hann_window(taps, periodic=False, dtype=values.dtype)
    kernel /= kernel.sum()

    original_length = values.numel()
    signal = values.reshape(1, 1, original_length)
    padding_mode = "reflect" if original_length > half else "replicate"
    signal = functional.pad(signal, (half, half), mode=padding_mode)
    filtered = functional.conv1d(signal, kernel.reshape(1, 1, -1)).flatten()
    output = filtered[::factor]
    expected = int(round(original_length / factor))
    return output[:expected]


__all__ = ["decimate"]
