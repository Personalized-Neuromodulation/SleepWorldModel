"""Stable subject-level splits; sessions from one subject never cross splits."""

import hashlib
import math


def subject_split(
    dataset: str,
    subject: str,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> str:
    if (
        len(ratios) != 3
        or any(x < 0 for x in ratios)
        or not math.isclose(sum(ratios), 1)
    ):
        raise ValueError("split ratios must be three nonnegative values summing to 1")
    digest = hashlib.sha256(f"{seed}:{dataset}:{subject}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return (
        "train"
        if value < ratios[0]
        else "validation"
        if value < sum(ratios[:2])
        else "test"
    )
