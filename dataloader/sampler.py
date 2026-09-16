"""Stable subject-level splits; sessions from one subject never cross splits."""

import hashlib
import math

import torch
from torch.utils.data import Sampler


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


class NightGradeBatchSampler(Sampler):
    """Draw grade-homogeneous batches without materializing all window indices.

    Choose grades uniformly, then windows uniformly within that grade. Sampling
    is with replacement; only the small recording span table is held in memory.
    """

    def __init__(self, dataset, batch_size, num_batches, seed=42):
        self.batch_size, self.num_batches, self.seed = batch_size, num_batches, seed
        by_grade = {}
        start = 0
        for record, end in zip(dataset.records, dataset._ends):
            grade = record.get("night_grade")
            if grade is None:
                raise ValueError("grade-based batches require night_grade metadata")
            by_grade.setdefault(grade, []).append((start, end - start))
            start = end
        self.spans = []
        for grade in sorted(by_grade):
            spans = by_grade[grade]
            starts = torch.tensor([s for s, _ in spans])
            ends = torch.tensor([n for _, n in spans]).cumsum(0)
            self.spans.append((starts, ends))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.num_batches):
            which = torch.randint(len(self.spans), (), generator=generator).item()
            starts, ends = self.spans[which]
            offsets = torch.randint(
                int(ends[-1]), (self.batch_size,), generator=generator
            )
            spans = torch.searchsorted(ends, offsets, right=True)
            preceding = torch.cat((torch.zeros(1, dtype=torch.long), ends[:-1]))
            yield (starts[spans] + offsets - preceding[spans]).tolist()
