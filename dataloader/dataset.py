"""Window indexing shared by independently registered dataset readers."""

import bisect
from pathlib import Path
from typing import Callable

from torch.utils.data import Dataset

from .sampler import subject_split
from .schema import Reader

_READERS: dict[str, Callable[..., Reader]] = {}


def register_reader(name: str, factory: Callable[..., Reader]) -> None:
    if not name or name in _READERS or name == "hsp":
        raise ValueError(f"reader name is empty or already registered: {name!r}")
    _READERS[name] = factory


class WindowDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        dataset: str = "hsp",
        version: str = "v1.0.0",
        split: str | None = None,
        split_seed: int = 42,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        context_epochs: int = 20,
        stride_epochs: int | None = None,
        drop_last: bool = False,
        tasks: tuple[str, ...] = (),
        night_grades: tuple[int, ...] | None = None,
        **reader_options,
    ):
        stride = context_epochs if stride_epochs is None else stride_epochs
        if night_grades is not None and (
            not night_grades
            or any(type(g) is not int or g not in range(1, 6) for g in night_grades)
        ):
            raise ValueError("night_grades must select integer grades 1–5")
        if context_epochs <= 0 or stride <= 0:
            raise ValueError("context_epochs and stride_epochs must be positive")
        if split not in (None, "train", "validation", "test"):
            raise ValueError("split must be train, validation, test, or None")
        # Validate the ratios even when no split is selected.
        subject_split(dataset, "", split_seed, split_ratios)
        if dataset == "hsp":
            from .readers.hsp import HSPReleaseReader

            factory = HSPReleaseReader
        else:
            try:
                factory = _READERS[dataset]
            except KeyError as error:
                raise ValueError(f"unknown dataset reader: {dataset!r}") from error
        self.reader = factory(
            Path(root), version=version, tasks=tasks, **reader_options
        )
        self.metadata = self.reader.metadata
        self.context_epochs, self.stride_epochs = context_epochs, stride
        self.records, self._ends = [], []
        total = 0
        for record in self.reader.records:
            if (
                night_grades is not None
                and record.get("night_grade") not in night_grades
            ):
                continue
            if (
                split is not None
                and subject_split(
                    dataset, record["subject_id"], split_seed, split_ratios
                )
                != split
            ):
                continue
            epochs = int(record["n_epochs"])
            count = (
                max(0, (epochs - context_epochs) // stride + 1)
                if drop_last
                else (epochs + stride - 1) // stride
            )
            if count:
                self.records.append(record)
                total += count
                self._ends.append(total)
        if not self.records:
            self.close()
            raise ValueError("no windows match the requested split and window length")
        self.split = split
        self.split_seed, self.split_ratios = split_seed, split_ratios

    def __len__(self):
        return self._ends[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        pos = bisect.bisect_right(self._ends, index)
        local = index - (self._ends[pos - 1] if pos else 0)
        start = local * self.stride_epochs
        record = self.records[pos]
        return self.reader.read_window(
            record, start, min(self.context_epochs, int(record["n_epochs"]) - start)
        )

    def close(self):
        if hasattr(self, "reader"):
            self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
