"""Dataset-independent metadata and the reader extension contract."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    version: str
    channels: dict[str, tuple[str, ...]]
    sample_rates: dict[str, float]
    units: dict[str, tuple[str, ...]]
    epoch_seconds: float = 30.0


class Reader(Protocol):
    metadata: DatasetMetadata
    records: list[dict[str, Any]]

    def read_window(self, record: dict, start: int, count: int) -> dict: ...

    def close(self) -> None: ...
