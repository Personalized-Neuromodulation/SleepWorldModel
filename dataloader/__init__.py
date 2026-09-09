"""Versioned dataset readers, record-bounded windows, QC-aware batches."""

from .collate import collate_windows
from .dataset import WindowDataset, register_reader
from .schema import DatasetMetadata, Reader
from .signals import SignalBatch, SignalGroup, as_signal_batch

__all__ = [
    "WindowDataset",
    "collate_windows",
    "register_reader",
    "DatasetMetadata",
    "Reader",
    "SignalBatch",
    "SignalGroup",
    "as_signal_batch",
]
