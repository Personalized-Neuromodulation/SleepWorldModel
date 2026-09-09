"""Human Sleep Project manifest and lazy multi-modal data loading."""

from .channels import CHANNEL_PROFILES, DEFAULT_SAMPLE_RATES, ChannelSpec
from .collate import hsp_collate_fn
from .dataset import HSPDataset
from .errors import HSPDataError
from .manifest import (
    build_hsp_manifest,
    inspect_hsp_file,
    iter_hsp_files,
    load_hsp_manifest,
    subject_split,
)
from .transforms import RandomChannelDropout

__all__ = [
    "CHANNEL_PROFILES",
    "DEFAULT_SAMPLE_RATES",
    "ChannelSpec",
    "HSPDataError",
    "HSPDataset",
    "RandomChannelDropout",
    "build_hsp_manifest",
    "hsp_collate_fn",
    "inspect_hsp_file",
    "iter_hsp_files",
    "load_hsp_manifest",
    "subject_split",
]
