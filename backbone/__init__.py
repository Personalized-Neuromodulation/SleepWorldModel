"""Reusable continuous PSG features, independent of pretraining objectives."""

from .configuration import (
    BackboneConfig,
    BlockConfig,
    PatchEncoderConfig,
    SequenceEncoderConfig,
)
from .contracts import BackboneOutput, MaskPlan, TokenGrid, TokenSequence
from .factory import build_backbone
from .modeling.psg_backbone import PSGBackbone

__all__ = [
    "BackboneConfig",
    "BlockConfig",
    "PatchEncoderConfig",
    "SequenceEncoderConfig",
    "BackboneOutput",
    "MaskPlan",
    "TokenGrid",
    "TokenSequence",
    "PSGBackbone",
    "build_backbone",
]
