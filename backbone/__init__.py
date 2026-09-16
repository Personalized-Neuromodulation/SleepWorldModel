"""Reusable continuous PSG features, independent of pretraining objectives."""

from .configuration import BackboneConfig
from .contracts import BackboneOutput, MaskPlan, TokenGrid, TokenSequence
from .factory import build_backbone
from .modeling.foundation import FoundationBackbone

__all__ = [
    "BackboneConfig",
    "BackboneOutput",
    "MaskPlan",
    "TokenGrid",
    "TokenSequence",
    "FoundationBackbone",
    "build_backbone",
]
