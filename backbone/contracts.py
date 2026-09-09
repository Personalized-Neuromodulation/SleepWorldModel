"""Tensor contracts shared by PSG modeling and independent architecture families."""

from dataclasses import dataclass, field

from torch import Tensor


@dataclass(frozen=True)
class PatchLayout:
    sample_rate_hz: float
    patch_samples: int
    patches_per_epoch: int
    epoch_count: int
    epoch_in_record: Tensor  # [B,N]
    sample_start: Tensor  # [N], offset inside its epoch
    time_intervals_ns: Tensor  # [B,N,2], recording-relative [start,end)


@dataclass(frozen=True)
class MaskPlan:
    stage: str = "none"
    visible: dict[str, Tensor] = field(default_factory=dict)

    def __post_init__(self):
        if self.stage not in ("none", "waveform", "token"):
            raise ValueError("mask stage must be none, waveform or token")
        if self.stage == "none" and self.visible:
            raise ValueError("none mask cannot contain visibility tensors")


@dataclass(frozen=True)
class PatchBatch:
    values: Tensor  # [B,C,N,L]
    sample_visible: Tensor  # [B,C,N,L]
    data_valid: Tensor  # [B,C,N]
    layout: PatchLayout
    channel_ids: tuple[str, ...]


@dataclass(frozen=True)
class SequenceState:
    tokens: Tensor  # [M,T,H]
    data_valid: Tensor  # [M,T]
    visible: Tensor
    positions: Tensor  # [M,T], integer nanoseconds
    time_intervals_ns: Tensor  # [M,T,2]
    context_intervals_ns: Tensor
    available_at_ns: Tensor  # -1 means unknown
    connection_mask: Tensor | None = None  # [M,T,T], True permits an edge

    @property
    def active(self) -> Tensor:
        return self.data_valid & self.visible


@dataclass(frozen=True)
class TokenGrid:
    tokens: Tensor  # [B,C,N,D]
    data_valid: Tensor  # [B,C,N]
    visible: Tensor
    coverage: Tensor
    time_intervals_ns: Tensor  # [B,N,2]
    context_intervals_ns: Tensor
    available_at_ns: Tensor
    channel_ids: tuple[str, ...]
    patch_layout: PatchLayout

    @property
    def active(self) -> Tensor:
        return self.data_valid & self.visible


@dataclass(frozen=True)
class TokenSequence:
    tokens: Tensor  # [B,N,D]
    data_valid: Tensor  # [B,N]
    visible: Tensor
    coverage: Tensor
    time_intervals_ns: Tensor  # [B,N,2]
    context_intervals_ns: Tensor
    available_at_ns: Tensor
    support_count: Tensor  # [B,N], original channel support denominator

    @property
    def active(self) -> Tensor:
        return self.data_valid & self.visible


@dataclass
class SignalOutput:
    patch_tokens: TokenGrid | None = None
    local: TokenGrid | None = None


@dataclass
class ModalityOutput(SignalOutput):
    features: TokenSequence | None = None


@dataclass
class BackboneOutput:
    patch_tokens: dict[str, TokenGrid] = field(default_factory=dict)
    local: dict[str, TokenGrid] = field(default_factory=dict)
    features: dict[str, TokenSequence] = field(default_factory=dict)
    joint: TokenSequence | None = None
    aux: dict[str, Tensor] = field(default_factory=dict)
