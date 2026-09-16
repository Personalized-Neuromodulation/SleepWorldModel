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
    target_mask: dict[str, Tensor] = field(default_factory=dict)

    def __post_init__(self):
        if self.stage not in ("none", "waveform", "token"):
            raise ValueError("mask stage must be none, waveform or token")
        if self.stage == "none" and self.visible:
            raise ValueError("none mask cannot contain visibility tensors")


@dataclass(frozen=True)
class PatchBatch:
    values: Tensor  # [B,C,N,L]
    sample_visible: Tensor  # [B,C,N,L], view visibility only, independent of QC
    data_valid: Tensor  # [B,C,E], prepared source epoch QC
    layout: PatchLayout
    channel_ids: tuple[str, ...]

    @property
    def token_valid(self):
        return self.data_valid.repeat_interleave(self.layout.patches_per_epoch, -1)


@dataclass(frozen=True)
class TokenGrid:
    tokens: Tensor  # [B,C,N,D]
    data_valid: Tensor  # [B,C,E], prepared source epoch QC
    visible: Tensor
    coverage: Tensor
    time_intervals_ns: Tensor  # [B,N,2]
    context_intervals_ns: Tensor
    available_at_ns: Tensor
    channel_ids: tuple[str, ...]
    patch_layout: PatchLayout

    @property
    def token_valid(self):
        return self.data_valid.repeat_interleave(
            self.patch_layout.patches_per_epoch, -1
        )

    @property
    def active(self) -> Tensor:
        return self.token_valid & self.visible


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
class BackboneOutput:
    patch_tokens: dict[str, TokenGrid] = field(default_factory=dict)
    local: dict[str, TokenGrid] = field(default_factory=dict)
    features: dict[str, TokenSequence] = field(default_factory=dict)
    joint: TokenSequence | None = None
    aux: dict[str, Tensor] = field(default_factory=dict)
    foundation_representation: Tensor | None = None
    valid: Tensor | None = None
    fused_features: dict[str, TokenSequence] = field(default_factory=dict)
