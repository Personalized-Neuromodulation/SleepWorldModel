"""Serializable configuration for the single PSG foundation backbone."""

import math
from dataclasses import asdict, dataclass, field

from .architectures.cnn.configuration import PatchTokenizerConfig
from .architectures.moe import MoEConfig
from .architectures.numeric import NumericTokenizerConfig


def validate_attention(heads, ffn_dim, dim):
    if type(heads) is not int or heads < 1 or dim % heads:
        raise ValueError(f"attention heads must divide {dim}")
    if type(ffn_dim) is not int or ffn_dim < 1:
        raise ValueError("ffn_dim must be a positive integer")


@dataclass(frozen=True)
class CrissCrossConfig:
    heads_per_branch: int = 4
    ffn_dim: int = 1024

    def __post_init__(self):
        validate_attention(self.heads_per_branch, self.ffn_dim, 128)


@dataclass(frozen=True)
class TemporalConfig:
    num_heads: int = 8
    ffn_dim: int = 1024

    def __post_init__(self):
        validate_attention(self.num_heads, self.ffn_dim, 256)


@dataclass(frozen=True)
class ReadoutConfig:
    """Independent controls for latent scale and modality aggregation."""

    encoder_output_norm: bool = True
    fusion_output_norm: bool = False
    modality_pooling: str = "attention"
    score_kind: str = "cosine"
    cosine_scale: float = 2.0
    temporal_pooling: str = "attention"
    preferred_modalities: list[str] = field(default_factory=list)

    def __post_init__(self):
        if (not isinstance(self.preferred_modalities, list)
                or any(not isinstance(n, str) or not n for n in self.preferred_modalities)
                or len(set(self.preferred_modalities)) != len(self.preferred_modalities)):
            raise ValueError("preferred_modalities must contain unique modality names")
        if any(
            type(v) is not bool
            for v in (self.encoder_output_norm, self.fusion_output_norm)
        ):
            raise ValueError("output normalization switches must be boolean")
        if self.modality_pooling not in ("mean", "attention"):
            raise ValueError("modality_pooling must be mean or attention")
        if self.temporal_pooling not in ("mean", "attention"):
            raise ValueError("temporal_pooling must be mean or attention")
        if self.score_kind not in ("linear", "normalized_linear", "cosine"):
            raise ValueError("unknown pooling score_kind")
        if not math.isfinite(self.cosine_scale) or self.cosine_scale <= 0:
            raise ValueError("cosine_scale must be finite and positive")


@dataclass(frozen=True)
class BackboneConfig:
    feature_dim: int = 256
    patch_seconds: float = 1.0
    sample_rate: int = 200
    patch_tokenizer: PatchTokenizerConfig = field(default_factory=PatchTokenizerConfig)
    numeric_tokenizers: dict[str, NumericTokenizerConfig] = field(default_factory=dict)
    criss_cross: CrissCrossConfig = field(default_factory=CrissCrossConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    fusion_attention: CrissCrossConfig = field(default_factory=CrissCrossConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    channel_aggregation: str = "attention"
    channel_embedding: bool = True
    modality_embedding: bool = True
    modality_encoder: dict = field(
        default_factory=lambda: {
            "eeg_depth": 4,
            "multichannel_depth": 4,
            "single_channel_depth": 4,
        }
    )
    fusion_depth: int = 4
    post_fusion_temporal_depth: int = 0
    dropout: float = 0.0

    def __post_init__(self):
        if (self.feature_dim, self.patch_seconds, self.sample_rate) != (256, 1, 200):
            raise ValueError("backbone requires D=256, 1 second and 200 Hz")
        self.patch_tokenizer.validate_output(self.sample_rate, self.feature_dim)
        if not isinstance(self.numeric_tokenizers, dict) or any(
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(spec, NumericTokenizerConfig)
            for name, spec in self.numeric_tokenizers.items()
        ):
            raise ValueError(
                "numeric_tokenizers maps modality names to numeric configs"
            )
        if self.channel_aggregation not in ("mean", "attention"):
            raise ValueError("channel_aggregation must be mean or attention")
        expected = {"eeg_depth", "multichannel_depth", "single_channel_depth"}
        if set(self.modality_encoder) != expected:
            raise ValueError("modality_encoder requires the three depth keys")
        depths = [*self.modality_encoder.values(), self.fusion_depth]
        if any(type(d) is not int or d < 1 for d in depths):
            raise ValueError("encoder depths must be positive integers")
        if (
            type(self.post_fusion_temporal_depth) is not int
            or not 0 <= self.post_fusion_temporal_depth <= 2
        ):
            raise ValueError("post fusion depth must be 0–2")
        if not 0 <= self.dropout < 1:
            raise ValueError("network dropout must be in [0,1)")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "BackboneConfig":
        """Load YAML or serialized config; removed and unknown options fail explicitly."""
        values = dict(values)
        # Old foundation exports lack readout controls. Preserve their exact
        # architecture; new constructors/YAML explicitly serialize the defaults.
        if "readout" not in values:
            values["readout"] = {
                "encoder_output_norm": False,
                "fusion_output_norm": False,
                "score_kind": "linear",
            }
        if "fusion" in values:
            section = dict(values.pop("fusion"))
            if "depth" not in section:
                raise ValueError("fusion config requires depth")
            if "fusion_depth" in values or "fusion_attention" in values:
                raise ValueError("provide fusion or serialized fusion fields, not both")
            values["fusion_depth"] = section.pop("depth")
            values["fusion_attention"] = section
        for key, config_type in (
            ("criss_cross", CrissCrossConfig),
            ("temporal", TemporalConfig),
            ("fusion_attention", CrissCrossConfig),
            ("readout", ReadoutConfig),
            ("moe", MoEConfig),
        ):
            if key in values:
                values[key] = config_type(**values[key])
        if "patch_tokenizer" in values:
            values["patch_tokenizer"] = PatchTokenizerConfig.from_dict(
                values["patch_tokenizer"]
            )
        if "numeric_tokenizers" in values:
            values["numeric_tokenizers"] = {
                name: NumericTokenizerConfig(**spec)
                for name, spec in values["numeric_tokenizers"].items()
            }
        return cls(**values)
