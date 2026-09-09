"""Small, serializable configuration objects; no networks are built here."""

from dataclasses import asdict, dataclass, field

from .architectures.cnn.configuration import CNNConfig
from .architectures.transformer.configuration import TransformerConfig


@dataclass(frozen=True)
class BlockConfig:
    kind: str = "transformer"
    depth: int = 1
    cnn: CNNConfig = field(default_factory=CNNConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)

    def __post_init__(self):
        if self.kind not in ("cnn", "transformer") or self.depth < 1:
            raise ValueError("block kind must be cnn/transformer with positive depth")


@dataclass(frozen=True)
class PatchEncoderConfig:
    variant: str = "sequence"
    subpatch_samples: int = 20
    hidden_dim: int = 64
    position_embedding: bool = True
    readout: str = "mean"
    blocks: tuple[BlockConfig, ...] = field(
        default_factory=lambda: (BlockConfig(kind="cnn"),)
    )

    def __post_init__(self):
        if self.variant not in ("direct_linear", "sequence"):
            raise ValueError("patch variant must be direct_linear or sequence")
        if self.subpatch_samples < 1 or self.hidden_dim < 1:
            raise ValueError("patch dimensions must be positive")
        if self.readout not in ("mean", "attention"):
            raise ValueError("patch readout must be mean or attention")


@dataclass(frozen=True)
class SequenceEncoderConfig:
    window_seconds: float = 30.0
    position_embedding: bool = True
    causal: bool = False
    blocks: tuple[BlockConfig, ...] = field(default_factory=lambda: (BlockConfig(),))

    def __post_init__(self):
        if not 0 < self.window_seconds < float("inf"):
            raise ValueError("window_seconds must be finite and positive")


@dataclass(frozen=True)
class BackboneConfig:
    feature_dim: int = 128
    patch_seconds: float = 1.0
    patch_encoder: PatchEncoderConfig = field(default_factory=PatchEncoderConfig)
    sequence_encoder: SequenceEncoderConfig = field(
        default_factory=SequenceEncoderConfig
    )
    channel_aggregation: str = "attention"
    channel_embedding: bool = True
    fusion: str = "none"
    modality_embedding: bool = True
    fusion_heads: int = 4

    def __post_init__(self):
        if self.feature_dim < 1 or not 0 < self.patch_seconds < float("inf"):
            raise ValueError("positive feature_dim and finite patch_seconds required")
        if self.channel_aggregation not in ("none", "mean", "attention"):
            raise ValueError("channel_aggregation must be none, mean or attention")
        if self.fusion not in ("none", "pool", "cross_attention"):
            raise ValueError("fusion must be none, pool or cross_attention")
        if self.fusion != "none" and self.channel_aggregation == "none":
            raise ValueError("fusion requires channel aggregation")
        if self.fusion_heads < 1:
            raise ValueError("fusion_heads must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "BackboneConfig":
        """Recreate a JSON/YAML/checkpoint config; unknown keys raise TypeError."""
        values = dict(values)
        for key, config_type in (
            ("patch_encoder", PatchEncoderConfig),
            ("sequence_encoder", SequenceEncoderConfig),
        ):
            if key not in values:
                continue
            section = dict(values[key])
            if "blocks" in section:
                blocks = []
                for item in section["blocks"]:
                    item = dict(item)
                    if "cnn" in item:
                        item["cnn"] = CNNConfig(**item["cnn"])
                    if "transformer" in item:
                        item["transformer"] = TransformerConfig(**item["transformer"])
                    blocks.append(BlockConfig(**item))
                section["blocks"] = tuple(blocks)
            values[key] = config_type(**section)
        return cls(**values)
