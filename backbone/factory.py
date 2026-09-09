"""Construction-time selection; forward paths receive already built modules."""

from dataloader.signals import SignalBatch

from .architectures.cnn.modeling import CNNSequenceBlock
from .architectures.transformer.configuration import TransformerConfig
from .architectures.transformer.modeling import (
    CrossAttentionBlock,
    TransformerSequenceBlock,
)
from .configuration import BackboneConfig
from .modeling.channel_aggregator import ChannelAggregator
from .modeling.fusion import Fusion
from .modeling.modality_encoder import ModalityEncoder
from .modeling.patch_encoder import PatchEncoder
from .modeling.patch_sequence_encoder import PatchSequenceEncoder
from .modeling.patching import Patchifier
from .modeling.psg_backbone import PSGBackbone
from .modeling.signal_encoder import SignalEncoder


def build_blocks(configs, dim):
    result = []
    for config in configs:
        for _ in range(config.depth):
            if config.kind == "cnn":
                result.append(CNNSequenceBlock(dim, config.cnn))
            else:
                result.append(TransformerSequenceBlock(dim, config.transformer))
    return result


def build_backbone(config: BackboneConfig, example: SignalBatch | dict) -> PSGBackbone:
    """Build on CPU from an adapted batch or a saved, plain-Python input_spec."""
    if isinstance(example, SignalBatch):
        example.validate()
        spec = {
            name: {
                "channel_ids": tuple(group.channel_ids),
                "units": tuple(group.units),
                "sample_rate_hz": float(group.sample_rate_hz[0]),
                "epoch_samples": group.values.shape[-1],
            }
            for name, group in example.groups.items()
        }
    else:
        spec = example
    if not spec:
        raise ValueError("at least one encoding group required")
    encoders = {}
    dim = config.feature_dim
    for name, info in spec.items():
        if "." in name or not name:
            raise ValueError(
                "encoding group names must be nonempty and contain no dots"
            )
        samples = config.patch_seconds * info["sample_rate_hz"]
        if samples < 1 or abs(samples - round(samples)) > 1e-6:
            raise ValueError("patch_seconds must map to an integer sample count")
        samples = round(samples)
        if info["epoch_samples"] % samples:
            raise ValueError("epoch samples must be divisible by patch samples")
        epoch_seconds = info["epoch_samples"] / info["sample_rate_hz"]
        if config.sequence_encoder.window_seconds > epoch_seconds:
            raise ValueError("first version limits context windows to one epoch")
        patch_cfg = config.patch_encoder
        blocks = (
            build_blocks(patch_cfg.blocks, patch_cfg.hidden_dim)
            if patch_cfg.variant == "sequence"
            else []
        )
        signal = SignalEncoder(
            Patchifier(samples, info["sample_rate_hz"]),
            PatchEncoder(samples, dim, patch_cfg, blocks),
            PatchSequenceEncoder(
                dim,
                config.patch_seconds,
                config.sequence_encoder,
                build_blocks(config.sequence_encoder.blocks, dim),
            ),
        )
        aggregator = (
            None
            if config.channel_aggregation == "none"
            else ChannelAggregator(
                len(info["channel_ids"]),
                dim,
                config.channel_aggregation,
                config.channel_embedding,
            )
        )
        encoders[name] = ModalityEncoder(signal, aggregator)
    fusion = None
    if config.fusion != "none":
        cross = (
            CrossAttentionBlock(dim, TransformerConfig(num_heads=config.fusion_heads))
            if config.fusion == "cross_attention"
            else None
        )
        fusion = Fusion(tuple(spec), dim, config.modality_embedding, cross)
    return PSGBackbone(config, spec, encoders, fusion)
