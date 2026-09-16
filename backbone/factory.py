"""Build the single backbone from CPU metadata or a saved input specification."""

from dataloader.signals import SignalBatch

from .configuration import BackboneConfig
from .modeling.foundation import FoundationBackbone


def build_backbone(
    config: BackboneConfig, example: SignalBatch | dict
) -> FoundationBackbone:
    """The adapter has already validated input tensors; only construct modules here."""
    if isinstance(example, SignalBatch):
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
    return FoundationBackbone(config, spec)
