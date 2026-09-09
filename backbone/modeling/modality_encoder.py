"""One modality: signal encoder followed by optional channel aggregation."""

from torch import nn

from ..contracts import ModalityOutput


class ModalityEncoder(nn.Module):
    def __init__(self, signal_encoder, channel_aggregator):
        super().__init__()
        self.signal_encoder = signal_encoder
        self.channel_aggregator = channel_aggregator

    def forward(self, group, batch, stage, visible, need_local, need_features):
        result = self.signal_encoder(group, batch, stage, visible, need_local)
        features = self.channel_aggregator(result.local) if need_features else None
        return ModalityOutput(result.patch_tokens, result.local, features)
