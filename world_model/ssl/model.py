"""Small per-modality CNNs, two time crops, and a shared projection head."""

from typing import NamedTuple

import torch
from torch import nn

from .config import SSLConfig


class SSLOutput(NamedTuple):
    view1: torch.Tensor  # [batch, epochs, projection_dim]
    view2: torch.Tensor
    representation: torch.Tensor  # [batch, epochs, embedding_dim]
    valid: torch.Tensor  # [batch, epochs]


class SSLModel(nn.Module):
    def __init__(self, config: SSLConfig):
        super().__init__()
        self.config = config
        self.encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv1d(channels, config.hidden_dim, 7, stride=4, padding=3),
                    nn.GELU(),
                    nn.Conv1d(
                        config.hidden_dim, config.hidden_dim, 5, stride=4, padding=2
                    ),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                for name, channels in config.channels.items()
            }
        )
        self.fusion = nn.Sequential(
            nn.Linear(len(config.channels) * config.hidden_dim, config.embedding_dim),
            nn.GELU(),
        )
        self.projector = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.projection_dim),
        )

    def encode(self, batch, *, crop_position=None):
        """Full-epoch embeddings by default; QC applies to each epoch/channel."""
        features, modality_valid = [], []
        for name, encoder in self.encoders.items():
            signal = batch["signals"][name]
            mask = batch["valid"][name] & batch["epoch_mask"].unsqueeze(-1)
            # Remove invalid values first: NaN * zero would still be NaN.
            signal = torch.where(mask.unsqueeze(-1), signal, 0.0)
            # Compress amplitudes only inside the encoder; files stay unchanged.
            signal = signal.sign() * torch.log1p(signal.abs())
            if crop_position is not None:
                length = int(signal.shape[-1] * self.config.crop_fraction)
                start = int((signal.shape[-1] - length) * crop_position)
                signal = signal[..., start : start + length]
            b, e, c, s = signal.shape
            encoded = encoder(signal.reshape(b * e, c, s)).reshape(b, e, -1)
            available = mask.any(dim=-1)
            features.append(encoded * available.unsqueeze(-1))
            modality_valid.append(available)
        valid = torch.stack(modality_valid).any(dim=0)
        representation = self.fusion(torch.cat(features, dim=-1))
        return representation * valid.unsqueeze(-1), valid

    def forward(self, batch):
        # A common relative crop position keeps modalities aligned within each view.
        positions = torch.rand(2).tolist() if self.training else (None, None)
        first, valid = self.encode(batch, crop_position=positions[0])
        second, _ = self.encode(batch, crop_position=positions[1])
        return SSLOutput(
            self.projector(first), self.projector(second), (first + second) / 2, valid
        )
