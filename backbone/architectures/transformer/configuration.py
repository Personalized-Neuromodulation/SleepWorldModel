"""Transformer configuration; unsupported FFN recipes fail explicitly."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    num_heads: int = 4
    expansion: int = 4
    dropout: float = 0.0
    feed_forward: str = "dense"

    def __post_init__(self):
        if self.num_heads < 1 or self.expansion < 1 or not 0 <= self.dropout < 1:
            raise ValueError("invalid Transformer dimensions or dropout")
        if self.feed_forward != "dense":
            raise ValueError("only dense FFN is implemented; MoE is a future extension")
