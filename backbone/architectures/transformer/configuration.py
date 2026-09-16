"""Transformer configuration; unsupported FFN recipes fail explicitly."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    num_heads: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.0
    feed_forward: str = "dense"

    def __post_init__(self):
        if (
            type(self.num_heads) is not int
            or self.num_heads < 1
            or type(self.ffn_dim) is not int
            or self.ffn_dim < 1
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("invalid Transformer dimensions or dropout")
        if self.feed_forward != "dense":
            raise ValueError(
                "TransformerConfig defines dense FFN; configure fusion experts with model.moe"
            )
