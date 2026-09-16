"""Batched SDPA attention and dense FFN used by Criss-Cross/Temporal blocks."""

from torch import nn
from torch.nn import functional as F

from ..sequence import clean
from .configuration import TransformerConfig


class Attention(nn.Module):
    def __init__(self, dim: int, config: TransformerConfig):
        super().__init__()
        if dim % config.num_heads:
            raise ValueError("attention dimension must be divisible by num_heads")
        self.heads = config.num_heads
        self.dropout = config.dropout
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, query, source, allowed):
        def split(value):
            b, t, d = value.shape
            return value.reshape(b, t, self.heads, d // self.heads).transpose(1, 2)

        # Give empty rows one harmless key, then explicitly zero their outputs.
        # This avoids NaNs on backends that do not handle all-masked softmax rows.
        has_source = allowed.any(-1)
        safe = allowed.clone()
        safe[..., 0] |= ~has_source
        value = F.scaled_dot_product_attention(
            split(self.query(query)),
            split(self.key(source)),
            split(self.value(source)),
            attn_mask=safe[:, None],
            dropout_p=self.dropout if self.training else 0.0,
        )
        value = value.transpose(1, 2).reshape_as(query)
        return clean(self.output(value), has_source)


class DenseFFN(nn.Module):
    def __init__(self, dim: int, config: TransformerConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, dim),
        )

    def forward(self, tokens):
        return self.net(tokens)
