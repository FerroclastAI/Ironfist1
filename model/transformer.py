
"""
IronFist1 Transformer block.

Uses pre-normalization, causal self-attention, RoPE, SwiGLU,
and residual connections.
"""

import torch
import torch.nn as nn

from model.attention import CausalSelfAttention
from model.mlp import SwiGLU
from model.normalization import RMSNorm


class TransformerBlock(nn.Module):
    """One pre-normalized Transformer block."""

    def __init__(
        self,
        hidden_size: int = 1280,
        num_attention_heads: int = 20,
        intermediate_size: int = 5120,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
        use_bias: bool = False,
    ) -> None:
        super().__init__()

        self.attention_norm = RMSNorm(
            hidden_size=hidden_size,
            eps=layer_norm_eps,
        )

        self.attention = CausalSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            use_bias=use_bias,
        )

        self.mlp_norm = RMSNorm(
            hidden_size=hidden_size,
            eps=layer_norm_eps,
        )

        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
            use_bias=use_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply attention and MLP with residual connections."""

        # Attention sublayer.
        x = x + self.attention(
            self.attention_norm(x),
            attention_mask=attention_mask,
        )

        # Feed-forward sublayer.
        x = x + self.mlp(self.mlp_norm(x))

        return x
