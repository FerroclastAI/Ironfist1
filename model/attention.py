
"""
IronFist1 causal multi-head self-attention.

Includes rotary positional embeddings (RoPE) and a causal mask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embeddings import (
    RotaryPositionalEmbedding,
    apply_rotary_embeddings,
)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE."""

    def __init__(
        self,
        hidden_size: int = 1280,
        num_attention_heads: int = 20,
        attention_dropout: float = 0.0,
        use_bias: bool = False,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()

        if hidden_size <= 0 or num_attention_heads <= 0:
            raise ValueError("Hidden size and head count must be positive.")

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads."
            )

        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1).")

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.attention_dropout = attention_dropout

        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension.")

        # One projection produces queries, keys, and values.
        self.qkv_projection = nn.Linear(
            hidden_size,
            3 * hidden_size,
            bias=use_bias,
        )

        self.output_projection = nn.Linear(
            hidden_size,
            hidden_size,
            bias=use_bias,
        )

        self.rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            theta=rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Hidden states, (batch, sequence, hidden_size).
            attention_mask: Optional boolean mask shaped (batch, sequence).
                True means the token is valid; False means padding.

        Returns:
            Attention output shaped (batch, sequence, hidden_size).
        """
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, sequence, hidden_size).")

        batch_size, sequence_length, hidden_size = x.shape

        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got {hidden_size}."
            )

        if sequence_length == 0:
            raise ValueError("Sequence length must be positive.")

        # Shape: (batch, sequence, 3 * hidden_size)
        qkv = self.qkv_projection(x)

        # Shape: (batch, sequence, 3, heads, head_dim)
        qkv = qkv.reshape(
            batch_size,
            sequence_length,
            3,
            self.num_attention_heads,
            self.head_dim,
        )

        # Each becomes (batch, heads, sequence, head_dim).
        queries, keys, values = qkv.unbind(dim=2)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        cos, sin = self.rope(
            sequence_length=sequence_length,
            device=x.device,
            dtype=torch.float32,
        )

        queries = apply_rotary_embeddings(queries, cos, sin)
        keys = apply_rotary_embeddings(keys, cos, sin)

        # Boolean attention masks use True for positions that can be attended to.
        combined_mask = None

        if attention_mask is not None:
            if attention_mask.shape != (batch_size, sequence_length):
                raise ValueError(
                    "attention_mask must have shape (batch, sequence)."
                )

            if attention_mask.dtype != torch.bool:
                raise TypeError("attention_mask must be a boolean tensor.")

            if attention_mask.device != x.device:
                raise ValueError("attention_mask must be on the same device as x.")

            # Prevent attending to padding keys.
            combined_mask = attention_mask[:, None, None, :]

        # PyTorch applies causal masking so position i cannot attend to j > i.
        # The fused implementation may use Flash Attention on supported devices.
        output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=combined_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=attention_mask is None,
        )

        # Merge heads back into (batch, sequence, hidden_size).
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, sequence_length, hidden_size)

        return self.output_projection(output)
