
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embeddings import (
    RotaryPositionalEmbedding,
    apply_rotary_embeddings,
)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with rotary positional embeddings.

    Supports:
      - Fused QKV projections.
      - Rotary positional embeddings (RoPE).
      - PyTorch scaled dot-product attention.
      - Optional key-padding masks.
      - Causal masking with or without padding.

    Attention masks use True for valid tokens and False for padding.
    """

    def __init__(
        self,
        hidden_size: int = 1280,
        num_attention_heads: int = 20,
        attention_dropout: float = 0.0,
        use_bias: bool = False,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        if num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )

        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError(
                "attention_dropout must be in the interval [0, 1)"
            )

        if not math.isfinite(rope_theta) or rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.attention_dropout = attention_dropout

        if self.head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even for rotary positional embeddings"
            )

        self.qkv_proj = nn.Linear(
            hidden_size,
            3 * hidden_size,
            bias=use_bias,
        )

        self.out_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=use_bias,
        )

        self.rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            theta=rope_theta,
        )

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        seq_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert a padding mask into a broadcastable key-validity mask.

        Accepted shape: (batch, sequence).
        True/nonzero means valid; False/zero means padding.
        """
        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask must have shape (batch, sequence)"
            )

        if attention_mask.shape != (batch_size, seq_length):
            raise ValueError(
                "attention_mask shape must match input batch and sequence"
            )

        mask = attention_mask.to(device=device, dtype=torch.bool)

        # Shape: (batch, 1, 1, sequence)
        # This masks keys that correspond to padding.
        return mask[:, None, None, :]

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Hidden states shaped (batch, sequence, hidden_size).
            attention_mask: Optional (batch, sequence) padding mask.
                True or 1 means a real token; False or 0 means padding.

        Returns:
            Hidden states shaped (batch, sequence, hidden_size).
        """
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (batch, sequence, hidden_size)"
            )

        batch_size, seq_length, hidden_size = x.shape

        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, "
                f"received {hidden_size}"
            )

        if seq_length == 0:
            raise ValueError("Sequence length must be greater than zero")

        # Project to Q, K, and V.
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(
            batch_size,
            seq_length,
            3,
            self.num_attention_heads,
            self.head_dim,
        )

        # Each tensor: (batch, heads, sequence, head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rope(
            seq_length,
            device=x.device,
            dtype=q.dtype,
        )

        q = apply_rotary_embeddings(q, cos, sin)
        k = apply_rotary_embeddings(k, cos, sin)

        combined_mask = None

        if attention_mask is not None:
            key_mask = self._prepare_attention_mask(
                attention_mask,
                batch_size,
                seq_length,
                x.device,
            )

            # A True value means attention is allowed.
            causal_mask = torch.ones(
                (seq_length, seq_length),
                dtype=torch.bool,
                device=x.device,
            ).tril()

            # Shape: (batch, 1, sequence, sequence)
            combined_mask = (
                causal_mask[None, None, :, :]
                & key_mask
            )

            is_causal = False
        else:
            # PyTorch applies the causal mask directly when no explicit
            # mask is provided.
            is_causal = True

        dropout_p = (
            self.attention_dropout if self.training else 0.0
        )

        attention_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=combined_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        # (batch, heads, sequence, head_dim)
        # -> (batch, sequence, hidden_size)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(
            batch_size,
            seq_length,
            self.hidden_size,
        )

        return self.out_proj(attention_output)
