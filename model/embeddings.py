
"""
IronFist1 embedding layers.

Provides token embeddings and rotary positional embeddings (RoPE).
"""

import torch
import torch.nn as nn


class TokenEmbeddings(nn.Module):
    """Convert token IDs into learned hidden-state vectors."""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 1280,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=padding_idx,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Integer tensor shaped (batch_size, sequence_length).

        Returns:
            Embeddings shaped (batch_size, sequence_length, hidden_size).
        """
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token IDs.")

        return self.embedding(input_ids)


class RotaryPositionalEmbedding(nn.Module):
    """Generate cosine and sine values for rotary positional encoding."""

    def __init__(
        self,
        head_dim: int,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()

        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even number.")

        if theta <= 0:
            raise ValueError("theta must be positive.")

        self.head_dim = head_dim
        self.theta = theta

        inverse_frequencies = 1.0 / (
            theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / head_dim
            )
        )

        self.register_buffer(
            "inverse_frequencies",
            inverse_frequencies,
            persistent=False,
        )

    def forward(
        self,
        sequence_length: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cosine and sine tensors shaped (sequence_length, head_dim // 2).
        """
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")

        if device is None:
            device = self.inverse_frequencies.device

        positions = torch.arange(
            sequence_length,
            device=device,
            dtype=torch.float32,
        )

        frequencies = torch.outer(
            positions,
            self.inverse_frequencies.to(device=device),
        )

        return frequencies.cos().to(dtype), frequencies.sin().to(dtype)


def apply_rotary_embeddings(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE to a tensor shaped (batch, heads, sequence, head_dim).

    The last dimension is split into even and odd coordinates.
    """
    if x.ndim != 4:
        raise ValueError("x must have shape (batch, heads, sequence, head_dim).")

    head_dim = x.shape[-1]

    if head_dim % 2 != 0:
        raise ValueError("The attention head dimension must be even.")

    expected_shape = (x.shape[-2], head_dim // 2)

    if cos.shape != expected_shape or sin.shape != expected_shape:
        raise ValueError(
            f"cos and sin must both have shape {expected_shape}."
        )

    # Do the rotation math in float32 for numerical stability.
    x_float = x.float()
    even = x_float[..., 0::2]
    odd = x_float[..., 1::2]

    cos = cos.to(device=x.device, dtype=torch.float32)[None, None, :, :]
    sin = sin.to(device=x.device, dtype=torch.float32)[None, None, :, :]

    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos

    result = torch.stack(
        (rotated_even, rotated_odd),
        dim=-1,
    ).flatten(-2)

    return result.to(dtype=x.dtype)
