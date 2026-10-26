
"""
IronFist1 normalization layers.

Implements RMSNorm for Transformer blocks.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")

        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute variance-like RMS in float32 for numerical stability.
        input_dtype = x.dtype
        x_float = x.float()

        rms = torch.sqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )

        normalized = x_float / rms
        normalized = normalized * self.weight.float()

        return normalized.to(dtype=input_dtype)
