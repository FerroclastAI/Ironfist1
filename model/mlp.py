
"""
IronFist1 SwiGLU feed-forward network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """Gated feed-forward network using the SiLU activation."""

    def __init__(
        self,
        hidden_size: int = 1280,
        intermediate_size: int = 5120,
        dropout: float = 0.0,
        use_bias: bool = False,
    ) -> None:
        super().__init__()

        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("Layer dimensions must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.gate_projection = nn.Linear(
            hidden_size, intermediate_size, bias=use_bias
        )
        self.up_projection = nn.Linear(
            hidden_size, intermediate_size, bias=use_bias
        )
        self.down_projection = nn.Linear(
            intermediate_size, hidden_size, bias=use_bias
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shaped (batch, sequence, hidden_size).

        Returns:
            Tensor with the same shape as x.
        """
        gate = F.silu(self.gate_projection(x))
        up = self.up_projection(x)

        hidden = gate * up
        hidden = self.dropout(hidden)

        output = self.down_projection(hidden)
        return self.dropout(output)
