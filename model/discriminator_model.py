
"""
IronFist1 discriminator model.

Scores a candidate sequence using a Transformer encoder-style
scoring head built from the same causal Transformer components
as the policy model.

The model returns one scalar score per sequence. Higher scores
can represent better candidates after suitable training.
"""

import torch
import torch.nn as nn

from model.embeddings import TokenEmbeddings
from model.normalization import RMSNorm
from model.transformer import TransformerBlock


class DiscriminatorModel(nn.Module):
    """Score candidate token sequences."""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 2048,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
        use_bias: bool = False,
        config=None,
    ) -> None:
        super().__init__()

        if config is not None:
            vocab_size = config.vocab_size
            hidden_size = config.hidden_size
            num_layers = config.num_layers
            num_attention_heads = config.num_attention_heads
            intermediate_size = config.intermediate_size
            max_seq_len = config.max_seq_len
            dropout = config.dropout
            attention_dropout = config.attention_dropout
            layer_norm_eps = config.layer_norm_eps
            use_bias = config.use_bias

        if min(
            vocab_size,
            hidden_size,
            num_layers,
            num_attention_heads,
            intermediate_size,
            max_seq_len,
        ) <= 0:
            raise ValueError("Model dimensions must be positive.")

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads."
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        self.token_embeddings = TokenEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    layer_norm_eps=layer_norm_eps,
                    use_bias=use_bias,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(
            hidden_size=hidden_size,
            eps=layer_norm_eps,
        )

        # A scalar reward-like score for each sequence.
        self.score_head = nn.Linear(hidden_size, 1, bias=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Integer token IDs shaped (batch, sequence).
            attention_mask: Optional boolean mask shaped (batch, sequence).
                True means valid token; False means padding.

        Returns:
            One raw score per sequence, shaped (batch,).
        """

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence).")

        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token IDs.")

        batch_size, sequence_length = input_ids.shape

        if sequence_length == 0 or sequence_length > self.max_seq_len:
            raise ValueError(
                f"Sequence length must be between 1 and {self.max_seq_len}."
            )

        if torch.any(input_ids < 0) or torch.any(input_ids >= self.vocab_size):
            raise ValueError("input_ids contain token IDs outside the vocabulary.")

        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "attention_mask must have the same shape as input_ids."
                )

            if attention_mask.dtype != torch.bool:
                raise TypeError("attention_mask must be a boolean tensor.")

            if attention_mask.device != input_ids.device:
                raise ValueError(
                    "attention_mask and input_ids must be on the same device."
                )

            if not torch.all(attention_mask.any(dim=1)):
                raise ValueError(
                    "Every sequence must contain at least one valid token."
                )

        x = self.token_embeddings(input_ids)
        x = self.embedding_dropout(x)

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        x = self.final_norm(x)

        if attention_mask is None:
            # Last hidden state of each sequence.
            pooled = x[:, -1, :]
        else:
            # Last valid token in each sequence.
            positions = torch.arange(
                sequence_length,
                device=input_ids.device,
            ).expand(batch_size, sequence_length)

            last_valid_positions = positions.masked_fill(
                ~attention_mask,
                -1,
            ).max(dim=1).values

            pooled = x[
                torch.arange(batch_size, device=input_ids.device),
                last_valid_positions,
            ]

        scores = self.score_head(pooled).squeeze(-1)

        return scores
