
"""
IronFist1 autoregressive policy language model.

Architecture:
- Token embeddings
- Causal Transformer blocks
- RMSNorm
- Next-token prediction head
- Optional tied input/output embedding weights
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embeddings import TokenEmbeddings
from model.normalization import RMSNorm
from model.transformer import TransformerBlock


class PolicyModel(nn.Module):
    """Decoder-only Transformer for next-token prediction."""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 1280,
        num_layers: int = 20,
        num_attention_heads: int = 20,
        intermediate_size: int = 4096,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
        use_bias: bool = False,
        config=None,
    ) -> None:
        super().__init__()

        # If a configuration object is supplied, use its model dimensions.
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
            tie_word_embeddings = config.tie_word_embeddings
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
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.tie_word_embeddings = tie_word_embeddings

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

        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
        )

        if tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.embedding.weight

        self.apply(self._initialize_weights)

        # Re-establish the shared parameter after initialization.
        if tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.embedding.weight

        # Scale residual output projections to help stabilize deep models.
        residual_scale = 1.0 / math.sqrt(2 * num_layers)

        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name.endswith("attention.out_proj.weight"):
                    parameter.mul_(residual_scale)
                elif name.endswith("mlp.down_projection.weight"):
                    parameter.mul_(residual_scale)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        """Initialize linear and embedding weights."""

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """
        Args:
            input_ids: Token IDs shaped (batch, sequence_length).
            labels: Optional next-token targets of the same shape.

        Returns:
            If labels are absent:
                logits shaped (batch, sequence_length, vocab_size).
            If labels are supplied:
                (logits, loss), where loss is next-token cross-entropy.
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

        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids.")

            if labels.dtype not in (torch.int32, torch.int64):
                raise TypeError("labels must contain integer token IDs.")

            if torch.any(labels < 0) or torch.any(labels >= self.vocab_size):
                raise ValueError("labels contain IDs outside the vocabulary.")

        x = self.token_embeddings(input_ids)
        x = self.embedding_dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if labels is None:
            return logits

        # Each position predicts the target token at the same position.
        # The data loader supplies labels shifted one token ahead.
        loss = F.cross_entropy(
            logits.reshape(batch_size * sequence_length, self.vocab_size),
            labels.reshape(batch_size * sequence_length).long(),
        )

        return logits, loss

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count unique parameters, avoiding double-counting tied weights."""

        parameters = self.parameters()

        if trainable_only:
            parameters = (
                parameter
                for parameter in parameters
                if parameter.requires_grad
            )

        return sum(parameter.numel() for parameter in parameters)
