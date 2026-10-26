
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from model.discriminator_model import DiscriminatorModel
from tokenizer.tokenizer import WordPieceTokenizer


@dataclass
class DiscriminatorScoringConfig:
    """Settings for scoring candidate reasoning states."""

    max_length: int = 2048
    batch_size: int = 4
    normalize_scores: bool = False
    score_scale: float = 1.0
    score_bias: float = 0.0

    def __post_init__(self) -> None:
        if self.max_length < 1:
            raise ValueError("max_length must be at least 1")

        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if not math.isfinite(self.score_scale) or self.score_scale <= 0:
            raise ValueError("score_scale must be finite and positive")

        if not math.isfinite(self.score_bias):
            raise ValueError("score_bias must be finite")


class DiscriminatorScorer:
    """
    Score reasoning candidates with a trained DiscriminatorModel.

    The model is expected to return one scalar score per input sequence.
    Higher scores should represent higher-quality reasoning, provided
    the discriminator has been trained with that objective.

    This class handles tokenization, padding, batching, device placement,
    and conversion to Python float scores.
    """

    def __init__(
        self,
        model: DiscriminatorModel,
        tokenizer: WordPieceTokenizer,
        config: DiscriminatorScoringConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or DiscriminatorScoringConfig()

        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self._validate_tokenizer()

    def _validate_tokenizer(self) -> None:
        model_vocab_size = getattr(self.model, "vocab_size", None)

        if (
            model_vocab_size is not None
            and len(self.tokenizer) != model_vocab_size
        ):
            raise ValueError(
                f"Tokenizer vocabulary has {len(self.tokenizer)} tokens, "
                f"but the discriminator expects {model_vocab_size}"
            )

    def _encode(self, text: str) -> list[int]:
        """Encode and truncate one reasoning state."""
        token_ids = self.tokenizer.encode(text)

        if not token_ids:
            token_ids = [
                self.tokenizer.vocab[self.tokenizer.unk_token]
            ]

        return token_ids[: self.config.max_length]

    def _prepare_batch(
        self,
        texts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a padded batch and an attention mask."""
        encoded = [self._encode(text) for text in texts]
        longest = max(len(ids) for ids in encoded)

        pad_id = self.tokenizer.vocab[self.tokenizer.pad_token]

        input_ids = torch.full(
            (len(encoded), longest),
            fill_value=pad_id,
            dtype=torch.long,
            device=self.device,
        )

        attention_mask = torch.zeros(
            (len(encoded), longest),
            dtype=torch.bool,
            device=self.device,
        )

        for row, token_ids in enumerate(encoded):
            length = len(token_ids)

            input_ids[row, :length] = torch.tensor(
                token_ids,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask[row, :length] = True

        return input_ids, attention_mask

    @torch.inference_mode()
    def score_batch(self, texts: list[str]) -> list[float]:
        """
        Score a batch of reasoning states.

        Returns one scalar per input, in the same order.
        """
        if not texts:
            return []

        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Every candidate must be a string")

        self.model.eval()
        input_ids, attention_mask = self._prepare_batch(texts)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if isinstance(output, dict):
            scores = output.get("scores")
            if scores is None:
                scores = output.get("logits")
            if scores is None:
                raise ValueError(
                    "Discriminator output must contain 'scores' or 'logits'"
                )
        else:
            scores = output

        scores = torch.as_tensor(scores).reshape(-1)

        if scores.numel() != len(texts):
            raise ValueError(
                "Discriminator must return exactly one score per input"
            )

        scores = scores.float()

        if not torch.isfinite(scores).all():
            raise ValueError("Discriminator returned non-finite scores")

        scores = scores * self.config.score_scale + self.config.score_bias

        if self.config.normalize_scores:
            # This is batch-relative normalization, not a calibrated
            # probability of correctness.
            mean = scores.mean()
            std = scores.std(unbiased=False).clamp_min(1e-6)
            scores = (scores - mean) / std

        return [float(score) for score in scores.cpu().tolist()]

    def score(self, text: str) -> float:
        """Score one reasoning state."""
        return self.score_batch([text])[0]

    def __call__(self, text: str) -> float:
        """Allow direct use as the MCTS evaluator callback."""
        return self.score(text)

    def score_candidates(
        self,
        candidates: list[str],
    ) -> list[tuple[str, float]]:
        """
        Score multiple candidates and return (candidate, score) pairs.
        """
        results: list[tuple[str, float]] = []

        for start in range(0, len(candidates), self.config.batch_size):
            batch = candidates[start : start + self.config.batch_size]
            scores = self.score_batch(batch)
            results.extend(zip(batch, scores))

        return results
