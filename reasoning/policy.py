
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model.policy_model import PolicyModel
from tokenizer.tokenizer import WordPieceTokenizer


@dataclass
class PolicySamplingConfig:
    """Controls candidate generation from the policy model."""

    max_new_tokens: int = 64
    num_candidates: int = 4
    temperature: float = 0.8
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    min_new_tokens: int = 1

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")

        if self.num_candidates < 1:
            raise ValueError("num_candidates must be at least 1")

        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")

        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")

        if (
            not math.isfinite(self.repetition_penalty)
            or self.repetition_penalty < 1.0
        ):
            raise ValueError("repetition_penalty must be at least 1")

        if self.min_new_tokens < 0:
            raise ValueError("min_new_tokens cannot be negative")

        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError(
                "min_new_tokens cannot exceed max_new_tokens"
            )


class PolicyCandidateGenerator:
    """
    Generate candidate reasoning states using a trained PolicyModel.

    The model predicts token probabilities. Candidate sequences are decoded
    and returned as complete text states for the MCTS search.

    The tokenizer must use the same vocabulary and token IDs as the model.
    """

    def __init__(
        self,
        model: PolicyModel,
        tokenizer: WordPieceTokenizer,
        config: PolicySamplingConfig | None = None,
        device: str | torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or PolicySamplingConfig()

        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.generator = torch.Generator(device=self.device)

        if seed is not None:
            self.generator.manual_seed(seed)

        self._validate_tokenizer()

    def _validate_tokenizer(self) -> None:
        vocab_size = getattr(self.model, "vocab_size", None)

        if vocab_size is not None and len(self.tokenizer) != vocab_size:
            raise ValueError(
                f"Tokenizer vocabulary has {len(self.tokenizer)} tokens, "
                f"but the model expects {vocab_size}"
            )

    @torch.inference_mode()
    def _next_token_logits(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits for the next token."""
        output = self.model(input_ids=input_ids)
        logits = output["logits"] if isinstance(output, dict) else output

        if logits.ndim != 3:
            raise ValueError(
                "Policy model must return logits shaped (batch, seq, vocab)"
            )

        return logits[:, -1, :].float()

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce the chance of repeatedly generating existing tokens."""
        penalty = self.config.repetition_penalty

        if penalty == 1.0:
            return logits

        adjusted = logits.clone()
        seen_ids = input_ids.unique()

        for token_id in seen_ids.tolist():
            if token_id < 0 or token_id >= adjusted.shape[-1]:
                continue

            score = adjusted[0, token_id]

            if score < 0:
                adjusted[0, token_id] = score * penalty
            else:
                adjusted[0, token_id] = score / penalty

        return adjusted

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        generated_count: int,
    ) -> tuple[int, float]:
        """
        Sample a token using temperature and nucleus (top-p) sampling.

        Returns:
            The sampled token ID and its log probability under the
            filtered sampling distribution.
        """
        logits = self._apply_repetition_penalty(logits, input_ids)
        logits = logits / self.config.temperature

        if generated_count < self.config.min_new_tokens:
            for token in (
                self.tokenizer.pad_token,
                self.tokenizer.cls_token,
                self.tokenizer.sep_token,
                self.tokenizer.mask_token,
            ):
                token_id = self.tokenizer.vocab.get(token)

                if token_id is not None and 0 <= token_id < logits.shape[-1]:
                    logits[0, token_id] = float("-inf")

        probabilities = F.softmax(logits, dim=-1)

        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Policy model produced invalid probabilities")

        sorted_probs, sorted_indices = torch.sort(
            probabilities,
            descending=True,
            dim=-1,
        )

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        remove = cumulative_probs > self.config.top_p

        # Keep the first token that crosses the threshold.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        filtered_probs = torch.zeros_like(probabilities)
        filtered_probs.scatter_(1, sorted_indices, sorted_probs)

        normalizer = filtered_probs.sum(dim=-1, keepdim=True)

        if not torch.isfinite(normalizer).all() or (normalizer <= 0).any():
            raise RuntimeError("No valid tokens remain for sampling")

        filtered_probs = filtered_probs / normalizer

        token_id = int(
            torch.multinomial(
                filtered_probs,
                num_samples=1,
                generator=self.generator,
            ).item()
        )

        token_probability = filtered_probs[0, token_id].clamp_min(1e-12)
        log_probability = float(torch.log(token_probability).item())

        return token_id, log_probability

    @torch.inference_mode()
    def generate_candidate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> tuple[str, float]:
        """
        Generate one candidate.

        Returns:
            (complete_generated_text, average_token_log_probability)

        The score is a log probability per generated token, not a
        discriminator reward.
        """
        limit = max_new_tokens or self.config.max_new_tokens

        if limit < 1:
            raise ValueError("max_new_tokens must be at least 1")

        self.model.eval()

        encoded = self.tokenizer.encode(prompt)
        if not encoded:
            encoded = [self.tokenizer.vocab[self.tokenizer.unk_token]]

        context_limit = getattr(self.model, "max_seq_len", 2048)
        input_ids = torch.tensor(
            [encoded[-context_limit:]],
            dtype=torch.long,
            device=self.device,
        )

        generated_ids: list[int] = []
        log_probabilities: list[float] = []

        eos_token_id = self.tokenizer.vocab.get(self.tokenizer.sep_token)

        for step in range(limit):
            context = input_ids[:, -context_limit:]
            logits = self._next_token_logits(context)

            token_id, log_probability = self._sample_next_token(
                logits,
                context,
                generated_count=step,
            )

            generated_ids.append(token_id)
            log_probabilities.append(log_probability)

            if eos_token_id is not None and token_id == eos_token_id:
                break

            next_token = torch.tensor(
                [[token_id]],
                dtype=torch.long,
                device=self.device,
            )
            input_ids = torch.cat((input_ids, next_token), dim=1)

        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        average_log_probability = (
            sum(log_probabilities) / len(log_probabilities)
            if log_probabilities
            else float("-inf")
        )

        return generated_text, average_log_probability

    def generate_candidates(
        self,
        prompt: str,
        num_candidates: int | None = None,
        max_new_tokens: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        Generate multiple complete candidate states for MCTS.

        The returned probabilities are normalized from candidate average
        log probabilities. They are useful as approximate policy priors,
        but are not calibrated confidence scores.
        """
        count = num_candidates or self.config.num_candidates

        if count < 1:
            raise ValueError("num_candidates must be at least 1")

        candidates: dict[str, float] = {}

        for _ in range(count):
            continuation, log_probability = self.generate_candidate(
                prompt,
                max_new_tokens=max_new_tokens,
            )

            # MCTS expects complete states, so include the original prompt.
            complete_state = (
                prompt.rstrip() + "\n" + continuation.lstrip()
                if continuation.strip()
                else prompt
            )

            if complete_state == prompt or complete_state in candidates:
                continue

            candidates[complete_state] = log_probability

        if not candidates:
            return []

        states = list(candidates.keys())
        scores = torch.tensor(
            [candidates[state] for state in states],
            dtype=torch.float64,
        )

        weights = torch.softmax(scores, dim=0)

        return [
            (state, float(weight))
            for state, weight in zip(states, weights.tolist())
        ]

    def __call__(
        self,
        prompt: str,
        num_candidates: int,
    ) -> list[tuple[str, float]]:
        """Allow this generator to be passed directly into MCTSSearch."""
        return self.generate_candidates(
            prompt,
            num_candidates=num_candidates,
        )
