
from __future__ import annotations

import math
from dataclasses import dataclass

from reasoning.discriminator import DiscriminatorScorer
from reasoning.mcts.search import MCTSConfig, MCTSResult, MCTSSearch
from reasoning.policy import PolicyCandidateGenerator
from reasoning.reward import RewardConfig, calculate_reward
from functools import lru_cache

@dataclass
class RStarConfig:
    """High-level configuration for IronFist1's reasoning system."""

    num_simulations: int = 32
    max_search_depth: int = 8
    max_candidates_per_node: int = 4
    exploration_constant: float = 1.414
    discount_factor: float = 0.99
    discriminator_weight: float = 1.0
    consistency_weight: float = 0.5
    penalty_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.num_simulations < 1:
            raise ValueError("num_simulations must be at least 1")

        if self.max_search_depth < 1:
            raise ValueError("max_search_depth must be at least 1")

        if self.max_candidates_per_node < 1:
            raise ValueError("max_candidates_per_node must be at least 1")

        if (
            not math.isfinite(self.exploration_constant)
            or self.exploration_constant < 0
        ):
            raise ValueError(
                "exploration_constant must be finite and non-negative"
            )

        if (
            not math.isfinite(self.discount_factor)
            or not 0 <= self.discount_factor <= 1
        ):
            raise ValueError("discount_factor must be between 0 and 1")

        for name, value in (
            ("discriminator_weight", self.discriminator_weight),
            ("consistency_weight", self.consistency_weight),
            ("penalty_weight", self.penalty_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass
class RStarResult:
    """Result returned by an rStar reasoning run."""

    prompt: str
    answer: str
    reward: float
    discriminator_score: float
    search_result: MCTSResult

    @property
    def candidate_count(self) -> int:
        return len(self.search_result.root.children)

    @property
    def simulations_completed(self) -> int:
        return self.search_result.simulations_completed


class RStarReasoner:
    """
    Coordinate policy-based candidate generation, MCTS, and scoring.

    The policy generator proposes candidate states. The discriminator
    estimates their quality. MCTS searches the resulting tree.

    This implementation is an inference-time search framework. It does
    not train either neural network.
    """

    def __init__(
        self,
        policy_generator: PolicyCandidateGenerator,
        discriminator: DiscriminatorScorer,
        config: RStarConfig | None = None,
        consistency_checker=None,
        penalty_checker=None,
    ) -> None:
        self.policy_generator = policy_generator
        self.discriminator = discriminator
        self.config = config or RStarConfig()

        self.consistency_checker = consistency_checker
        self.penalty_checker = penalty_checker

        self.reward_config = RewardConfig(
            discriminator_weight=self.config.discriminator_weight,
            consistency_weight=self.config.consistency_weight,
            penalty_weight=self.config.penalty_weight,
        )

        self.search_config = MCTSConfig(
            num_simulations=self.config.num_simulations,
            max_search_depth=self.config.max_search_depth,
            max_candidates_per_node=self.config.max_candidates_per_node,
            exploration_constant=self.config.exploration_constant,
            discount_factor=self.config.discount_factor,
        )

    def _consistency_score(self, state: str) -> float:
        """Get an optional consistency score, defaulting to zero."""
        if self.consistency_checker is None:
            return 0.0

        score = float(self.consistency_checker(state))

        if not math.isfinite(score):
            raise ValueError("Consistency checker returned a non-finite score")

        return score

    def _penalty(self, state: str) -> float:
        """Get an optional penalty, defaulting to zero."""
        if self.penalty_checker is None:
            return 0.0

        penalty = float(self.penalty_checker(state))

        if not math.isfinite(penalty) or penalty < 0:
            raise ValueError(
                "Penalty checker must return a finite, non-negative value"
            )

        return penalty


    def _reward(self, state: str) -> tuple[float, float]:
        """Calculate a validated reward for a candidate state."""
        discriminator_score = float(self.discriminator.score(state))

        if not math.isfinite(discriminator_score):
            raise ValueError(
                "Discriminator returned a non-finite score"
            )

        consistency = self._consistency_score(state)
        penalty = self._penalty(state)

        reward = calculate_reward(
            discriminator_score=discriminator_score,
            consistency_score=consistency,
            penalty=penalty,
            config=self.reward_config,
        )

        reward = float(reward)

        if not math.isfinite(reward):
            raise ValueError("Calculated reward must be finite")

        return reward, discriminator_score

  
    def reason(self, prompt: str) -> RStarResult:
        """
        Search for a high-scoring reasoning continuation.

        Cache evaluations so repeated visits to the same state do not
        repeatedly invoke the discriminator and optional checkers.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        @lru_cache(maxsize=4096)
        def evaluate_details(state: str) -> tuple[float, float]:
            reward, discriminator_score = self._reward(state)
            return reward, discriminator_score

        def evaluate(state: str) -> float:
            return evaluate_details(state)[0]

        search = MCTSSearch(
            candidate_generator=self.policy_generator,
            evaluator=evaluate,
            config=self.search_config,
        )

        search_result = search.search(prompt)
        selected_state = search_result.best_state

        if selected_state.startswith(prompt):
            answer = selected_state[len(prompt):].strip()
        else:
            answer = selected_state.strip()

        if not answer:
            answer = selected_state

        final_reward, final_discriminator_score = evaluate_details(
            selected_state
        )

        return RStarResult(
            prompt=prompt,
            answer=answer,
            reward=final_reward,
            discriminator_score=final_discriminator_score,
            search_result=search_result,
        )


    def __call__(self, prompt: str) -> str:
        """Return just the selected answer for simple chat integration."""
        return self.reason(prompt).answer
