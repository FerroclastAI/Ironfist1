from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from reasoning.discriminator import DiscriminatorScorer
from reasoning.mcts.search import MCTSConfig, MCTSResult, MCTSSearch
from reasoning.policy import PolicyCandidateGenerator
from reasoning.reward import RewardConfig, calculate_reward

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
    """Complete result of an IronFist1 rStar reasoning run."""

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

    @property
    def diagnostics(self):
        return self.search_result.diagnostics

    @property
    def trace(self):
        return self.search_result.trace

    def to_dict(self) -> dict:
        """Export the reasoning result as a serializable dictionary."""
        return {
            "prompt": self.prompt,
            "answer": self.answer,
            "reward": self.reward,
            "discriminator_score": self.discriminator_score,
            "candidate_count": self.candidate_count,
            "simulations_completed": self.simulations_completed,
            "diagnostics": {
                "nodes_explored": self.diagnostics.nodes_explored,
                "candidates_evaluated": (
                    self.diagnostics.candidates_evaluated
                ),
                "terminal_nodes": self.diagnostics.terminal_nodes,
                "maximum_depth_reached": (
                    self.diagnostics.maximum_depth_reached
                ),
                "unique_states": self.diagnostics.unique_states,
                "average_reward": self.diagnostics.average_reward,
            },
            "trace": self.trace.to_dict(),
        }
