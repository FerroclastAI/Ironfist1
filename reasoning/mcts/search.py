
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence

from reasoning.mcts.node import MCTSNode


# Generates complete candidate states and their policy probabilities.
CandidateGenerator = Callable[
    [str, int],
    Sequence[tuple[str, float]],
]

# Evaluates a complete state and returns a scalar reward.
StateEvaluator = Callable[[str], float]

# Determines whether a state should terminate the search.
TerminalChecker = Callable[[str, int], bool]


@dataclass
class MCTSConfig:
    """Settings for single-agent reasoning search."""

    num_simulations: int = 32
    max_search_depth: int = 8
    max_candidates_per_node: int = 4
    exploration_constant: float = 1.414
    discount_factor: float = 0.99
    random_seed: int | None = None

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


@dataclass
class MCTSResult:
    """Summary of a completed MCTS run."""

    root: MCTSNode
    best_state: str
    best_value: float
    simulations_completed: int

    @property
    def best_path(self) -> list[str]:
        """Return the states along the most-visited path."""
        path: list[str] = []
        node = self.root

        while node.children:
            child = node.most_visited_child()
            if child is None:
                break

            path.append(child.state)
            node = child

            if node.is_terminal:
                break

        return path


class MCTSSearch:
    """
    Monte Carlo Tree Search for single-agent reasoning.

    The caller supplies:
      - candidate_generator: produces complete next-state candidates.
      - evaluator: scores the quality of a complete state.
      - terminal_checker: optionally detects completed reasoning paths.

    The evaluator should return higher values for better reasoning states.
    """

    def __init__(
        self,
        candidate_generator: CandidateGenerator,
        evaluator: StateEvaluator,
        config: MCTSConfig | None = None,
        terminal_checker: TerminalChecker | None = None,
    ) -> None:
        self.candidate_generator = candidate_generator
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self.terminal_checker = terminal_checker
        self.rng = random.Random(self.config.random_seed)

    def _is_terminal(self, state: str, depth: int) -> bool:
        if depth >= self.config.max_search_depth:
            return True

        if self.terminal_checker is not None:
            return bool(self.terminal_checker(state, depth))

        return False

    def _generate_candidates(
        self,
        node: MCTSNode,
        depth: int,
    ) -> list[tuple[str, float]]:
        """Generate and validate the candidates for a node."""
        candidates = self.candidate_generator(
            node.state,
            self.config.max_candidates_per_node,
        )

        validated: list[tuple[str, float]] = []
        seen: set[str] = set()

        for state, prior in candidates:
            if not isinstance(state, str):
                raise TypeError("Candidate states must be strings")

            if not math.isfinite(prior) or prior < 0:
                raise ValueError(
                    "Candidate probabilities must be finite "
                    "and non-negative"
                )

            if state in seen:
                continue

            seen.add(state)

            # A state identical to its parent would create a pointless loop.
            if state == node.state:
                continue

            validated.append((state, float(prior)))

            if len(validated) >= self.config.max_candidates_per_node:
                break

        return validated

    def _evaluate(self, state: str) -> float:
        """Evaluate a state and reject invalid reward values."""
        value = float(self.evaluator(state))

        if not math.isfinite(value):
            raise ValueError("The evaluator returned a non-finite reward")

        return value

    def _select_leaf(self, root: MCTSNode) -> tuple[MCTSNode, int]:
        """Follow the best-child policy until reaching a leaf."""
        node = root
        depth = 0

        while node.is_expanded and not node.is_terminal:
            if depth >= self.config.max_search_depth:
                break

            child = node.best_child(
                self.config.exploration_constant
            )

            if child is None:
                break

            node = child
            depth += 1

        return node, depth

    def search(self, initial_state: str) -> MCTSResult:
        """
        Run MCTS from an initial prompt or reasoning state.

        The root's value is evaluated first. Each simulation then selects
        a leaf, expands it if possible, evaluates a resulting state, and
        backs up the discounted reward through the selected path.
        """
        if not isinstance(initial_state, str):
            raise TypeError("initial_state must be a string")

        root = MCTSNode(state=initial_state)
        root_reward = self._evaluate(initial_state)
        root.update(root_reward)

        simulations_completed = 0

        for _ in range(self.config.num_simulations):
            leaf, depth = self._select_leaf(root)

            if leaf.is_terminal:
                leaf_reward = self._evaluate(leaf.state)
                leaf.backpropagate(leaf_reward)
                simulations_completed += 1
                continue

            if depth >= self.config.max_search_depth:
                leaf.is_terminal = True
                leaf_reward = self._evaluate(leaf.state)
                leaf.backpropagate(leaf_reward)
                simulations_completed += 1
                continue

            candidates = self._generate_candidates(leaf, depth)

            if not candidates:
                leaf.is_terminal = True
                leaf_reward = self._evaluate(leaf.state)
                leaf.backpropagate(leaf_reward)
                simulations_completed += 1
                continue

            children = leaf.expand(candidates)

            # Prefer children created by this expansion. If all candidates
            # already existed, select among the existing children instead.
            unvisited = [child for child in children if child.visit_count == 0]

            if unvisited:
                child = max(
                    unvisited,
                    key=lambda item: item.prior
                    + self.rng.random() * 1e-12,
                )
            else:
                child = leaf.best_child(
                    self.config.exploration_constant
                )

            if child is None:
                leaf.is_terminal = True
                leaf.backpropagate(self._evaluate(leaf.state))
                simulations_completed += 1
                continue

            child_depth = depth + 1
            child.is_terminal = self._is_terminal(
                child.state,
                child_depth,
            )

            child_reward = self._evaluate(child.state)

            # Apply discount according to how far this candidate is from
            # the root. All ancestors receive the same discounted outcome.
            discounted_reward = (
                child_reward
                * self.config.discount_factor ** child_depth
            )

            child.backpropagate(discounted_reward)
            simulations_completed += 1

        # Select the most-visited root child as the final candidate.
        best_child = root.most_visited_child()

        if best_child is None:
            best_state = initial_state
            best_value = root.value
        else:
            best_state = best_child.state
            best_value = best_child.value

        return MCTSResult(
            root=root,
            best_state=best_state,
            best_value=best_value,
            simulations_completed=simulations_completed,
        )
