
from __future__ import annotations

import math
from typing import Iterable


class MCTSNode:
    """
    A node in a single-agent Monte Carlo Tree Search.

    Each node represents a complete reasoning state. Child candidates
    should also be complete resulting states, not merely text fragments.

    Attributes:
        state: The complete reasoning state represented by this node.
        parent: The parent node, or None for the root.
        action: The candidate state that led to this node.
        prior: The policy model's prior probability for this candidate.
        is_terminal: Whether this state ends the search path.
        visit_count: Number of times this node has been visited.
        value_sum: Sum of rewards backed up through this node.
        children: Child nodes indexed by their candidate state.
    """

    def __init__(
        self,
        state: str,
        parent: MCTSNode | None = None,
        action: str | None = None,
        prior: float = 1.0,
        is_terminal: bool = False,
    ) -> None:
        if not isinstance(state, str):
            raise TypeError("state must be a string")

        if not math.isfinite(prior) or prior < 0.0:
            raise ValueError("prior must be finite and non-negative")

        self.state = state
        self.parent = parent
        self.action = action
        self.prior = float(prior)
        self.is_terminal = is_terminal

        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[str, MCTSNode] = {}

    @property
    def value(self) -> float:
        """Return the average backed-up value, or zero if unvisited."""
        if self.visit_count == 0:
            return 0.0

        return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        """Whether this node has child candidates."""
        return bool(self.children)

    @property
    def is_root(self) -> bool:
        """Whether this node is the root of its search tree."""
        return self.parent is None

    def expand(
        self,
        candidates: Iterable[tuple[str, float]],
    ) -> list[MCTSNode]:
        """
        Add candidate states as children.

        Args:
            candidates: An iterable of (complete_state, prior_probability)
                pairs. Prior probabilities are normalized to sum to one.

        Returns:
            The child nodes, including existing children for duplicate states.

        Notes:
            Existing children are not overwritten. A terminal node cannot
            be expanded.
        """
        if self.is_terminal:
            raise RuntimeError("A terminal node cannot be expanded")

        candidate_list = list(candidates)

        if not candidate_list:
            return list(self.children.values())

        for candidate_state, prior in candidate_list:
            if not isinstance(candidate_state, str):
                raise TypeError("Each candidate state must be a string")
            if not math.isfinite(prior) or prior < 0.0:
                raise ValueError(
                    "Candidate prior probabilities must be finite "
                    "and non-negative"
                )

        total_prior = sum(prior for _, prior in candidate_list)

        if total_prior <= 0.0:
            normalized_priors = [
                (state, 1.0 / len(candidate_list))
                for state, _ in candidate_list
            ]
        else:
            normalized_priors = [
                (state, prior / total_prior)
                for state, prior in candidate_list
            ]

        for candidate_state, prior in normalized_priors:
            if candidate_state in self.children:
                continue

            self.children[candidate_state] = MCTSNode(
                state=candidate_state,
                parent=self,
                action=candidate_state,
                prior=prior,
            )

        return list(self.children.values())

    def update(self, reward: float) -> None:
        """
        Add one simulation result to this node's statistics.

        Call this once for each node on the path being backed up.
        This method does not automatically update parent nodes.
        """
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")

        self.visit_count += 1
        self.value_sum += float(reward)

    def ucb_score(self, exploration_constant: float = 1.414) -> float:
        """
        Calculate a PUCT-style selection score.

        The exploitation term favors high-value candidates. The exploration
        term favors candidates with larger policy priors and fewer visits.
        """
        if not math.isfinite(exploration_constant):
            raise ValueError("exploration_constant must be finite")

        if exploration_constant < 0.0:
            raise ValueError("exploration_constant cannot be negative")

        exploitation = self.value

        if self.parent is None:
            return exploitation

        parent_visits = self.parent.visit_count

        exploration = (
            exploration_constant
            * self.prior
            * math.sqrt(parent_visits)
            / (1 + self.visit_count)
        )

        return exploitation + exploration

    def best_child(
        self,
        exploration_constant: float = 1.414,
    ) -> MCTSNode | None:
        """
        Select the child with the highest PUCT score.

        Returns None if this node has no children.
        """
        if not self.children:
            return None

        return max(
            self.children.values(),
            key=lambda child: child.ucb_score(exploration_constant),
        )

    def most_visited_child(self) -> MCTSNode | None:
        """Select the child visited most often."""
        if not self.children:
            return None

        return max(
            self.children.values(),
            key=lambda child: child.visit_count,
        )

    def backpropagate(self, reward: float) -> None:
        """
        Update this node and every ancestor with the same reward.

        This is suitable for the current single-agent reasoning search.
        More complex reward perspectives can be implemented in search.py.
        """
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")

        node: MCTSNode | None = self

        while node is not None:
            node.update(reward)
            node = node.parent

    def __repr__(self) -> str:
        return (
            f"MCTSNode(visits={self.visit_count}, "
            f"value={self.value:.4f}, "
            f"prior={self.prior:.4f}, "
            f"children={len(self.children)}, "
            f"terminal={self.is_terminal})"
        )
