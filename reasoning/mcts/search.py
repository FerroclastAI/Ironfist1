
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from reasoning.mcts.node import MCTSNode
from reasoning.mcts.trace import MCTSTrace


@dataclass
class MCTSConfig:
    num_simulations: int = 32
    max_search_depth: int = 8
    max_candidates_per_node: int = 4
    exploration_constant: float = 1.414
    discount_factor: float = 0.99
    random_seed: int | None = None

    def __post_init__(self):
        if self.num_simulations < 1:
            raise ValueError("num_simulations must be >= 1")
        if self.max_search_depth < 1:
            raise ValueError("max_search_depth must be >= 1")
        if self.max_candidates_per_node < 1:
            raise ValueError("max_candidates_per_node must be >= 1")
        if not math.isfinite(self.exploration_constant) or self.exploration_constant < 0:
            raise ValueError("exploration_constant must be finite and >= 0")
        if not math.isfinite(self.discount_factor) or not 0 <= self.discount_factor <= 1:
            raise ValueError("discount_factor must be between 0 and 1")


@dataclass
class MCTSDiagnostics:
    nodes_explored: int = 0
    candidates_evaluated: int = 0
    terminal_nodes: int = 0
    maximum_depth_reached: int = 0
    unique_states: int = 0
    average_reward: float = 0.0


@dataclass
class MCTSResult:
    root: MCTSNode
    best_state: str
    best_value: float
    simulations_completed: int
    diagnostics: MCTSDiagnostics
    trace: MCTSTrace

    @property
    def best_path(self) -> list[str]:
        path = [self.root.state]
        node = self.root

        while node.most_visited_child is not None:
            node = node.most_visited_child
            path.append(node.state)

        return path


class MCTSSearch:
    def __init__(
        self,
        candidate_generator,
        evaluator,
        config: MCTSConfig | None = None,
        terminal_checker=None,
    ):
        self.candidate_generator = candidate_generator
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self.terminal_checker = terminal_checker or (lambda state, depth: False)

        self.rng = random.Random(self.config.random_seed)

    def _evaluate(self, state: str) -> float:
        reward = float(self.evaluator(state))

        if not math.isfinite(reward):
            raise ValueError("Evaluator returned a non-finite reward")

        return reward

    def _is_terminal(self, state: str, depth: int) -> bool:
        return bool(self.terminal_checker(state, depth))

    def _generate_candidates(self, node: MCTSNode) -> list[tuple[str, float]]:
        candidates = self.candidate_generator(
            node.state,
            self.config.max_candidates_per_node,
        )

        if candidates is None:
            return []

        validated = []
        seen = set()

        for candidate in candidates:
            if not isinstance(candidate, (tuple, list)) or len(candidate) != 2:
                raise ValueError("Candidates must be (state, prior) pairs")

            state, prior = candidate

            if not isinstance(state, str):
                raise TypeError("Candidate state must be a string")

            prior = float(prior)

            if not math.isfinite(prior) or prior < 0:
                raise ValueError("Candidate prior must be finite and >= 0")

            if state == node.state or state in seen:
                continue

            seen.add(state)
            validated.append((state, prior))

            if len(validated) >= self.config.max_candidates_per_node:
                break

        return validated

    def _depth(self, node: MCTSNode) -> int:
        depth = 0
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth

    def _select_leaf(self, root: MCTSNode) -> MCTSNode:
        node = root

        while node.is_expanded and not node.is_terminal:
            if self._depth(node) >= self.config.max_search_depth:
                break

            child = node.best_child(self.config.exploration_constant)

            if child is None:
                break

            node = child

        return node

    def _collect_nodes(self, root: MCTSNode) -> list[MCTSNode]:
        nodes = []
        stack = [root]

        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children.values())

        return nodes

    def search(self, initial_state: str) -> MCTSResult:
        if not isinstance(initial_state, str):
            raise TypeError("initial_state must be a string")

        trace = MCTSTrace()
        evaluated_rewards = []
        evaluated_states = set()
        candidates_evaluated = 0
        terminal_nodes = 0
        maximum_depth = 0

        root = MCTSNode(state=initial_state)
        root_reward = self._evaluate(initial_state)

        evaluated_rewards.append(root_reward)
        evaluated_states.add(initial_state)
        candidates_evaluated += 1
        root.update(root_reward)

        trace.record(
            state=initial_state,
            depth=0,
            event="root_evaluated",
            reward=root_reward,
        )

        simulations_completed = 0

        for _ in range(self.config.num_simulations):
            leaf = self._select_leaf(root)
            depth = self._depth(leaf)
            maximum_depth = max(maximum_depth, depth)

            if leaf.is_terminal:
                terminal_nodes += 1
                reward = self._evaluate(leaf.state)
                evaluated_rewards.append(reward)
                candidates_evaluated += 1
                leaf.backpropagate(reward)

                trace.record(
                    state=leaf.state,
                    depth=depth,
                    event="terminal_evaluated",
                    reward=reward,
                )

                simulations_completed += 1
                continue

            if depth >= self.config.max_search_depth:
                reward = self._evaluate(leaf.state)
                evaluated_rewards.append(reward)
                candidates_evaluated += 1
                leaf.backpropagate(reward)

                trace.record(
                    state=leaf.state,
                    depth=depth,
                    event="depth_limit",
                    reward=reward,
                )

                simulations_completed += 1
                continue

            candidates = self._generate_candidates(leaf)

            if not candidates:
                reward = self._evaluate(leaf.state)
                evaluated_rewards.append(reward)
                candidates_evaluated += 1
                leaf.is_terminal = True
                terminal_nodes += 1
                leaf.backpropagate(reward)

                trace.record(
                    state=leaf.state,
                    depth=depth,
                    event="no_candidates",
                    reward=reward,
                )

                simulations_completed += 1
                continue

            children = leaf.expand(candidates)

            for child in children:
                trace.record(
                    state=child.state,
                    depth=depth + 1,
                    event="candidate_generated",
                    prior=child.prior,
                )

            unvisited = [
                child for child in children
                if child.visit_count == 0
            ]

            if unvisited:
                child = max(unvisited, key=lambda item: item.prior)
            else:
                child = leaf.best_child(self.config.exploration_constant)

            if child is None:
                simulations_completed += 1
                continue

            child_depth = depth + 1
            maximum_depth = max(maximum_depth, child_depth)

            child.is_terminal = self._is_terminal(
                child.state,
                child_depth,
            )

            if child.is_terminal:
                terminal_nodes += 1

            reward = self._evaluate(child.state)

            evaluated_rewards.append(reward)
            evaluated_states.add(child.state)
            candidates_evaluated += 1

            discounted_reward = (
                reward * self.config.discount_factor ** child_depth
            )

            child.backpropagate(discounted_reward)

            trace.record(
                state=child.state,
                depth=child_depth,
                event="candidate_evaluated",
                reward=reward,
                prior=child.prior,
            )

            simulations_completed += 1

        if root.children:
            best_child = max(
                root.children.values(),
                key=lambda child: (child.value, child.visit_count),
            )
            best_state = best_child.state
            best_value = best_child.value
        else:
            best_state = initial_state
            best_value = root.value

        nodes = self._collect_nodes(root)

        trace.mark_selected(best_state)

        diagnostics = MCTSDiagnostics(
            nodes_explored=len(nodes),
            candidates_evaluated=candidates_evaluated,
            terminal_nodes=terminal_nodes,
            maximum_depth_reached=maximum_depth,
            unique_states=len(evaluated_states),
            average_reward=(
                sum(evaluated_rewards) / len(evaluated_rewards)
                if evaluated_rewards else 0.0
            ),
        )

        return MCTSResult(
            root=root,
            best_state=best_state,
            best_value=best_value,
            simulations_completed=simulations_completed,
            diagnostics=diagnostics,
            trace=trace,
        )
