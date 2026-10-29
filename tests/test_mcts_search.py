
"""Focused behavior tests for IronFist1's MCTS search."""

import math
import unittest
from collections.abc import Callable, Sequence

from reasoning.mcts.search import MCTSConfig, MCTSSearch

CandidateList = Sequence[tuple[str, float]]

class FakePolicy:
    def __init__(
        self,
        candidates: dict[str, CandidateList] | Callable[[str, int], CandidateList] | None = None,
    ):
        self.candidates = {} if candidates is None else candidates

    def __call__(self, state: str, count: int) -> CandidateList:
        if callable(self.candidates):
            return self.candidates(state, count)
        return self.candidates.get(state, [])



class TestMCTSSearch(unittest.TestCase):
    def test_search_returns_a_candidate(self):
        policy = FakePolicy({
            "prompt": [("prompt good", 0.9), ("prompt bad", 0.1)],
        })

        search = MCTSSearch(
            candidate_generator=policy,
            evaluator=lambda state: 1.0 if "good" in state else -1.0,
            config=MCTSConfig(num_simulations=8, max_search_depth=3),
        )

        result = search.search("prompt")

        self.assertIn(result.best_state, ("prompt good", "prompt bad"))
        self.assertGreaterEqual(result.simulations_completed, 1)

    def test_best_state_is_finite_scored(self):
        search = MCTSSearch(
            candidate_generator=FakePolicy({
                "p": [("p answer", 1.0)],
            }),
            evaluator=lambda state: 0.5,
            config=MCTSConfig(num_simulations=4),
        )

        result = search.search("p")

        self.assertTrue(math.isfinite(result.best_value))

    def test_empty_policy_leaves_root_as_best_state(self):
        search = MCTSSearch(
            candidate_generator=FakePolicy({}),
            evaluator=lambda state: 0.25,
            config=MCTSConfig(num_simulations=4),
        )

        result = search.search("prompt")

        self.assertEqual(result.best_state, "prompt")
        self.assertEqual(result.best_value, 0.25)

    def test_non_finite_evaluation_is_rejected(self):
        search = MCTSSearch(
            candidate_generator=FakePolicy({}),
            evaluator=lambda state: float("nan"),
            config=MCTSConfig(num_simulations=1),
        )

        with self.assertRaises(ValueError):
            search.search("prompt")

    def test_terminal_checker_stops_expansion(self):
        policy = FakePolicy({
            "prompt": [("prompt child", 1.0)],
            "prompt child": [("prompt grandchild", 1.0)],
        })

        search = MCTSSearch(
            candidate_generator=policy,
            evaluator=lambda state: 1.0,
            config=MCTSConfig(num_simulations=4, max_search_depth=4),
            terminal_checker=lambda state, depth: state == "prompt child",
        )

        result = search.search("prompt")

        self.assertGreaterEqual(result.simulations_completed, 1)
        child = next(iter(result.root.children.values()))
        self.assertEqual(child.state, "prompt child")
        self.assertEqual(len(child.children), 0)

    
    def test_final_selection_prefers_higher_value(self):
        policy = FakePolicy({
            "prompt": [
                ("prompt low", 0.9),
                ("prompt high", 0.1),
            ],
        })

        search = MCTSSearch(
            candidate_generator=policy,
            evaluator=lambda state: (
                10.0 if "high" in state
                else -10.0 if "low" in state
                else 0.0
            ),
            config=MCTSConfig(
                num_simulations=20,
                max_search_depth=2,
                random_seed=42,
            ),
        )

        result = search.search("prompt")

        self.assertEqual(result.best_state, "prompt high")



if __name__ == "__main__":
    unittest.main()
