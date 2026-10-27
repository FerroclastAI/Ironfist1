
import unittest

from reasoning.mcts.node import MCTSNode
from reasoning.mcts.search import MCTSConfig, MCTSSearch


class TestMCTSNode(unittest.TestCase):
    def test_new_node_starts_unvisited(self):
        node = MCTSNode(state="hello")

        self.assertEqual(node.visit_count, 0)
        self.assertEqual(node.value_sum, 0.0)
        self.assertEqual(node.value, 0.0)
        self.assertFalse(node.is_expanded)

    def test_expand_creates_children(self):
        root = MCTSNode(state="root")

        children = root.expand([
            ("candidate A", 0.75),
            ("candidate B", 0.25),
        ])

        self.assertEqual(len(children), 2)
        self.assertIn("candidate A", root.children)
        self.assertIn("candidate B", root.children)
        self.assertAlmostEqual(
            sum(child.prior for child in children), 1.0
        )

    def test_backpropagation_updates_ancestors(self):
        root = MCTSNode(state="root")
        child = root.expand([("child", 1.0)])[0]
        grandchild = child.expand([("grandchild", 1.0)])[0]

        grandchild.backpropagate(0.8)

        self.assertEqual(grandchild.visit_count, 1)
        self.assertEqual(child.visit_count, 1)
        self.assertEqual(root.visit_count, 1)
        self.assertAlmostEqual(root.value, 0.8)

    def test_terminal_node_cannot_expand(self):
        node = MCTSNode(state="finished", is_terminal=True)

        with self.assertRaises(RuntimeError):
            node.expand([("another state", 1.0)])


class TestMCTSSearch(unittest.TestCase):
    def setUp(self):
        self.config = MCTSConfig(
            num_simulations=12,
            max_search_depth=4,
            max_candidates_per_node=2,
            random_seed=42,
        )

    def candidate_generator(self, state, max_candidates):
        # Each state can produce two deterministic continuations.
        return [
            (state + " A", 0.7),
            (state + " B", 0.3),
        ][:max_candidates]

    def evaluator(self, state):
        # Prefer states containing A over states containing B.
        return 1.0 if "A" in state else 0.0

    def test_search_returns_a_result(self):
        search = MCTSSearch(
            candidate_generator=self.candidate_generator,
            evaluator=self.evaluator,
            config=self.config,
        )

        result = search.search("start")

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.best_state)
        self.assertGreater(result.simulations_completed, 0)

    def test_search_respects_simulation_limit(self):
        search = MCTSSearch(
            candidate_generator=self.candidate_generator,
            evaluator=self.evaluator,
            config=self.config,
        )

        result = search.search("start")

        self.assertLessEqual(
            result.simulations_completed,
            self.config.num_simulations,
        )

    def test_terminal_checker(self):
        search = MCTSSearch(
            candidate_generator=self.candidate_generator,
            evaluator=self.evaluator,
            config=self.config,
            terminal_checker=lambda state, depth: state.endswith("A A"),
        )

        result = search.search("start")

        self.assertIsNotNone(result.best_state)

    def test_empty_candidate_list(self):
        search = MCTSSearch(
            candidate_generator=lambda state, count: [],
            evaluator=self.evaluator,
            config=self.config,
        )

        result = search.search("start")

        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
