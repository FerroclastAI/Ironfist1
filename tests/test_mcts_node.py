
"""Unit tests for IronFist1's MCTSNode."""

import math
import unittest

from reasoning.mcts.node import MCTSNode


class TestMCTSNode(unittest.TestCase):
    def test_new_node_has_zero_statistics(self):
        node = MCTSNode("root")

        self.assertEqual(node.visit_count, 0)
        self.assertEqual(node.value_sum, 0.0)
        self.assertEqual(node.value, 0.0)
        self.assertTrue(node.is_root)
        self.assertFalse(node.is_expanded)

    def test_update_tracks_average_value(self):
        node = MCTSNode("state")

        node.update(2.0)
        node.update(4.0)

        self.assertEqual(node.visit_count, 2)
        self.assertAlmostEqual(node.value_sum, 6.0)
        self.assertAlmostEqual(node.value, 3.0)

    def test_update_rejects_non_finite_reward(self):
        node = MCTSNode("state")

        for reward in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(reward=reward):
                with self.assertRaises(ValueError):
                    node.update(reward)

    def test_expand_normalizes_priors(self):
        root = MCTSNode("root")
        children = root.expand([
            ("first", 3.0),
            ("second", 1.0),
        ])

        self.assertEqual(len(children), 2)
        self.assertAlmostEqual(root.children["first"].prior, 0.75)
        self.assertAlmostEqual(root.children["second"].prior, 0.25)
        self.assertAlmostEqual(
            sum(child.prior for child in children),
            1.0,
        )

    def test_zero_priors_are_distributed_equally(self):
        root = MCTSNode("root")
        root.expand([("first", 0.0), ("second", 0.0)])

        self.assertAlmostEqual(root.children["first"].prior, 0.5)
        self.assertAlmostEqual(root.children["second"].prior, 0.5)

    def test_duplicate_child_is_not_overwritten(self):
        root = MCTSNode("root")
        root.expand([("child", 1.0)])

        child = root.children["child"]
        child.update(5.0)

        root.expand([("child", 10.0)])

        self.assertIs(root.children["child"], child)
        self.assertEqual(child.visit_count, 1)
        self.assertAlmostEqual(child.prior, 1.0)

    def test_terminal_node_cannot_expand(self):
        node = MCTSNode("terminal", is_terminal=True)

        with self.assertRaises(RuntimeError):
            node.expand([("child", 1.0)])

    def test_backpropagate_updates_ancestors(self):
        root = MCTSNode("root")
        child = root.expand([("child", 1.0)])[0]
        grandchild = child.expand([("grandchild", 1.0)])[0]

        grandchild.backpropagate(2.5)

        self.assertEqual(grandchild.visit_count, 1)
        self.assertEqual(child.visit_count, 1)
        self.assertEqual(root.visit_count, 1)
        self.assertAlmostEqual(root.value, 2.5)

    def test_puct_prefers_unvisited_candidate(self):
        root = MCTSNode("root")
        children = root.expand([
            ("visited", 0.5),
            ("unvisited", 0.5),
        ])

        visited, unvisited = children
        visited.update(1.0)
        root.update(1.0)

        self.assertGreater(
            unvisited.ucb_score(),
            visited.ucb_score(),
        )

    def test_best_child_uses_puct_score(self):
        root = MCTSNode("root")
        root.expand([("low", 0.5), ("high", 0.5)])

        root.children["low"].update(-1.0)
        root.children["high"].update(1.0)

        self.assertEqual(root.best_child(0.0).state, "high")

    def test_most_visited_child(self):
        root = MCTSNode("root")
        root.expand([("first", 0.5), ("second", 0.5)])

        root.children["second"].update(1.0)
        root.children["second"].update(1.0)

        self.assertEqual(root.most_visited_child().state, "second")

    def test_invalid_prior_is_rejected(self):
        for prior in (-1.0, math.inf, float("nan")):
            with self.subTest(prior=prior):
                with self.assertRaises(ValueError):
                    MCTSNode("state", prior=prior)


if __name__ == "__main__":
    unittest.main()
