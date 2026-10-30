
"""Tests for structured rStar result output."""

import unittest

from reasoning.rstar import RStarConfig, RStarReasoner


class FakePolicy:
    def __call__(self, state, count):
        if state == "Question":
            return [("Question Answer", 1.0)]
        return []


class FakeDiscriminator:
    def score(self, state):
        return 1.0 if "Answer" in state else 0.0


class TestRStarResult(unittest.TestCase):
    def setUp(self):
        self.reasoner = RStarReasoner(
            policy_generator=FakePolicy(),
            discriminator=FakeDiscriminator(),
            config=RStarConfig(num_simulations=4),
        )

        self.result = self.reasoner.reason("Question")

    def test_result_has_diagnostics(self):
        self.assertIsNotNone(self.result.diagnostics)
        self.assertGreaterEqual(
            self.result.diagnostics.nodes_explored, 1
        )

    def test_result_has_trace(self):
        self.assertIsNotNone(self.result.trace)
        self.assertGreaterEqual(len(self.result.trace.steps), 1)

    def test_result_exports_dictionary(self):
        data = self.result.to_dict()

        self.assertEqual(data["prompt"], "Question")
        self.assertEqual(data["answer"], "Answer")
        self.assertIn("diagnostics", data)
        self.assertIn("trace", data)

    def test_result_is_json_serializable(self):
        import json

        serialized = json.dumps(self.result.to_dict())
        self.assertIsInstance(serialized, str)

    def test_simulation_count_is_exposed(self):
        self.assertGreaterEqual(self.result.simulations_completed, 1)


if __name__ == "__main__":
    unittest.main()
