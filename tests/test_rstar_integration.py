
"""Integration tests for IronFist1's rStar reasoning pipeline."""

import math
import unittest

from reasoning.rstar import RStarConfig, RStarReasoner


class DeterministicPolicy:
    """Produces predictable candidate continuations."""

    def __call__(self, state: str, depth: int):
        if depth >= 3:
            return []

        if state == "Solve 2 + 2":
            return [
                ("Solve 2 + 2 The answer is 4.", 0.8),
                ("Solve 2 + 2 The answer is 5.", 0.2),
            ]

        if "The answer is 4." in state:
            return [
                (state + " This is basic addition.", 0.8),
            ]

        if "The answer is 5." in state:
            return [
                (state + " This is an incorrect calculation.", 0.2),
            ]

        return []


class DeterministicDiscriminator:
    """Scores the candidate answer's quality."""

    def score(self, state: str) -> float:
        if "The answer is 4." in state:
            return 1.0

        if "The answer is 5." in state:
            return -1.0

        return 0.0


class RStarIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.reasoner = RStarReasoner(
            policy_generator=DeterministicPolicy(),
            discriminator=DeterministicDiscriminator(),
            config=RStarConfig(
                num_simulations=12,
                max_search_depth=3,
                max_candidates_per_node=2,
                exploration_constant=1.414,
                discount_factor=0.99,
            ),
        )

    def test_full_pipeline_returns_answer(self):
        result = self.reasoner.reason("Solve 2 + 2")

        self.assertTrue(result.answer.strip())
        self.assertIsInstance(result.reward, float)
        self.assertTrue(math.isfinite(result.reward))

    def test_selected_answer_contains_correct_candidate(self):
        result = self.reasoner.reason("Solve 2 + 2")

        self.assertIn("The answer is 4.", result.answer)

    def test_result_contains_search_statistics(self):
        result = self.reasoner.reason("Solve 2 + 2")

        self.assertGreaterEqual(result.simulations_completed, 1)
        self.assertGreaterEqual(result.candidate_count, 1)
        self.assertTrue(math.isfinite(result.search_result.best_value))

    def test_optional_checkers_integrate(self):
        reasoner = RStarReasoner(
            policy_generator=DeterministicPolicy(),
            discriminator=DeterministicDiscriminator(),
            config=RStarConfig(
                num_simulations=4,
                max_search_depth=3,
            ),
            consistency_checker=lambda state: 0.5,
            penalty_checker=lambda state: 0.1,
        )

        result = reasoner.reason("Solve 2 + 2")

        self.assertTrue(math.isfinite(result.reward))
        self.assertTrue(math.isfinite(result.discriminator_score))

    def test_invalid_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            self.reasoner.reason("  ")


if __name__ == "__main__":
    unittest.main()
