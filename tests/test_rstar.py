
"""Tests for IronFist1's rStar reasoning system."""

import math
import unittest

from reasoning.rstar import RStarConfig, RStarReasoner
from reasoning.reward import RewardConfig, calculate_reward


class FakePolicy:
    """Small deterministic policy generator for unit tests."""

    def __call__(self, state: str, depth: int):
        if state == "What is 2 + 2?":
            return [
                ("What is 2 + 2? The answer is 4.", 1.0),
            ]
        return []


class FakeDiscriminator:
    """Simple scorer that avoids loading a real neural network."""

    def score(self, state: str) -> float:
        if "The answer is 4." in state:
            return 1.0
        return 0.0


class RStarConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self):
        config = RStarConfig()

        self.assertGreaterEqual(config.num_simulations, 1)
        self.assertGreaterEqual(config.max_search_depth, 1)

    def test_invalid_num_simulations_raises(self):
        with self.assertRaises(ValueError):
            RStarConfig(num_simulations=0)

    def test_invalid_search_depth_raises(self):
        with self.assertRaises(ValueError):
            RStarConfig(max_search_depth=0)

    def test_invalid_candidate_count_raises(self):
        with self.assertRaises(ValueError):
            RStarConfig(max_candidates_per_node=0)

    def test_negative_exploration_constant_raises(self):
        with self.assertRaises(ValueError):
            RStarConfig(exploration_constant=-1.0)

    def test_non_finite_exploration_constant_raises(self):
        with self.assertRaises(ValueError):
            RStarConfig(exploration_constant=math.inf)

    def test_invalid_discount_factor_raises(self):
        with self.assertRaises(ValueError):
            RStarConfig(discount_factor=1.5)

    def test_negative_reward_weight_raises(self):
        with self.assertRaises(ValueError):
            RStarConfig(discriminator_weight=-1.0)


class RewardTests(unittest.TestCase):
    def test_default_reward(self):
        reward = calculate_reward(
            discriminator_score=1.0,
            consistency_score=0.5,
            penalty=0.2,
        )

        self.assertAlmostEqual(reward, 1.05)

    def test_custom_reward_weights(self):
        config = RewardConfig(
            discriminator_weight=2.0,
            consistency_weight=3.0,
            penalty_weight=4.0,
        )

        reward = calculate_reward(
            discriminator_score=1.0,
            consistency_score=0.5,
            penalty=0.25,
            config=config,
        )

        self.assertAlmostEqual(reward, 2.5)

    def test_non_finite_score_raises(self):
        with self.assertRaises(ValueError):
            calculate_reward(discriminator_score=float("nan"))


class RStarReasonerTests(unittest.TestCase):
    def setUp(self):
        self.prompt = "What is 2 + 2?"
        self.reasoner = RStarReasoner(
            policy_generator=FakePolicy(),
            discriminator=FakeDiscriminator(),
            config=RStarConfig(
                num_simulations=4,
                max_search_depth=3,
                max_candidates_per_node=2,
            ),
        )

    def test_reason_returns_result(self):
        result = self.reasoner.reason(self.prompt)

        self.assertEqual(result.prompt, self.prompt)
        self.assertIsInstance(result.answer, str)
        self.assertTrue(result.answer.strip())
        self.assertIsInstance(result.reward, float)
        self.assertIsInstance(result.discriminator_score, float)

    def test_answer_excludes_prompt_prefix(self):
        result = self.reasoner.reason(self.prompt)

        self.assertNotIn(self.prompt, result.answer)
        self.assertEqual(result.answer, "The answer is 4.")

    def test_discriminator_score_is_returned(self):
        result = self.reasoner.reason(self.prompt)

        self.assertAlmostEqual(result.discriminator_score, 1.0)

    def test_reward_is_finite(self):
        result = self.reasoner.reason(self.prompt)

        self.assertTrue(math.isfinite(result.reward))

    def test_search_completes_simulations(self):
        result = self.reasoner.reason(self.prompt)

        self.assertGreaterEqual(result.simulations_completed, 1)

    def test_call_returns_answer_string(self):
        answer = self.reasoner(self.prompt)

        self.assertIsInstance(answer, str)
        self.assertEqual(answer, "The answer is 4.")

    def test_empty_prompt_raises(self):
        with self.assertRaises(ValueError):
            self.reasoner.reason("")

    def test_whitespace_prompt_raises(self):
        with self.assertRaises(ValueError):
            self.reasoner.reason("   ")

    def test_non_string_prompt_raises(self):
        with self.assertRaises(ValueError):
            self.reasoner.reason(None)

    def test_consistency_checker_affects_reward(self):
        reasoner = RStarReasoner(
            policy_generator=FakePolicy(),
            discriminator=FakeDiscriminator(),
            consistency_checker=lambda state: 2.0,
            config=RStarConfig(
                num_simulations=2,
                consistency_weight=0.5,
            ),
        )

        result = reasoner.reason(self.prompt)

        # Discriminator score 1.0 + consistency 2.0 * weight 0.5.
        self.assertAlmostEqual(result.reward, 2.0)

    def test_penalty_checker_reduces_reward(self):
        reasoner = RStarReasoner(
            policy_generator=FakePolicy(),
            discriminator=FakeDiscriminator(),
            penalty_checker=lambda state: 0.25,
            config=RStarConfig(
                num_simulations=2,
                penalty_weight=2.0,
            ),
        )

        result = reasoner.reason(self.prompt)

        # Discriminator score 1.0 - penalty 0.25 * weight 2.0.
        self.assertAlmostEqual(result.reward, 0.5)

    def test_negative_penalty_raises(self):
        reasoner = RStarReasoner(
            policy_generator=FakePolicy(),
            discriminator=FakeDiscriminator(),
            penalty_checker=lambda state: -1.0,
            config=RStarConfig(num_simulations=2),
        )

        with self.assertRaises(ValueError):
            reasoner.reason(self.prompt)

    def test_non_finite_consistency_raises(self):
        reasoner = RStarReasoner(
            policy_generator=FakePolicy(),
            discriminator=FakeDiscriminator(),
            consistency_checker=lambda state: float("nan"),
            config=RStarConfig(num_simulations=2),
        )

        with self.assertRaises(ValueError):
            reasoner.reason(self.prompt)

    def test_non_finite_discriminator_score_raises(self):
        class BadDiscriminator:
            def score(self, state):
                return float("nan")

        reasoner = RStarReasoner(
            policy_generator=FakePolicy(),
            discriminator=BadDiscriminator(),
            config=RStarConfig(num_simulations=2),
        )

        with self.assertRaises(ValueError):
            reasoner.reason(self.prompt)


if __name__ == "__main__":
    unittest.main()
