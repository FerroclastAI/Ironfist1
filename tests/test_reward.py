
"""Unit tests for IronFist1 reward calculation."""

import math
import unittest

import torch

from reasoning.reward import RewardConfig, calculate_reward


class TestRewardConfig(unittest.TestCase):
    def test_default_config(self):
        config = RewardConfig()
        self.assertEqual(config.discriminator_weight, 1.0)
        self.assertEqual(config.consistency_weight, 0.5)
        self.assertEqual(config.penalty_weight, 1.0)

    def test_negative_weights_are_rejected(self):
        for field in (
            "discriminator_weight",
            "consistency_weight",
            "penalty_weight",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    RewardConfig(**{field: -1.0})


class TestCalculateReward(unittest.TestCase):
    def test_default_formula(self):
        reward = calculate_reward(1.0, 0.5, 0.2)
        self.assertAlmostEqual(reward, 1.05)

    def test_custom_weights(self):
        config = RewardConfig(2.0, 3.0, 4.0)
        reward = calculate_reward(1.0, 0.5, 0.25, config)
        self.assertAlmostEqual(reward, 2.5)

    def test_zero_scores(self):
        self.assertAlmostEqual(calculate_reward(0.0), 0.0)

    def test_negative_penalty_increases_reward(self):
        # The reward function accepts negative penalties mathematically;
        # RStarReasoner itself rejects them in its penalty checker.
        reward = calculate_reward(1.0, penalty=-0.5)
        self.assertAlmostEqual(reward, 1.5)

    def test_nan_score_is_rejected(self):
        with self.assertRaises(ValueError):
            calculate_reward(float("nan"))

    def test_infinite_score_is_rejected(self):
        with self.assertRaises(ValueError):
            calculate_reward(float("inf"))

    def test_tensor_reward(self):
        reward = calculate_reward(
            torch.tensor(1.0),
            torch.tensor(0.5),
            torch.tensor(0.2),
        )
        self.assertIsInstance(reward, torch.Tensor)
        self.assertTrue(torch.isfinite(reward).item())
        self.assertAlmostEqual(reward.item(), 1.05, places=5)

    def test_multi_element_tensor_is_rejected(self):
        with self.assertRaises(ValueError):
            calculate_reward(torch.tensor([1.0, 2.0]))

    def test_tensor_components_on_different_devices_are_rejected(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")

        with self.assertRaises(ValueError):
            calculate_reward(
                torch.tensor(1.0),
                torch.tensor(0.5, device="cuda"),
            )

    def test_non_finite_tensor_is_rejected(self):
        with self.assertRaises(ValueError):
            calculate_reward(torch.tensor(float("nan")))


if __name__ == "__main__":
    unittest.main()
