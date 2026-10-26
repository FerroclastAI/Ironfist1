
import unittest

import torch

from model.normalization import RMSNorm


class TestRMSNorm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_output_shape(self):
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(2, 10, 64)

        output = norm(x)

        self.assertEqual(output.shape, x.shape)

    def test_output_is_finite(self):
        norm = RMSNorm(hidden_size=32)
        x = torch.randn(2, 8, 32)

        output = norm(x)

        self.assertTrue(torch.isfinite(output).all().item())

    def test_normalization_scale(self):
        norm = RMSNorm(hidden_size=16)
        x = torch.randn(2, 5, 16)

        output = norm(x)

        # RMSNorm should normalize each vector's root-mean-square
        # magnitude to approximately 1 when its learned weights
        # are initialized to 1.
        rms = torch.sqrt(torch.mean(output.float() ** 2, dim=-1))

        self.assertTrue(
            torch.allclose(rms, torch.ones_like(rms), atol=1e-4, rtol=1e-4)
        )

    def test_learnable_weights_exist(self):
        norm = RMSNorm(hidden_size=32)

        self.assertTrue(hasattr(norm, "weight"))
        self.assertEqual(norm.weight.shape, (32,))
        self.assertTrue(norm.weight.requires_grad)

    def test_gradient_flow(self):
        norm = RMSNorm(hidden_size=32)
        x = torch.randn(2, 5, 32, requires_grad=True)

        output = norm(x)
        loss = output.square().mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(norm.weight.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertTrue(torch.isfinite(norm.weight.grad).all().item())

    def test_zero_input_is_finite(self):
        norm = RMSNorm(hidden_size=32)
        x = torch.zeros(2, 5, 32)

        output = norm(x)

        self.assertTrue(torch.isfinite(output).all().item())
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_different_batch_and_sequence_sizes(self):
        norm = RMSNorm(hidden_size=32)

        for batch_size, seq_length in [(1, 1), (2, 8), (4, 16)]:
            with self.subTest(
                batch_size=batch_size, seq_length=seq_length
            ):
                x = torch.randn(batch_size, seq_length, 32)
                output = norm(x)

                self.assertEqual(
                    output.shape, (batch_size, seq_length, 32)
                )


if __name__ == "__main__":
    unittest.main()
