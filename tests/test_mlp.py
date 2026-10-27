
import unittest

import torch

from model.mlp import SwiGLU


class TestSwiGLU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.hidden_size = 64
        self.intermediate_size = 128

        self.mlp = SwiGLU(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            dropout=0.0,
        )

    def test_output_shape(self):
        x = torch.randn(2, 10, self.hidden_size)

        output = self.mlp(x)

        self.assertEqual(
            output.shape,
            (2, 10, self.hidden_size),
        )

    def test_output_is_finite(self):
        x = torch.randn(2, 10, self.hidden_size)

        output = self.mlp(x)

        self.assertTrue(torch.isfinite(output).all().item())

    def test_gradient_flow(self):
        x = torch.randn(
            2, 10, self.hidden_size, requires_grad=True
        )

        output = self.mlp(x)
        loss = output.square().mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())

        for parameter in self.mlp.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(
                torch.isfinite(parameter.grad).all().item()
            )

    def test_different_sequence_lengths(self):
        for seq_length in (1, 5, 16):
            with self.subTest(seq_length=seq_length):
                x = torch.randn(2, seq_length, self.hidden_size)
                output = self.mlp(x)

                self.assertEqual(
                    output.shape,
                    (2, seq_length, self.hidden_size),
                )

    def test_dropout_configuration(self):
        mlp = SwiGLU(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            dropout=0.2,
        )

        self.assertTrue(hasattr(mlp, "dropout"))

    def test_invalid_dropout_raises(self):
        with self.assertRaises(ValueError):
            SwiGLU(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                dropout=1.5,
            )

    def test_invalid_hidden_size_raises(self):
        with self.assertRaises(ValueError):
            SwiGLU(
                hidden_size=0,
                intermediate_size=self.intermediate_size,
            )


if __name__ == "__main__":
    unittest.main()
