
import unittest

import torch

from model.transformer import TransformerBlock


class TestTransformerBlock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.hidden_size = 64
        self.block = TransformerBlock(
            hidden_size=64,
            num_attention_heads=4,
            intermediate_size=128,
            dropout=0.0,
            attention_dropout=0.0,
        )

    def test_output_shape(self):
        x = torch.randn(2, 10, self.hidden_size)

        output = self.block(x)

        self.assertEqual(output.shape, x.shape)

    def test_output_is_finite(self):
        x = torch.randn(2, 10, self.hidden_size)

        output = self.block(x)

        self.assertTrue(torch.isfinite(output).all().item())

    def test_gradient_flow(self):
        x = torch.randn(
            2, 10, self.hidden_size, requires_grad=True
        )

        output = self.block(x)
        output.square().mean().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())

        for parameter in self.block.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(
                torch.isfinite(parameter.grad).all().item()
            )

    def test_padding_mask(self):
        x = torch.randn(2, 8, self.hidden_size)
        mask = torch.tensor([
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0],
        ])

        output = self.block(x, attention_mask=mask)

        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all().item())

    def test_single_token_sequence(self):
        x = torch.randn(1, 1, self.hidden_size)

        output = self.block(x)

        self.assertEqual(output.shape, (1, 1, self.hidden_size))
        self.assertTrue(torch.isfinite(output).all().item())


if __name__ == "__main__":
    unittest.main()
