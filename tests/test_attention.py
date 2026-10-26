
import unittest

import torch

from model.attention import CausalSelfAttention


class TestCausalSelfAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        self.attention = CausalSelfAttention(
            hidden_size=64,
            num_attention_heads=4,
            attention_dropout=0.0,
            use_bias=False,
        )

        self.attention.eval()

    def test_output_shape(self):
        """Attention must preserve batch, sequence, and hidden dimensions."""
        x = torch.randn(2, 8, 64)

        output = self.attention(x)

        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_gradient_flow(self):
        """Gradients must flow through the attention module."""
        x = torch.randn(2, 8, 64, requires_grad=True)

        output = self.attention(x)
        loss = output.square().mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

        for parameter in self.attention.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_causal_mask_prevents_future_information(self):
        """
        Changing future tokens must not change earlier output positions.

        We compare the first four outputs while changing only the inputs
        at positions four and later.
        """
        x1 = torch.randn(1, 8, 64)
        x2 = x1.clone()

        x2[:, 4:, :] = torch.randn(1, 4, 64) * 10.0

        with torch.no_grad():
            output1 = self.attention(x1)
            output2 = self.attention(x2)

        torch.testing.assert_close(
            output1[:, :4, :],
            output2[:, :4, :],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_padding_mask_blocks_padded_keys(self):
        """
        Changing padded token embeddings must not affect valid outputs.

        The first four positions are valid; the final four are padding.
        """
        x1 = torch.randn(1, 8, 64)
        x2 = x1.clone()

        x2[:, 4:, :] = torch.randn(1, 4, 64) * 10.0

        mask = torch.tensor(
            [[True, True, True, True, False, False, False, False]]
        )

        with torch.no_grad():
            output1 = self.attention(x1, attention_mask=mask)
            output2 = self.attention(x2, attention_mask=mask)

        torch.testing.assert_close(
            output1[:, :4, :],
            output2[:, :4, :],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_padding_mask_preserves_output_shape(self):
        """An explicit padding mask must preserve the output shape."""
        x = torch.randn(2, 8, 64)

        mask = torch.tensor(
            [
                [True, True, True, True, True, False, False, False],
                [True, True, True, True, True, True, True, False],
            ]
        )

        output = self.attention(x, attention_mask=mask)

        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_invalid_attention_mask_shape_raises(self):
        """Incorrect mask dimensions should fail with a clear error."""
        x = torch.randn(2, 8, 64)
        mask = torch.ones(2, 7, dtype=torch.bool)

        with self.assertRaises(ValueError):
            self.attention(x, attention_mask=mask)

    def test_invalid_hidden_size_raises(self):
        """The input hidden dimension must match the module configuration."""
        x = torch.randn(2, 8, 32)

        with self.assertRaises(ValueError):
            self.attention(x)

    def test_invalid_head_configuration_raises(self):
        """Hidden size must be divisible by the number of attention heads."""
        with self.assertRaises(ValueError):
            CausalSelfAttention(
                hidden_size=63,
                num_attention_heads=4,
            )


if __name__ == "__main__":
    unittest.main()
