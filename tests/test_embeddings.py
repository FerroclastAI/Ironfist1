
import unittest

import torch

from model.embeddings import (
    TokenEmbeddings,
    RotaryPositionalEmbedding,
    apply_rotary_embeddings,
)


class TestTokenEmbeddings(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 256
        self.hidden_size = 64
        self.embeddings = TokenEmbeddings(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
        )

    def test_output_shape(self):
        input_ids = torch.randint(0, self.vocab_size, (2, 10))

        output = self.embeddings(input_ids)

        self.assertEqual(output.shape, (2, 10, self.hidden_size))

    def test_output_is_finite(self):
        input_ids = torch.randint(0, self.vocab_size, (2, 10))

        output = self.embeddings(input_ids)

        self.assertTrue(torch.isfinite(output).all().item())

    def test_same_token_has_same_embedding(self):
        input_ids = torch.tensor([[5, 10, 5]])

        output = self.embeddings(input_ids)

        self.assertTrue(
            torch.allclose(output[:, 0, :], output[:, 2, :])
        )

    def test_invalid_token_id_raises(self):
        input_ids = torch.tensor([[self.vocab_size]])

        with self.assertRaises((IndexError, RuntimeError)):
            self.embeddings(input_ids)

    def test_gradient_flow(self):
        input_ids = torch.randint(0, self.vocab_size, (2, 8))

        output = self.embeddings(input_ids)
        loss = output.square().mean()
        loss.backward()

        weight = self.embeddings.embedding.weight

        self.assertIsNotNone(weight.grad)
        self.assertTrue(torch.isfinite(weight.grad).all().item())


class TestRotaryPositionalEmbedding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.head_dim = 32
        self.rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim
        )

    def test_output_shapes(self):
        cos, sin = self.rope(16)

        expected_shape = (16, self.head_dim // 2)

        self.assertEqual(cos.shape, expected_shape)
        self.assertEqual(sin.shape, expected_shape)

    def test_output_is_finite(self):
        cos, sin = self.rope(16)

        self.assertTrue(torch.isfinite(cos).all().item())
        self.assertTrue(torch.isfinite(sin).all().item())

    def test_position_zero(self):
        cos, sin = self.rope(4)

        self.assertTrue(
            torch.allclose(cos[0], torch.ones_like(cos[0]))
        )
        self.assertTrue(
            torch.allclose(
                sin[0],
                torch.zeros_like(sin[0]),
                atol=1e-6,
            )
        )

    def test_invalid_odd_head_dimension_raises(self):
        with self.assertRaises(ValueError):
            RotaryPositionalEmbedding(head_dim=31)

    def test_gradient_flow_through_rotary_application(self):
        batch_size = 2
        num_heads = 4
        seq_length = 8

        x = torch.randn(
            batch_size,
            num_heads,
            seq_length,
            self.head_dim,
            requires_grad=True,
        )

        cos, sin = self.rope(seq_length)
        output = apply_rotary_embeddings(x, cos, sin)

        self.assertEqual(output.shape, x.shape)

        output.square().mean().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())


if __name__ == "__main__":
    unittest.main()
