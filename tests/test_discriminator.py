
import unittest

import torch

from model.discriminator_model import DiscriminatorModel


class TestDiscriminatorModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        self.vocab_size = 128
        self.hidden_size = 64

        self.model = DiscriminatorModel(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            max_seq_len=32,
            dropout=0.0,
            attention_dropout=0.0,
        )

    def test_output_shape(self):
        input_ids = torch.randint(0, self.vocab_size, (2, 12))

        scores = self.model(input_ids)

        self.assertEqual(scores.shape, (2,))

    def test_scores_are_finite(self):
        input_ids = torch.randint(0, self.vocab_size, (2, 12))

        scores = self.model(input_ids)

        self.assertTrue(torch.isfinite(scores).all().item())

    def test_gradient_flow(self):
        input_ids = torch.randint(0, self.vocab_size, (2, 12))
        targets = torch.randn(2)

        scores = self.model(input_ids)
        loss = torch.nn.functional.mse_loss(scores, targets)
        loss.backward()

        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

        self.assertTrue(any(grad is not None for grad in gradients))

        for grad in gradients:
            if grad is not None:
                self.assertTrue(torch.isfinite(grad).all().item())

    def test_padding_mask(self):
        input_ids = torch.randint(0, self.vocab_size, (2, 8))

        attention_mask = torch.tensor([
            [True, True, True, True, True, False, False, False],
            [True, True, True, True, True, True, True, False],
        ], dtype=torch.bool)

        scores = self.model(
            input_ids,
            attention_mask=attention_mask,
        )

        self.assertEqual(scores.shape, (2,))
        self.assertTrue(torch.isfinite(scores).all().item())

    def test_single_sequence(self):
        input_ids = torch.randint(0, self.vocab_size, (1, 8))

        scores = self.model(input_ids)

        self.assertEqual(scores.shape, (1,))

    def test_invalid_token_id_raises(self):
        input_ids = torch.tensor([[self.vocab_size]])

        with self.assertRaises(ValueError):
            self.model(input_ids)


if __name__ == "__main__":
    unittest.main()
