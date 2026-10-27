
import unittest
import torch

from model.policy_model import PolicyModel


class TestPolicyModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        self.vocab_size = 128
        self.hidden_size = 64

        self.model = PolicyModel(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            max_seq_len=32,
            dropout=0.0,
            attention_dropout=0.0,
            tie_word_embeddings=True,
        )

    def test_logits_shape(self):
        input_ids = torch.randint(0, self.vocab_size, (2, 12))
        logits = self.model(input_ids)

        self.assertEqual(logits.shape, (2, 12, self.vocab_size))

    def test_loss_is_finite(self):
        input_ids = torch.randint(0, self.vocab_size, (2, 12))
        labels = torch.randint(0, self.vocab_size, (2, 12))

        logits, loss = self.model(input_ids, labels=labels)

        self.assertEqual(logits.shape, (2, 12, self.vocab_size))
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)

    def test_gradient_flow(self):
        input_ids = torch.randint(0, self.vocab_size, (2, 12))
        labels = torch.randint(0, self.vocab_size, (2, 12))

        logits, loss = self.model(input_ids, labels=labels)
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

    def test_tied_embeddings(self):
        self.assertIs(
            self.model.lm_head.weight,
            self.model.token_embeddings.embedding.weight,
        )

    def test_different_sequence_lengths(self):
        for seq_length in (1, 8, 16):
            with self.subTest(seq_length=seq_length):
                input_ids = torch.randint(
                    0, self.vocab_size, (2, seq_length)
                )
                logits = self.model(input_ids)

                self.assertEqual(
                    logits.shape,
                    (2, seq_length, self.vocab_size),
                )

    def test_invalid_token_id_raises(self):
        input_ids = torch.tensor([[self.vocab_size]])

        with self.assertRaises(ValueError):
            self.model(input_ids)


if __name__ == "__main__":
    unittest.main()
