
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from training.trainer import Trainer, create_scheduler


class TinyPolicyModel(nn.Module):
    """Small language model for testing the training engine."""

    def __init__(self, vocab_size=16, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, labels=None):
        logits = self.output(self.embedding(input_ids))

        if labels is None:
            return logits

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        return logits, loss


def make_batches():
    """Return a few deterministic next-token prediction batches."""
    return [
        {
            "input_ids": torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]]),
            "labels": torch.tensor([[2, 3, 4, 5], [3, 4, 5, 6]]),
        },
        {
            "input_ids": torch.tensor([[3, 4, 5, 6], [4, 5, 6, 7]]),
            "labels": torch.tensor([[4, 5, 6, 7], [5, 6, 7, 8]]),
        },
        {
            "input_ids": torch.tensor([[5, 6, 7, 8], [6, 7, 8, 9]]),
            "labels": torch.tensor([[6, 7, 8, 9], [7, 8, 9, 10]]),
        },
    ]


class TestCreateScheduler(unittest.TestCase):
    def test_scheduler_creation(self):
        model = TinyPolicyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        scheduler = create_scheduler(
            optimizer,
            warmup_steps=2,
            total_steps=10,
        )

        self.assertIsNotNone(scheduler)

    def test_invalid_total_steps_raise(self):
        model = TinyPolicyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        with self.assertRaises(ValueError):
            create_scheduler(
                optimizer,
                warmup_steps=0,
                total_steps=0,
            )

    def test_warmup_must_be_less_than_total(self):
        model = TinyPolicyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        with self.assertRaises(ValueError):
            create_scheduler(
                optimizer,
                warmup_steps=10,
                total_steps=10,
            )

    def test_invalid_min_lr_ratio_raises(self):
        model = TinyPolicyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        with self.assertRaises(ValueError):
            create_scheduler(
                optimizer,
                warmup_steps=0,
                total_steps=10,
                min_lr_ratio=1.5,
            )


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_dir = Path(self.temp_dir.name)

        torch.manual_seed(1234)

        self.model = TinyPolicyModel()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.01,
        )

        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            train_loader=make_batches(),
            device="cpu",
            checkpoint_dir=self.checkpoint_dir,
            use_amp=False,
            log_interval=1,
            save_interval=0,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_train_step_returns_finite_loss(self):
        loss = self.trainer.train_step([make_batches()[0]])

        self.assertTrue(torch.isfinite(torch.tensor(loss)).item())
        self.assertGreater(loss, 0.0)
        self.assertEqual(self.trainer.global_step, 1)

    def test_train_step_updates_parameters(self):
        before = [
            parameter.detach().clone()
            for parameter in self.trainer.model.parameters()
        ]

        self.trainer.train_step([make_batches()[0]])

        after = list(self.trainer.model.parameters())

        self.assertTrue(
            any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, after)
            )
        )

    def test_empty_batch_list_raises(self):
        with self.assertRaises(ValueError):
            self.trainer.train_step([])

    def test_gradient_accumulation_accepts_multiple_batches(self):
        loss = self.trainer.train_step(make_batches()[:2])

        self.assertTrue(torch.isfinite(torch.tensor(loss)).item())
        self.assertEqual(self.trainer.global_step, 1)

    def test_checkpoint_saves(self):
        self.trainer.train_step([make_batches()[0]])

        path = self.trainer.save_checkpoint("test_checkpoint.pt")

        self.assertTrue(path.is_file())

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        self.assertEqual(checkpoint["global_step"], 1)
        self.assertIn("model", checkpoint)
        self.assertIn("optimizer", checkpoint)

    def test_checkpoint_restores_model_and_step(self):
        self.trainer.train_step([make_batches()[0]])
        path = self.trainer.save_checkpoint("restore.pt")

        saved_parameters = [
            parameter.detach().clone()
            for parameter in self.trainer.model.parameters()
        ]

        with torch.no_grad():
            for parameter in self.trainer.model.parameters():
                parameter.add_(1.0)

        self.trainer.global_step = 99
        self.trainer.load_checkpoint(path, restore_rng=False)

        self.assertEqual(self.trainer.global_step, 1)

        for saved, restored in zip(
            saved_parameters,
            self.trainer.model.parameters(),
        ):
            self.assertTrue(torch.equal(saved, restored))

    def test_fit_runs_requested_steps(self):
        history = self.trainer.fit(max_steps=2)

        self.assertEqual(self.trainer.global_step, 2)
        self.assertEqual(len(history), 2)
        self.assertTrue((self.checkpoint_dir / "latest.pt").is_file())

    def test_invalid_max_steps_raises(self):
        with self.assertRaises(ValueError):
            self.trainer.fit(max_steps=0)

    def test_invalid_max_epochs_raises(self):
        with self.assertRaises(ValueError):
            self.trainer.fit(max_steps=2, max_epochs=0)

    def test_invalid_accumulation_steps_raise(self):
        with self.assertRaises(ValueError):
            Trainer(
                model=TinyPolicyModel(),
                optimizer=torch.optim.AdamW(
                    TinyPolicyModel().parameters(),
                    lr=0.001,
                ),
                train_loader=make_batches(),
                device="cpu",
                checkpoint_dir=self.checkpoint_dir,
                gradient_accumulation_steps=0,
            )

    def test_invalid_save_interval_raises(self):
        with self.assertRaises(ValueError):
            Trainer(
                model=TinyPolicyModel(),
                optimizer=self.optimizer,
                train_loader=make_batches(),
                device="cpu",
                checkpoint_dir=self.checkpoint_dir,
                save_interval=-1,
            )


if __name__ == "__main__":
    unittest.main()
