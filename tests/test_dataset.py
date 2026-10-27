
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from training.data_loader import TokenShardDataset, create_dataloader


class TestTokenShardDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.shard_dir = Path(self.temp_dir.name)

        # Tokens are deliberately sequential so window boundaries
        # and next-token labels are easy to verify.
        tokens = np.arange(20, dtype=np.int64)
        np.save(self.shard_dir / "shard_000.npy", tokens)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_first_window_and_next_token_labels(self):
        dataset = TokenShardDataset(
            self.shard_dir,
            seq_length=4,
        )

        sample = next(iter(dataset))

        self.assertTrue(
            torch.equal(
                sample["input_ids"],
                torch.tensor([0, 1, 2, 3]),
            )
        )
        self.assertTrue(
            torch.equal(
                sample["labels"],
                torch.tensor([1, 2, 3, 4]),
            )
        )

    def test_sample_shapes_and_dtypes(self):
        dataset = TokenShardDataset(
            self.shard_dir,
            seq_length=5,
        )

        for sample in dataset:
            self.assertEqual(sample["input_ids"].shape, (5,))
            self.assertEqual(sample["labels"].shape, (5,))
            self.assertEqual(sample["input_ids"].dtype, torch.int64)
            self.assertEqual(sample["labels"].dtype, torch.int64)

    def test_default_stride(self):
        dataset = TokenShardDataset(
            self.shard_dir,
            seq_length=4,
        )

        samples = list(dataset)

        # 20 tokens produce windows of 5 tokens each.
        # With stride 4, there are four complete windows.
        self.assertEqual(len(samples), 4)

    def test_custom_stride(self):
        dataset = TokenShardDataset(
            self.shard_dir,
            seq_length=4,
            stride=2,
        )

        samples = list(dataset)

        # Valid starts are 0, 2, 4, 6, 8, 10, 12, 14.
        self.assertEqual(len(samples), 8)

        self.assertTrue(
            torch.equal(
                samples[1]["input_ids"],
                torch.tensor([2, 3, 4, 5]),
            )
        )

    def test_last_window_ends_at_final_token(self):
        dataset = TokenShardDataset(
            self.shard_dir,
            seq_length=4,
        )

        samples = list(dataset)

        self.assertTrue(
            torch.equal(
                samples[-1]["input_ids"],
                torch.tensor([12, 13, 14, 15]),
            )
        )
        self.assertTrue(
            torch.equal(
                samples[-1]["labels"],
                torch.tensor([13, 14, 15, 16]),
            )
        )

    def test_multiple_shards(self):
        np.save(
            self.shard_dir / "shard_001.npy",
            np.arange(100, 110, dtype=np.int64),
        )

        dataset = TokenShardDataset(
            self.shard_dir,
            seq_length=4,
        )

        samples = list(dataset)

        # First shard: four windows.
        # Second shard: one complete window.
        self.assertEqual(len(samples), 5)

        self.assertTrue(
            torch.equal(
                samples[-1]["input_ids"],
                torch.tensor([100, 101, 102, 103]),
            )
        )

    def test_short_shard_is_skipped(self):
        short_dir = self.shard_dir / "short"
        short_dir.mkdir()

        np.save(
            short_dir / "shard_000.npy",
            np.array([1, 2, 3, 4], dtype=np.int64),
        )

        dataset = TokenShardDataset(
            short_dir,
            seq_length=4,
        )

        self.assertEqual(list(dataset), [])

    def test_invalid_sequence_length_raises(self):
        with self.assertRaises(ValueError):
            TokenShardDataset(self.shard_dir, seq_length=0)

        with self.assertRaises(ValueError):
            TokenShardDataset(self.shard_dir, seq_length=-1)

    def test_invalid_stride_raises(self):
        with self.assertRaises(ValueError):
            TokenShardDataset(
                self.shard_dir,
                seq_length=4,
                stride=0,
            )

        with self.assertRaises(ValueError):
            TokenShardDataset(
                self.shard_dir,
                seq_length=4,
                stride=-1,
            )

    def test_missing_directory_raises(self):
        missing_dir = self.shard_dir / "does_not_exist"

        with self.assertRaises(FileNotFoundError):
            TokenShardDataset(missing_dir, seq_length=4)

    def test_empty_shard_directory_raises(self):
        empty_dir = self.shard_dir / "empty"
        empty_dir.mkdir()

        with self.assertRaises(FileNotFoundError):
            TokenShardDataset(empty_dir, seq_length=4)

    def test_non_integer_tokens_raise(self):
        invalid_dir = self.shard_dir / "invalid"
        invalid_dir.mkdir()

        np.save(
            invalid_dir / "shard_000.npy",
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        )

        dataset = TokenShardDataset(
            invalid_dir,
            seq_length=4,
        )

        with self.assertRaises(TypeError):
            list(dataset)

    def test_non_1d_shard_raises(self):
        invalid_dir = self.shard_dir / "two_dimensional"
        invalid_dir.mkdir()

        np.save(
            invalid_dir / "shard_000.npy",
            np.ones((4, 4), dtype=np.int64),
        )

        dataset = TokenShardDataset(
            invalid_dir,
            seq_length=2,
        )

        with self.assertRaises(ValueError):
            list(dataset)


class TestCreateDataLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.shard_dir = Path(self.temp_dir.name)

        np.save(
            self.shard_dir / "shard_000.npy",
            np.arange(40, dtype=np.int64),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_batch_shapes(self):
        loader = create_dataloader(
            shard_dir=self.shard_dir,
            seq_length=4,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )

        batch = next(iter(loader))

        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["labels"].shape, (2, 4))

    def test_batches_have_next_token_labels(self):
        loader = create_dataloader(
            shard_dir=self.shard_dir,
            seq_length=4,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )

        for batch in loader:
            self.assertTrue(
                torch.equal(
                    batch["labels"][:, :-1],
                    batch["input_ids"][:, 1:],
                )
            )

    def test_invalid_batch_size_raises(self):
        with self.assertRaises(ValueError):
            create_dataloader(
                self.shard_dir,
                seq_length=4,
                batch_size=0,
            )

        with self.assertRaises(ValueError):
            create_dataloader(
                self.shard_dir,
                seq_length=4,
                batch_size=-1,
            )

    def test_invalid_worker_count_raises(self):
        with self.assertRaises(ValueError):
            create_dataloader(
                self.shard_dir,
                seq_length=4,
                batch_size=2,
                num_workers=-1,
            )


if __name__ == "__main__":
    unittest.main()
