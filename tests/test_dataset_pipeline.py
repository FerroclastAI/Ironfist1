import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from training.dataset import (
    choose_text_column,
    extract_text,
    find_data_files,
    prepare_dataset,
)


class FakeTokenizer:
    """Small deterministic tokenizer for pipeline tests."""

    vocab_size = 32_000

    @classmethod
    def from_pretrained(cls, directory):
        return cls()

    def encode(
        self,
        text,
        add_special_tokens=True,
        truncation=False,
    ):
        # Three deterministic IDs per record.
        return [2, len(text), 3]


class TestTextUtilities(unittest.TestCase):
    def test_extract_string(self):
        self.assertEqual(extract_text("Hello world"), "Hello world")

    def test_extract_nested_chat_record(self):
        record = {
            "messages": [
                {"content": "Hello"},
                {"content": "IronFist1"},
            ]
        }

        self.assertEqual(
            extract_text(record),
            "Hello\nIronFist1",
        )

    def test_extract_bytes(self):
        self.assertEqual(extract_text(b"hello"), "hello")

    def test_extract_none(self):
        self.assertEqual(extract_text(None), "")

    def test_choose_text_column_case_insensitive(self):
        self.assertEqual(
            choose_text_column(
                ["ID", "TEXT", "label"],
                requested_column=None,
            ),
            "TEXT",
        )

    def test_choose_requested_text_column(self):
        self.assertEqual(
            choose_text_column(
                ["prompt", "answer"],
                requested_column="answer",
            ),
            "answer",
        )

    def test_missing_requested_column_raises(self):
        with self.assertRaises(ValueError):
            choose_text_column(
                ["prompt", "answer"],
                requested_column="missing",
            )

    def test_no_recognized_text_column_raises(self):
        with self.assertRaises(ValueError):
            choose_text_column(
                ["id", "category"],
                requested_column=None,
            )

    def test_find_supported_files_recursively(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "one.txt").write_text("hello", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "two.parquet").touch()
            (root / "ignored.csv").touch()

            files = find_data_files(root)

            self.assertEqual(
                {path.name for path in files},
                {"one.txt", "two.parquet"},
            )

    def test_missing_input_directory_raises(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing"

            with self.assertRaises(FileNotFoundError):
                find_data_files(missing)


class TestPrepareDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        self.input_dir = self.root / "input"
        self.tokenizer_dir = self.root / "tokenizer"
        self.output_dir = self.root / "output"

        self.input_dir.mkdir()
        self.tokenizer_dir.mkdir()

        # Four records produce 12 tokens with FakeTokenizer.
        (self.input_dir / "records.txt").write_text(
            "alpha\nbravo\ncharlie\ndelta\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("training.dataset.WordPieceTokenizer", FakeTokenizer)
    def test_creates_shards_and_metadata(self):
        prepare_dataset(
            input_dir=self.input_dir,
            tokenizer_dir=self.tokenizer_dir,
            output_dir=self.output_dir,
            seq_length=2,
            tokens_per_shard=6,
        )

        shards = sorted(self.output_dir.glob("shard_*.npy"))

        self.assertEqual(len(shards), 2)

        for shard_path in shards:
            tokens = np.load(shard_path)

            self.assertEqual(tokens.dtype, np.int32)
            self.assertEqual(tokens.ndim, 1)
            self.assertEqual(len(tokens), 6)

        metadata = json.loads(
            (self.output_dir / "metadata.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(metadata["format"], "numpy_int32_token_shards")
        self.assertEqual(metadata["tokenizer_type"], "wordpiece")
        self.assertEqual(metadata["vocab_size"], 32_000)
        self.assertEqual(metadata["seq_length"], 2)
        self.assertEqual(metadata["total_records"], 4)
        self.assertEqual(metadata["total_tokens"], 12)
        self.assertEqual(metadata["saved_tokens"], 12)
        self.assertEqual(metadata["shard_count"], 2)

    @patch("training.dataset.WordPieceTokenizer", FakeTokenizer)
    def test_replaces_old_shards(self):
        self.output_dir.mkdir()
        np.save(
            self.output_dir / "shard_99999.npy",
            np.array([99, 99, 99], dtype=np.int32),
        )

        prepare_dataset(
            input_dir=self.input_dir,
            tokenizer_dir=self.tokenizer_dir,
            output_dir=self.output_dir,
            seq_length=2,
            tokens_per_shard=6,
        )

        self.assertFalse(
            (self.output_dir / "shard_99999.npy").exists()
        )

    def test_invalid_sequence_length_raises(self):
        with self.assertRaises(ValueError):
            prepare_dataset(
                input_dir=self.input_dir,
                tokenizer_dir=self.tokenizer_dir,
                output_dir=self.output_dir,
                seq_length=0,
            )

    def test_shard_too_small_raises(self):
        with self.assertRaises(ValueError):
            prepare_dataset(
                input_dir=self.input_dir,
                tokenizer_dir=self.tokenizer_dir,
                output_dir=self.output_dir,
                seq_length=2048,
                tokens_per_shard=100,
            )

    def test_invalid_parquet_batch_size_raises(self):
        with self.assertRaises(ValueError):
            prepare_dataset(
                input_dir=self.input_dir,
                tokenizer_dir=self.tokenizer_dir,
                output_dir=self.output_dir,
                parquet_batch_size=0,
            )


if __name__ == "__main__":
    unittest.main()
