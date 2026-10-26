
"""
IronFist1 WordPiece Vocabulary Trainer.

Supports:
    .txt      Plain-text training files
    .parquet  Structured datasets containing text or conversations

Default input:
    data/processed/

Default output:
    tokenizer/vocab/
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

from tokenizers import Tokenizer, models, normalizers
from tokenizers import pre_tokenizers, trainers

try:
    from .tokenizer import WordPieceTokenizer
except ImportError:
    from tokenizer.tokenizer import WordPieceTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tokenizer" / "vocab"

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
]

# Columns commonly used to store text in datasets.
PREFERRED_TEXT_COLUMNS = [
    "text",
    "content",
    "document",
    "article",
    "body",
    "messages",
    "conversation",
    "prompt",
    "completion",
    "response",
]


def find_data_files(data_dir: Path) -> list[Path]:
    """Find supported training files recursively."""
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Training data directory does not exist: {data_dir}"
        )

    files = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".txt", ".parquet"}
    )

    if not files:
        raise FileNotFoundError(
            f"No .txt or .parquet files found in {data_dir}."
        )

    return files


def extract_text(value: Any) -> str:
    """
    Convert common Parquet values into training text.

    Supports strings, lists, dictionaries, and nested
    conversation structures.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, dict):
        # Conversation records often store their text in
        # one of these fields.
        preferred_keys = [
            "text",
            "content",
            "value",
            "message",
            "messages",
            "prompt",
            "completion",
            "response",
        ]

        parts = []

        for key in preferred_keys:
            if key in value:
                text = extract_text(value[key])
                if text:
                    parts.append(text)

        if parts:
            return "\n".join(parts)

        # Handle other nested dictionaries without dumping
        # metadata such as IDs or timestamps into the corpus.
        return ""

    if isinstance(value, (list, tuple)):
        parts = [
            extract_text(item)
            for item in value
        ]
        return "\n".join(part for part in parts if part)

    # Ignore numbers, booleans, timestamps, and other
    # non-text values instead of training on their string form.
    return ""


def choose_text_column(
    column_names: list[str],
    requested_column: str | None = None,
) -> str:
    """Select a text column using an explicit or automatic choice."""
    if requested_column is not None:
        if requested_column not in column_names:
            raise ValueError(
                f"Requested column '{requested_column}' was not found. "
                f"Available columns: {column_names}"
            )
        return requested_column

    columns_by_lowercase = {
        name.lower(): name for name in column_names
    }

    for preferred in PREFERRED_TEXT_COLUMNS:
        if preferred in columns_by_lowercase:
            return columns_by_lowercase[preferred]

    raise ValueError(
        "Could not automatically identify a text column. "
        f"Available columns: {column_names}. "
        "Use --text-column COLUMN_NAME."
    )


def convert_parquet_to_text(
    parquet_path: Path,
    output_path: Path,
    text_column: str | None = None,
    batch_size: int = 1000,
) -> int:
    """
    Stream a Parquet column into a temporary UTF-8 text file.

    Returns the number of non-empty text records written.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet support requires PyArrow. Install it with: "
            "python -m pip install pyarrow"
        ) from exc

    parquet_file = pq.ParquetFile(parquet_path)
    column = choose_text_column(
        parquet_file.schema_arrow.names,
        text_column,
    )

    print(f"  Reading Parquet: {parquet_path.name}")
    print(f"  Text column: {column}")

    records_written = 0

    with output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=[column],
        ):
            values = batch.column(0).to_pylist()

            for value in values:
                text = extract_text(value).strip()

                if not text:
                    continue

                # Keep each record separated from the next.
                output_file.write(text)
                output_file.write("\n\n")
                records_written += 1

    if records_written == 0:
        print(
            f"  Warning: no usable text found in "
            f"{parquet_path.name}",
            file=sys.stderr,
        )

    return records_written


def prepare_corpus(
    data_files: list[Path],
    temporary_dir: Path,
    text_column: str | None = None,
) -> list[Path]:
    """Prepare tokenizer-compatible text corpus files."""
    corpus_files = []
    total_records = 0

    for index, path in enumerate(data_files):
        if path.suffix.lower() == ".txt":
            # Tokenizers can read existing text files directly.
            corpus_files.append(path)
            print(f"Using text file: {path.name}")
            continue

        if path.suffix.lower() == ".parquet":
            temporary_text = temporary_dir / f"parquet_{index}.txt"

            records = convert_parquet_to_text(
                parquet_path=path,
                output_path=temporary_text,
                text_column=text_column,
            )

            if records > 0:
                corpus_files.append(temporary_text)
                total_records += records

    if total_records:
        print(f"Parquet text records extracted: {total_records:,}")

    if not corpus_files:
        raise ValueError(
            "No usable training text was found in the input files."
        )

    return corpus_files


def train_wordpiece(
    data_dir: Path,
    output_dir: Path,
    vocab_size: int = 32_000,
    lowercase: bool = False,
    min_frequency: int = 2,
    limit_alphabet: int = 1000,
    text_column: str | None = None,
) -> None:
    """Train and save IronFist1's WordPiece vocabulary."""
    if vocab_size < len(SPECIAL_TOKENS):
        raise ValueError(
            f"vocab_size must be at least {len(SPECIAL_TOKENS)}."
        )

    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1.")

    if limit_alphabet < 1:
        raise ValueError("limit_alphabet must be positive.")

    data_files = find_data_files(data_dir)

    print(f"Found {len(data_files)} supported data files.")
    print(f"Target vocabulary size: {vocab_size:,}")
    print(f"Lowercase normalization: {lowercase}")

    # Temporary converted files are automatically deleted afterward.
    with tempfile.TemporaryDirectory(
        prefix="ironfist1_tokenizer_"
    ) as temporary_name:
        temporary_dir = Path(temporary_name)

        corpus_files = prepare_corpus(
            data_files=data_files,
            temporary_dir=temporary_dir,
            text_column=text_column,
        )

        tokenizer = Tokenizer(
            models.WordPiece(
                unk_token="[UNK]",
                continuing_subword_prefix="##",
            )
        )

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=False,
            lowercase=lowercase,
        )

        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            continuing_subword_prefix="##",
            limit_alphabet=limit_alphabet,
            show_progress=True,
        )

        tokenizer.train(
            files=[str(path) for path in corpus_files],
            trainer=trainer,
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        vocab = tokenizer.get_vocab()
        ordered_tokens = sorted(
            vocab.items(),
            key=lambda item: item[1],
        )

        vocab_path = output_dir / "vocab.txt"

        with vocab_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as file:
            for token, _token_id in ordered_tokens:
                file.write(token + "\n")

        custom_tokenizer = WordPieceTokenizer(
            vocab=dict(ordered_tokens),
            lowercase=lowercase,
            special_tokens=SPECIAL_TOKENS,
        )

        custom_tokenizer.save_config(output_dir)
        tokenizer.save(str(output_dir / "tokenizer.json"))

        print("\nWordPiece training complete.")
        print(f"Actual vocabulary size: {len(vocab):,}")
        print(f"Vocabulary: {vocab_path}")
        print(
            f"Configuration: "
            f"{output_dir / 'tokenizer_config.json'}"
        )
        print(
            f"Backend tokenizer: "
            f"{output_dir / 'tokenizer.json'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train IronFist1's WordPiece tokenizer."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing .txt and .parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the vocabulary will be saved.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32_000,
        help="Target vocabulary size.",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Normalize training text to lowercase.",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum frequency for vocabulary tokens.",
    )
    parser.add_argument(
        "--limit-alphabet",
        type=int,
        default=1000,
        help="Maximum initial alphabet size.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=None,
        help=(
            "Explicit Parquet text column. If omitted, "
            "common column names are detected automatically."
        ),
    )

    args = parser.parse_args()

    try:
        train_wordpiece(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            vocab_size=args.vocab_size,
            lowercase=args.lowercase,
            min_frequency=args.min_frequency,
            limit_alphabet=args.limit_alphabet,
            text_column=args.text_column,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
