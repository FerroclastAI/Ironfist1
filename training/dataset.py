
"""
IronFist1 Dataset Pipeline.

Reads .txt and .parquet files, tokenizes text using the trained
WordPiece vocabulary, and stores fixed-length token IDs in shards.

Each shard is a NumPy array of int32 token IDs.
Training examples are contiguous sequences of seq_length + 1 tokens:
    input_ids = tokens[:-1]
    labels    = tokens[1:]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:
    from tokenizer.tokenizer import WordPieceTokenizer
except ImportError:
    # Supports running this file directly from the project root.
    PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT_FALLBACK) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))
    from tokenizer.tokenizer import WordPieceTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "tokenizer" / "vocab"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "tokenized"

TEXT_COLUMNS = [
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


def extract_text(value: Any) -> str:
    """Extract textual content from strings or nested chat records."""
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, dict):
        parts = []

        for key in (
            "text",
            "content",
            "value",
            "message",
            "messages",
            "prompt",
            "completion",
            "response",
        ):
            if key in value:
                part = extract_text(value[key])
                if part:
                    parts.append(part)

        return "\n".join(parts)

    if isinstance(value, (list, tuple)):
        parts = [extract_text(item) for item in value]
        return "\n".join(part for part in parts if part)

    return ""


def find_data_files(input_dir: Path) -> list[Path]:
    """Find all supported data files recursively."""
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".txt", ".parquet"}
    )

    if not files:
        raise FileNotFoundError(
            f"No .txt or .parquet files found in {input_dir}"
        )

    return files


def choose_text_column(
    column_names: list[str],
    requested_column: str | None,
) -> str:
    """Choose the text column in a Parquet table."""
    if requested_column is not None:
        if requested_column not in column_names:
            raise ValueError(
                f"Column '{requested_column}' not found. "
                f"Available columns: {column_names}"
            )
        return requested_column

    lookup = {name.lower(): name for name in column_names}

    for candidate in TEXT_COLUMNS:
        if candidate in lookup:
            return lookup[candidate]

    raise ValueError(
        f"Could not find a text column in {column_names}. "
        "Specify one with --text-column."
    )


def iter_text_records(
    files: list[Path],
    text_column: str | None = None,
    parquet_batch_size: int = 1000,
) -> Iterator[str]:
    """
    Yield text records from text and Parquet files.

    Parquet is streamed in record batches to avoid loading entire
    datasets into memory.
    """
    for path in files:
        if path.suffix.lower() == ".txt":
            with path.open("r", encoding="utf-8", errors="replace") as file:
                for line in file:
                    text = line.strip()
                    if text:
                        yield text
            continue

        if path.suffix.lower() == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError(
                    "Install Parquet support with: "
                    "python -m pip install pyarrow"
                ) from exc

            parquet_file = pq.ParquetFile(path)
            column = choose_text_column(
                parquet_file.schema_arrow.names,
                text_column,
            )

            print(f"Reading {path.name}, column '{column}'")

            for batch in parquet_file.iter_batches(
                batch_size=parquet_batch_size,
                columns=[column],
            ):
                for value in batch.column(0).to_pylist():
                    text = extract_text(value).strip()
                    if text:
                        yield text


def flush_shard(
    token_buffer: list[int],
    output_dir: Path,
    shard_index: int,
) -> int:
    """Save the current token buffer as an int32 NumPy shard."""
    if not token_buffer:
        return 0

    shard_path = output_dir / f"shard_{shard_index:05d}.npy"

    array = np.asarray(token_buffer, dtype=np.int32)
    np.save(shard_path, array)

    return len(array)


def prepare_dataset(
    input_dir: Path = DEFAULT_INPUT_DIR,
    tokenizer_dir: Path = DEFAULT_TOKENIZER_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seq_length: int = 2048,
    tokens_per_shard: int = 10_000_000,
    text_column: str | None = None,
    parquet_batch_size: int = 1000,
) -> None:
    """Tokenize source text and write reusable token shards."""
    if seq_length <= 0:
        raise ValueError("seq_length must be positive.")

    if tokens_per_shard < seq_length + 1:
        raise ValueError(
            "tokens_per_shard must be at least seq_length + 1."
        )

    if parquet_batch_size <= 0:
        raise ValueError("parquet_batch_size must be positive.")

    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(
            f"Tokenizer directory not found: {tokenizer_dir}"
        )

    files = find_data_files(input_dir)
    tokenizer = WordPieceTokenizer.from_pretrained(tokenizer_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a fresh output folder so old shards cannot accidentally
    # be mixed with the new dataset.
    for old_shard in output_dir.glob("shard_*.npy"):
        old_shard.unlink()

    token_buffer: list[int] = []
    shard_index = 0
    total_tokens = 0
    total_records = 0
    total_shard_tokens = 0

    print(f"Found {len(files)} input files.")
    print(f"Tokenizer vocabulary: {tokenizer.vocab_size:,}")
    print(f"Sequence length: {seq_length:,}")
    print(f"Tokens per shard: {tokens_per_shard:,}")

    for text in iter_text_records(
        files=files,
        text_column=text_column,
        parquet_batch_size=parquet_batch_size,
    ):
        # Add sequence boundary markers to every document.
        ids = tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
        )

        if not ids:
            continue

        token_buffer.extend(ids)
        total_tokens += len(ids)
        total_records += 1

        while len(token_buffer) >= tokens_per_shard:
            shard_tokens = token_buffer[:tokens_per_shard]
            del token_buffer[:tokens_per_shard]

            total_shard_tokens += flush_shard(
                shard_tokens,
                output_dir,
                shard_index,
            )
            shard_index += 1

        if total_records % 10_000 == 0:
            print(
                f"Processed {total_records:,} records; "
                f"{total_tokens:,} tokens."
            )

    if token_buffer:
        total_shard_tokens += flush_shard(
            token_buffer,
            output_dir,
            shard_index,
        )
        shard_index += 1

    if total_shard_tokens < seq_length + 1:
        raise ValueError(
            "Not enough tokens to create even one training sequence. "
            "Check the dataset and tokenizer."
        )

    metadata = {
        "format": "numpy_int32_token_shards",
        "tokenizer_type": "wordpiece",
        "vocab_size": tokenizer.vocab_size,
        "seq_length": seq_length,
        "tokens_per_shard": tokens_per_shard,
        "total_records": total_records,
        "total_tokens": total_tokens,
        "saved_tokens": total_shard_tokens,
        "shard_count": shard_index,
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nDataset preparation complete.")
    print(f"Records processed: {total_records:,}")
    print(f"Tokens generated: {total_tokens:,}")
    print(f"Tokens saved: {total_shard_tokens:,}")
    print(f"Shards created: {shard_index}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata: {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare IronFist1's tokenized training dataset."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=DEFAULT_TOKENIZER_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--tokens-per-shard",
        type=int,
        default=10_000_000,
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    try:
        prepare_dataset(
            input_dir=args.input_dir,
            tokenizer_dir=args.tokenizer_dir,
            output_dir=args.output_dir,
            seq_length=args.seq_length,
            tokens_per_shard=args.tokens_per_shard,
            text_column=args.text_column,
            parquet_batch_size=args.parquet_batch_size,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
