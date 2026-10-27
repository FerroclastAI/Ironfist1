
"""
IronFist1 PyTorch data loader.

Reads tokenized NumPy shards and returns next-token prediction batches.
"""

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


class TokenShardDataset(IterableDataset):
    """
    Streams token sequences from .npy shards.

    Each sample contains:
        input_ids: tokens [start : start + seq_length]
        labels:    tokens [start + 1 : start + seq_length + 1]
    """

    def __init__(
        self,
        shard_dir: str | Path,
        seq_length: int = 2048,
        stride: int | None = None,
    ):
        super().__init__()

        self.shard_dir = Path(shard_dir)
        self.seq_length = seq_length
        self.stride = seq_length if stride is None else stride

        if self.seq_length <= 0:
            raise ValueError("seq_length must be positive.")

        if self.stride <= 0:
            raise ValueError("stride must be positive.")

        if not self.shard_dir.is_dir():
            raise FileNotFoundError(
                f"Token shard directory not found: {self.shard_dir}"
            )

        self.shard_paths = sorted(
            self.shard_dir.glob("shard_*.npy")
        )

        if not self.shard_paths:
            raise FileNotFoundError(
                f"No shard_*.npy files found in {self.shard_dir}"
            )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = get_worker_info()

        if worker_info is None:
            shard_paths = self.shard_paths
        else:
            # Assign different shards to each worker.
            shard_paths = self.shard_paths[
                worker_info.id::worker_info.num_workers
            ]

        window_size = self.seq_length + 1

        for shard_path in shard_paths:
            tokens = np.load(shard_path, mmap_mode="r")

            if tokens.ndim != 1:
                raise ValueError(
                    f"Expected a one-dimensional token array: {shard_path}"
                )

            if not np.issubdtype(tokens.dtype, np.integer):
                raise TypeError(
                    f"Expected integer token IDs in {shard_path}, "
                    f"got {tokens.dtype}."
                )

            if len(tokens) < window_size:
                continue

            for start in range(
                0,
                len(tokens) - window_size + 1,
                self.stride,
            ):
                # Copy the window so PyTorch owns writable memory.
                window = np.array(
                    tokens[start : start + window_size],
                    dtype=np.int64,
                    copy=True,
                )

                input_ids = torch.from_numpy(window[:-1])
                labels = torch.from_numpy(window[1:])

                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                }


def create_dataloader(
    shard_dir: str | Path,
    seq_length: int = 2048,
    batch_size: int = 2,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    dataset = TokenShardDataset(
        shard_dir=shard_dir,
        seq_length=seq_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
