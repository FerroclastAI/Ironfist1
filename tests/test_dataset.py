
from pathlib import Path

import torch

from training.data_loader import create_dataloader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARD_DIR = PROJECT_ROOT / "data" / "tokenized"


def main():
    loader = create_dataloader(
        shard_dir=SHARD_DIR,
        seq_length=128,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )

    batch = next(iter(loader))

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    assert input_ids.shape == (2, 128)
    assert labels.shape == (2, 128)
    assert input_ids.dtype == torch.int64
    assert labels.dtype == torch.int64

    # For next-token prediction, labels are shifted by one token.
    print("Input shape:", tuple(input_ids.shape))
    print("Labels shape:", tuple(labels.shape))
    print("First input tokens:", input_ids[0, :10].tolist())
    print("First target tokens:", labels[0, :10].tolist())
    print("\nDataset loader test passed!")


if __name__ == "__main__":
    main()
