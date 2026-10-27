
"""Pretrain the IronFist1 policy language model."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from model.policy_model import PolicyModel
from training.data_loader import create_dataloader
from training.trainer import Trainer, create_scheduler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain the IronFist1 policy model."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/tokenized"),
        help="Directory containing shard_*.npy files.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/policy"),
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Small overrides make integration testing possible.
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--hidden-size", type=int, default=1280)
    parser.add_argument("--num-layers", type=int, default=20)
    parser.add_argument("--num-heads", type=int, default=20)
    parser.add_argument("--intermediate-size", type=int, default=4096)
    parser.add_argument("--max-seq-len", type=int, default=2048)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA GPU."
        )

    return torch.device(requested)


def build_model(args) -> PolicyModel:
    """Construct the policy model using its supported constructor."""

    return PolicyModel(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_seq_len=args.max_seq_len,
    )


def main() -> None:
    args = parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.seq_length <= 0:
        raise ValueError("--seq-length must be positive.")
    if args.seq_length > args.max_seq_len:
        raise ValueError(
            "--seq-length cannot exceed --max-seq-len."
        )
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative.")
    if args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        raise ValueError(
            "--warmup-steps must be non-negative and less than --steps."
        )

    set_seed(args.seed)
    device = select_device(args.device)

    print("=" * 60)
    print("IronFist1 policy pretraining")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset: {args.data_dir}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Maximum steps: {args.steps}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    loader = create_dataloader(
        shard_dir=args.data_dir,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Trainable parameters: {parameter_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    scheduler = create_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    amp_dtype = (
        torch.bfloat16
        if args.amp_dtype == "bfloat16"
        else torch.float16
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=args.checkpoint_dir,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=not args.no_amp,
        amp_dtype=amp_dtype,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )

    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {args.resume}"
            )
        trainer.load_checkpoint(args.resume)

    trainer.fit(
        max_steps=args.steps,
        max_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
