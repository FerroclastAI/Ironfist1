
"""IronFist1 training engine."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def create_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a warmup followed by cosine-decay learning-rate schedule."""

    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative.")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be less than total_steps.")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1.")

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)

        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(
            max((step - warmup_steps) / decay_steps, 0.0),
            1.0,
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, schedule)


class Trainer:
    """Train a causal language model using next-token prediction."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader,
        *,
        device: str | torch.device | None = None,
        scheduler=None,
        checkpoint_dir: str | Path = "checkpoints",
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
        log_interval: int = 10,
        save_interval: int = 1000,
    ):
        if gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be positive."
            )
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive.")
        if log_interval <= 0:
            raise ValueError("log_interval must be positive.")
        if save_interval < 0:
            raise ValueError("save_interval cannot be negative.")

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.scheduler = scheduler

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = (
            gradient_accumulation_steps
        )
        self.use_amp = use_amp and self.device.type == "cuda"
        self.amp_dtype = amp_dtype
        self.log_interval = log_interval
        self.save_interval = save_interval

        if self.use_amp and amp_dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            raise ValueError(
                "amp_dtype must be torch.float16 or torch.bfloat16."
            )

        self.global_step = 0
        self.epochs_completed = 0
        self.loss_history: list[float] = []

        self.scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=self.use_amp and amp_dtype == torch.float16,
        )

    def _autocast(self):
        if not self.use_amp:
            return torch.autocast(
                device_type=self.device.type,
                enabled=False,
            )

        return torch.autocast(
            device_type="cuda",
            dtype=self.amp_dtype,
        )

    def _get_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        input_ids = batch["input_ids"].to(
            self.device,
            dtype=torch.long,
            non_blocking=True,
        )
        labels = batch["labels"].to(
            self.device,
            dtype=torch.long,
            non_blocking=True,
        )

        with self._autocast():
            output = self.model(input_ids, labels=labels)

        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise TypeError(
                "Expected model(input_ids, labels=labels) to return "
                "(logits, loss)."
            )

        _, loss = output

        if loss.ndim != 0:
            raise ValueError("The model loss must be a scalar.")

        if not torch.isfinite(loss.detach()).item():
            raise FloatingPointError(
                f"Non-finite loss detected at step {self.global_step}."
            )

        return loss

    def train_step(
        self,
        batches: list[dict[str, Any]],
    ) -> float:
        """Perform one optimizer update using accumulated mini-batches."""

        if not batches:
            raise ValueError("At least one batch is required.")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        accumulation_count = len(batches)

        for batch in batches:
            loss = self._get_loss(batch)
            scaled_loss = loss / accumulation_count

            self.scaler.scale(scaled_loss).backward()
            total_loss += loss.detach().float().item()

        self.scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm,
        )

        if not torch.isfinite(torch.as_tensor(grad_norm)).item():
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"Non-finite gradient norm at step {self.global_step}."
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1

        mean_loss = total_loss / accumulation_count
        self.loss_history.append(mean_loss)

        return mean_loss

    def save_checkpoint(self, filename: str | None = None) -> Path:
        """Save model, optimizer, scheduler, scaler, and progress."""

        filename = filename or f"step_{self.global_step:08d}.pt"
        path = self.checkpoint_dir / filename
        temporary_path = path.with_suffix(path.suffix + ".tmp")

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "epochs_completed": self.epochs_completed,
            "loss_history": self.loss_history[-1000:],
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }

        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)

        print(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        restore_rng: bool = True,
    ) -> None:
        """Restore training state from a trusted checkpoint."""

        checkpoint = torch.load(
            Path(path),
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        scheduler_state = checkpoint.get("scheduler")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        scaler_state = checkpoint.get("scaler")
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

        self.global_step = checkpoint.get("global_step", 0)
        self.epochs_completed = checkpoint.get("epochs_completed", 0)
        self.loss_history = checkpoint.get("loss_history", [])

        if restore_rng:
            rng_state = checkpoint.get("torch_rng_state")
            if rng_state is not None:
                torch.set_rng_state(rng_state.cpu())

            cuda_states = checkpoint.get("cuda_rng_states")
            if cuda_states is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_states)

        print(f"Checkpoint restored: {path}")

    def fit(
        self,
        *,
        max_steps: int,
        max_epochs: int | None = None,
    ) -> list[float]:
        """Train until max_steps or max_epochs is reached."""

        if max_steps <= self.global_step:
            raise ValueError(
                "max_steps must be greater than the current global_step."
            )
        if max_epochs is not None and max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")

        pending_batches = []
        pending_target = self.gradient_accumulation_steps
        started_at = time.time()
        last_log_time = started_at

        while self.global_step < max_steps:
            self.model.train()
            batches_seen = 0

            for batch in self.train_loader:
                pending_batches.append(batch)
                batches_seen += 1

                if len(pending_batches) < pending_target:
                    continue

                loss = self.train_step(pending_batches)
                pending_batches = []

                if self.global_step % self.log_interval == 0:
                    now = time.time()
                    elapsed = now - last_log_time
                    last_log_time = now

                    print(
                        f"step={self.global_step} "
                        f"loss={loss:.4f} "
                        f"lr={self.optimizer.param_groups[0]['lr']:.3e} "
                        f"elapsed={elapsed:.1f}s"
                    )

                if (
                    self.save_interval > 0
                    and self.global_step % self.save_interval == 0
                ):
                    self.save_checkpoint()

                if self.global_step >= max_steps:
                    break

            if pending_batches and self.global_step < max_steps:
                loss = self.train_step(pending_batches)
                pending_batches = []

                print(
                    f"step={self.global_step} "
                    f"loss={loss:.4f} (partial accumulation)"
                )

            self.epochs_completed += 1

            if batches_seen == 0:
                raise RuntimeError(
                    "The training loader produced no batches. "
                    "Check the dataset, batch size, and drop_last setting."
                )

            if max_epochs is not None and self.epochs_completed >= max_epochs:
                break

        self.save_checkpoint("latest.pt")

        print(
            f"Training finished: step={self.global_step}, "
            f"epochs={self.epochs_completed}, "
            f"elapsed={time.time() - started_at:.1f}s"
        )

        return self.loss_history
