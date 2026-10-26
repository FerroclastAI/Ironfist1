
"""
IronFist1 reward calculation for rStar search.

Combines discriminator quality scores, consistency scores,
and optional penalties into a scalar reward.
"""

from dataclasses import dataclass

import torch


@dataclass
class RewardConfig:
    """Weights used to combine reward components."""

    discriminator_weight: float = 1.0
    consistency_weight: float = 0.5
    penalty_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.discriminator_weight < 0:
            raise ValueError("discriminator_weight must be non-negative.")

        if self.consistency_weight < 0:
            raise ValueError("consistency_weight must be non-negative.")

        if self.penalty_weight < 0:
            raise ValueError("penalty_weight must be non-negative.")


def calculate_reward(
    discriminator_score: float | torch.Tensor,
    consistency_score: float | torch.Tensor = 0.0,
    penalty: float | torch.Tensor = 0.0,
    config: RewardConfig | None = None,
) -> float | torch.Tensor:
    """
    Calculate a scalar reward.

    Formula:
        reward = quality_weight * quality
               + consistency_weight * consistency
               - penalty_weight * penalty

    Scores should use a documented, consistent scale. This function
    does not normalize or calibrate model outputs automatically.
    """

    if config is None:
        config = RewardConfig()

    values = (
        discriminator_score,
        consistency_score,
        penalty,
    )

    tensor_values = [
        value for value in values if isinstance(value, torch.Tensor)
    ]

    if tensor_values:
        reference = tensor_values[0]

        if reference.numel() != 1:
            raise ValueError("Each reward component must be a scalar.")

        device = reference.device
        dtype = (
            reference.dtype
            if reference.is_floating_point()
            else torch.float32
        )

        def as_scalar_tensor(value):
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(
                        "Each reward component must be a scalar."
                    )
                if value.device != device:
                    raise ValueError(
                        "All tensor reward components must share a device."
                    )
                return value.reshape(()).to(dtype=dtype)

            return torch.tensor(value, device=device, dtype=dtype)

        quality = as_scalar_tensor(discriminator_score)
        consistency = as_scalar_tensor(consistency_score)
        penalty_value = as_scalar_tensor(penalty)

        if not all(
            torch.isfinite(value).all().item()
            for value in (quality, consistency, penalty_value)
        ):
            raise ValueError("Reward components must be finite.")

        return (
            config.discriminator_weight * quality
            + config.consistency_weight * consistency
            - config.penalty_weight * penalty_value
        )

    quality = float(discriminator_score)
    consistency = float(consistency_score)
    penalty_value = float(penalty)

    import math

    if not all(
        math.isfinite(value)
        for value in (quality, consistency, penalty_value)
    ):
        raise ValueError("Reward components must be finite.")

    return (
        config.discriminator_weight * quality
        + config.consistency_weight * consistency
        - config.penalty_weight * penalty_value
    )
