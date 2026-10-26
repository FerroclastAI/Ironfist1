
import torch

from reasoning.reward import RewardConfig, calculate_reward


def main():
    config = RewardConfig(
        discriminator_weight=1.0,
        consistency_weight=0.5,
        penalty_weight=1.0,
    )

    reward = calculate_reward(
        discriminator_score=0.8,
        consistency_score=0.6,
        penalty=0.1,
        config=config,
    )

    # 0.8 + (0.5 * 0.6) - 0.1 = 1.0
    assert abs(reward - 1.0) < 1e-6

    tensor_reward = calculate_reward(
        discriminator_score=torch.tensor(0.8),
        consistency_score=torch.tensor(0.6),
        penalty=torch.tensor(0.1),
        config=config,
    )

    assert isinstance(tensor_reward, torch.Tensor)
    assert torch.isfinite(tensor_reward)
    assert abs(tensor_reward.item() - 1.0) < 1e-6

    print("Example reward:", reward)
    print("Tensor reward:", tensor_reward.item())
    print("Reward tests passed!")


if __name__ == "__main__":
    main()
