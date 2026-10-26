
import torch

from model.discriminator_model import DiscriminatorModel


def main():
    torch.manual_seed(42)

    model = DiscriminatorModel(
        vocab_size=512,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_seq_len=128,
    )

    input_ids = torch.randint(0, 512, (2, 32))

    scores = model(input_ids)

    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()

    # Verify that training gradients can reach the model parameters.
    target_scores = torch.tensor([1.0, -1.0])
    loss = torch.nn.functional.mse_loss(scores, target_scores)
    loss.backward()

    assert model.score_head.weight.grad is not None

    # Test pooling with padding masks.
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    mask[0, 24:] = False

    masked_scores = model(input_ids, attention_mask=mask)

    assert masked_scores.shape == (2,)
    assert torch.isfinite(masked_scores).all()

    print("Scores shape:", tuple(scores.shape))
    print("Training loss:", round(loss.item(), 4))
    print("Masked scoring test passed!")
    print("Discriminator test passed!")


if __name__ == "__main__":
    main()
