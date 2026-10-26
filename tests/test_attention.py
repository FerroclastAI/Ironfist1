
import torch

from model.attention import CausalSelfAttention


def main():
    batch_size = 2
    sequence_length = 128
    hidden_size = 1280
    num_heads = 20

    attention = CausalSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
    )

    x = torch.randn(batch_size, sequence_length, hidden_size)
    output = attention(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()

    # Confirm gradients propagate through attention.
    output.mean().backward()

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(output.shape))
    print("Attention test passed!")


if __name__ == "__main__":
    main()
