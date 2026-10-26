import torch

from model.mlp import SwiGLU


def main():
    batch_size = 2
    sequence_length = 128
    hidden_size = 1280

    mlp = SwiGLU(
        hidden_size=hidden_size,
        intermediate_size=5120,
    )

    x = torch.randn(
        batch_size,
        sequence_length,
        hidden_size,
        requires_grad=True,
    )

    output = mlp(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()

    output.mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(output.shape))
    print("Gradient check passed!")
    print("SwiGLU test passed!")


if __name__ == "__main__":
    main()
