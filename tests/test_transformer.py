
import torch

from model.transformer import TransformerBlock


def main():
    batch_size = 2
    sequence_length = 128
    hidden_size = 1280

    block = TransformerBlock(
        hidden_size=hidden_size,
        num_attention_heads=20,
        intermediate_size=5120,
    )

    x = torch.randn(
        batch_size,
        sequence_length,
        hidden_size,
        requires_grad=True,
    )

    output = block(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()

    output.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    parameter_count = sum(
        parameter.numel()
        for parameter in block.parameters()
    )

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(output.shape))
    print(f"Block parameters: {parameter_count:,}")
    print("Transformer block test passed!")


if __name__ == "__main__":
    main()
