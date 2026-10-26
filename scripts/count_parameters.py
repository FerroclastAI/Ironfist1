
"""
IronFist1 parameter-counting utility.

Calculates the expected number of parameters without instantiating
the full model, which is useful when working on limited hardware.
"""


def count_policy_parameters(
    vocab_size: int = 32000,
    hidden_size: int = 1280,
    num_layers: int = 20,
    num_attention_heads: int = 20,
    intermediate_size: int = 4096,
    tie_word_embeddings: bool = True,
    use_bias: bool = False,
) -> int:
    """Estimate the exact parameter count for the current PolicyModel."""

    dimensions = {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "intermediate_size": intermediate_size,
    }

    for name, value in dimensions.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "hidden_size must be divisible by num_attention_heads."
        )

    # Token embedding matrix.
    parameters = vocab_size * hidden_size

    # One Transformer block:
    # Q, K, V projections plus attention output projection.
    attention = 4 * hidden_size**2

    # SwiGLU: gate, up, and down projections.
    mlp = 3 * hidden_size * intermediate_size

    # Two RMSNorm weights per block.
    normalization = 2 * hidden_size

    if use_bias:
        attention += 4 * hidden_size
        mlp += 2 * intermediate_size + hidden_size

    parameters += num_layers * (
        attention + mlp + normalization
    )

    # Final RMSNorm.
    parameters += hidden_size

    # Untied output head adds another vocabulary projection.
    if not tie_word_embeddings:
        parameters += vocab_size * hidden_size

    return parameters


def main() -> None:
    settings = {
        "vocab_size": 32000,
        "hidden_size": 1280,
        "num_layers": 20,
        "num_attention_heads": 20,
        "intermediate_size": 4096,
        "tie_word_embeddings": True,
        "use_bias": False,
    }

    total = count_policy_parameters(**settings)

    print("IronFist1 Policy Model")
    print("-" * 32)

    for name, value in settings.items():
        print(f"{name}: {value}")

    print("-" * 32)
    print(f"Total parameters: {total:,}")
    print(f"Total in millions: {total / 1_000_000:.2f}M")
    print(f"Total in billions: {total / 1_000_000_000:.4f}B")

    target = 500_000_000
    difference = total - target

    print(f"Target parameters: {target:,}")
    print(f"Difference from target: {difference:+,}")


if __name__ == "__main__":
    main()
