
import torch

from model.embeddings import (
    TokenEmbeddings,
    RotaryPositionalEmbedding,
    apply_rotary_embeddings,
)


def main():
    batch_size = 2
    sequence_length = 128
    vocab_size = 32000
    hidden_size = 1280
    num_heads = 20
    head_dim = hidden_size // num_heads

    # Test token embeddings.
    embedding = TokenEmbeddings(vocab_size, hidden_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    hidden_states = embedding(input_ids)

    assert hidden_states.shape == (
        batch_size,
        sequence_length,
        hidden_size,
    )

    # Test rotary positional embeddings.
    rope = RotaryPositionalEmbedding(head_dim)
    cos, sin = rope(sequence_length)

    assert cos.shape == (sequence_length, head_dim // 2)
    assert sin.shape == cos.shape

    # Apply RoPE to simulated attention queries.
    queries = torch.randn(
        batch_size,
        num_heads,
        sequence_length,
        head_dim,
    )

    rotated = apply_rotary_embeddings(queries, cos, sin)

    assert rotated.shape == queries.shape
    assert torch.isfinite(rotated).all()

    print("Token embedding shape:", tuple(hidden_states.shape))
    print("RoPE shape:", tuple(cos.shape))
    print("Rotated query shape:", tuple(rotated.shape))
    print("Embedding tests passed!")


if __name__ == "__main__":
    main()
