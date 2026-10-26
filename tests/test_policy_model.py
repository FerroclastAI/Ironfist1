
import torch

from model.policy_model import PolicyModel


def main():
    torch.manual_seed(42)

    model = PolicyModel(
        vocab_size=512,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_seq_len=128,
    )

    input_ids = torch.randint(0, 512, (2, 32))

    # The data loader's labels are already shifted by one token.
    labels = torch.randint(0, 512, (2, 32))

    logits, loss = model(input_ids, labels=labels)

    assert logits.shape == (2, 32, 512)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    assert model.token_embeddings.embedding.weight.grad is not None

    # Verify that input embeddings and output weights are shared.
    assert (
        model.token_embeddings.embedding.weight
        is model.lm_head.weight
    )

    print("Logits shape:", tuple(logits.shape))
    print("Training loss:", round(loss.item(), 4))
    print(f"Parameters: {model.count_parameters():,}")
    print("Weight tying check passed!")
    print("Policy model test passed!")


if __name__ == "__main__":
    main()
