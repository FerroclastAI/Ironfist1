
from pathlib import Path

from tokenizer.tokenizer import WordPieceTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOCAB_DIR = PROJECT_ROOT / "tokenizer" / "vocab"


def main():
    tokenizer = WordPieceTokenizer.from_pretrained(VOCAB_DIR)

    text = (
        "IronFist1 is a language model built with "
        "PyTorch, WordPiece, and rStar reasoning."
    )

    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    print("Vocabulary size:", tokenizer.vocab_size)
    print("Original text:", text)
    print("Token IDs:", token_ids)
    print("Decoded text:", decoded)

    assert len(token_ids) > 2, "Encoding produced too few tokens."
    assert token_ids[0] == tokenizer.cls_token_id
    assert token_ids[-1] == tokenizer.sep_token_id
    assert decoded, "Decoded text is empty."

    print("\nTokenizer smoke test passed!")


if __name__ == "__main__":
    main()
