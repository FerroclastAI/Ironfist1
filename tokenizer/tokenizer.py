
"""
IronFist1 WordPiece Tokenizer.

Provides WordPiece tokenization, encoding, decoding, batching,
padding, truncation, and vocabulary persistence.
"""

import json
import re
from pathlib import Path
from typing import Optional, Sequence, Union


class WordPieceTokenizer:
    DEFAULT_SPECIAL_TOKENS = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
    ]

    def __init__(
        self,
        vocab: Optional[Union[dict[str, int], Sequence[str]]] = None,
        lowercase: bool = False,
        max_input_chars_per_word: int = 100,
        special_tokens: Optional[Sequence[str]] = None,
    ):
        self.lowercase = lowercase
        self.max_input_chars_per_word = max_input_chars_per_word
        self.special_tokens = list(
            special_tokens or self.DEFAULT_SPECIAL_TOKENS
        )

        if max_input_chars_per_word <= 0:
            raise ValueError("max_input_chars_per_word must be positive.")

        if vocab is None:
            self.vocab = {
                token: index
                for index, token in enumerate(self.special_tokens)
            }
        elif isinstance(vocab, dict):
            self.vocab = dict(vocab)
        else:
            self.vocab = {
                token: index for index, token in enumerate(vocab)
            }

        self._validate_vocab()

        self.ids_to_tokens = {
            index: token for token, index in self.vocab.items()
        }

        for token in self.DEFAULT_SPECIAL_TOKENS:
            if token not in self.vocab:
                raise ValueError(f"Missing special token: {token}")

        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"

        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.cls_token_id = self.vocab[self.cls_token]
        self.sep_token_id = self.vocab[self.sep_token]
        self.mask_token_id = self.vocab[self.mask_token]

        self._basic_pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _validate_vocab(self) -> None:
        if not self.vocab:
            raise ValueError("Vocabulary cannot be empty.")

        ids = list(self.vocab.values())

        if any(not isinstance(i, int) or i < 0 for i in ids):
            raise ValueError("Vocabulary IDs must be nonnegative integers.")

        if len(set(ids)) != len(ids):
            raise ValueError("Vocabulary IDs must be unique.")

        if set(ids) != set(range(len(ids))):
            raise ValueError("Vocabulary IDs must be contiguous from zero.")

        if "[UNK]" not in self.vocab:
            raise ValueError("Vocabulary must contain [UNK].")

    def tokenize(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if self.lowercase:
            text = text.lower()

        basic_tokens = self._basic_pattern.findall(text)
        pieces = []

        for token in basic_tokens:
            if token in self.special_tokens and token in self.vocab:
                pieces.append(token)
            else:
                pieces.extend(self._wordpiece(token))

        return pieces

    def _wordpiece(self, word: str) -> list[str]:
        """Greedy longest-match-first WordPiece segmentation."""
        if len(word) > self.max_input_chars_per_word:
            return [self.unk_token]

        if word in self.vocab:
            return [word]

        pieces = []
        start = 0

        while start < len(word):
            end = len(word)
            matched_piece = None

            while start < end:
                substring = word[start:end]

                if start > 0:
                    substring = "##" + substring

                if substring in self.vocab:
                    matched_piece = substring
                    break

                end -= 1

            if matched_piece is None:
                return [self.unk_token]

            pieces.append(matched_piece)
            start = end

        return pieces

    def convert_tokens_to_ids(
        self, tokens: Sequence[str]
    ) -> list[int]:
        return [
            self.vocab.get(token, self.unk_token_id)
            for token in tokens
        ]

    def convert_ids_to_tokens(
        self, token_ids: Sequence[int]
    ) -> list[str]:
        return [
            self.ids_to_tokens.get(int(token_id), self.unk_token)
            for token_id in token_ids
        ]

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> list[int]:
        tokens = self.tokenize(text)

        if add_special_tokens:
            tokens = [self.cls_token] + tokens + [self.sep_token]

        token_ids = self.convert_tokens_to_ids(tokens)

        if max_length is not None:
            if max_length < 0:
                raise ValueError("max_length cannot be negative.")

            if len(token_ids) > max_length:
                if not truncation:
                    raise ValueError(
                        "Sequence exceeds max_length. "
                        "Enable truncation to shorten it."
                    )

                token_ids = token_ids[:max_length]

                if add_special_tokens and max_length >= 1:
                    token_ids[-1] = self.sep_token_id

        return token_ids

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        tokens = self.convert_ids_to_tokens(token_ids)
        output = []

        for token in tokens:
            if skip_special_tokens and token in self.special_tokens:
                continue

            if token.startswith("##"):
                if output:
                    output[-1] += token[2:]
                else:
                    output.append(token[2:])
            else:
                output.append(token)

        text = " ".join(output)
        text = re.sub(r"\s+([.,!?;:%)\]}])", r"\1", text)
        text = re.sub(r"([( \[{])\s+", r"\1", text)

        return text.strip()

    def encode_plus(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        truncation: bool = False,
    ) -> dict[str, list[int]]:
        if padding not in (False, True, "max_length"):
            raise ValueError(
                "padding must be False, True, or 'max_length'."
            )

        if padding and max_length is None:
            raise ValueError(
                "max_length is required when padding is enabled."
            )

        token_ids = self.encode(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=truncation,
        )

        attention_mask = [1] * len(token_ids)

        if padding:
            # Explicit check also makes max_length safe for Pylance.
            if max_length is None:
                raise ValueError("max_length is required for padding.")

            padding_length = max_length - len(token_ids)

            if padding_length < 0:
                raise ValueError(
                    "Sequence exceeds max_length. Enable truncation."
                )

            token_ids.extend([self.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)

        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
        }

    def batch_encode(
        self,
        texts: Sequence[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> dict[str, list[list[int]]]:
        if not texts:
            return {"input_ids": [], "attention_mask": []}

        encoded = [
            self.encode(
                text,
                add_special_tokens=True,
                max_length=max_length,
                truncation=truncation,
            )
            for text in texts
        ]

        if padding:
            target_length = max(len(ids) for ids in encoded)

            if max_length is not None:
                target_length = min(target_length, max_length)

            encoded = [
                ids[:target_length] for ids in encoded
            ]

            masks = [
                [1] * len(ids)
                + [0] * (target_length - len(ids))
                for ids in encoded
            ]

            encoded = [
                ids + [self.pad_token_id] * (target_length - len(ids))
                for ids in encoded
            ]
        else:
            masks = [[1] * len(ids) for ids in encoded]

        return {
            "input_ids": encoded,
            "attention_mask": masks,
        }

    def save_vocabulary(self, directory: Union[str, Path]) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        vocab_path = directory / "vocab.txt"

        ordered_tokens = [
            self.ids_to_tokens[index]
            for index in range(self.vocab_size)
        ]

        vocab_path.write_text(
            "\n".join(ordered_tokens) + "\n",
            encoding="utf-8",
        )

        return vocab_path

    def save_config(self, directory: Union[str, Path]) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        config_path = directory / "tokenizer_config.json"

        config = {
            "tokenizer_type": "wordpiece",
            "lowercase": self.lowercase,
            "max_input_chars_per_word": self.max_input_chars_per_word,
            "special_tokens": self.special_tokens,
        }

        config_path.write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )

        return config_path

    def save_pretrained(self, directory: Union[str, Path]) -> None:
        self.save_vocabulary(directory)
        self.save_config(directory)

    @classmethod
    def from_pretrained(
        cls, directory: Union[str, Path]
    ) -> "WordPieceTokenizer":
        directory = Path(directory)
        vocab_path = directory / "vocab.txt"
        config_path = directory / "tokenizer_config.json"

        if not vocab_path.is_file():
            raise FileNotFoundError(
                f"Vocabulary not found: {vocab_path}"
            )

        tokens = vocab_path.read_text(
            encoding="utf-8"
        ).splitlines()

        config = {}

        if config_path.is_file():
            config = json.loads(
                config_path.read_text(encoding="utf-8")
            )

        return cls(
            vocab=tokens,
            lowercase=config.get("lowercase", False),
            max_input_chars_per_word=config.get(
                "max_input_chars_per_word", 100
            ),
            special_tokens=config.get(
                "special_tokens", cls.DEFAULT_SPECIAL_TOKENS
            ),
        )
