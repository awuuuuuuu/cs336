from typing import Iterable, Iterator
import regex as re

from .bpe import special_token_pattern, iter_pretokens


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str]| None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.special_tokens = special_tokens or []

        self.merge_map = {(a, b): i for i, (a, b) in enumerate(merges)}

    def encode(self, text: str) -> list[int]:
        # 1. 预分词
        sep = special_token_pattern(self.special_tokens)
        if self.special_tokens:
            parts = re.split(f"({sep})", text)
        else:
            parts = [text]

        tokens = []
        # 2. 合并
        for part in parts:
            if part in self.special_tokens:
                tokens.append(self.token_to_id[part.encode("utf-8")])
            else:
                for raw in iter_pretokens(part):
                    word = list(raw)
                    INF = float("inf")
                    while len(word) > 1:
                        rank, i = min(
                            (self.merge_map.get((word[j], word[j + 1]), INF), j)
                            for j in range(len(word) - 1)
                        )
                        if rank == INF:
                            break
                        word[i] = word[i] + word[i + 1]
                        del word[i + 1]
                    for token_bytes in word:
                        tokens.append(self.token_to_id[token_bytes])
                    
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return b"".join(self.vocab[t] for t in tokens).decode("utf-8", errors="replace")

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        carry = ""
        for chunk in iterable:
            text = carry + chunk
            i = len(text) - 1
            while i >= 0 and text[i].isspace():
                i -= 1
            yield from self.encode(text[:i + 1])
            carry = text[i + 1:]
        if carry:
            yield from self.encode(carry)

    @classmethod
    def from_files(
        cls,
        vocab_path: str,
        merges_path: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        import json
        with open(vocab_path) as f:
            vocab = {int(k): bytes(v) for k, v in json.load(f).items()}
        with open(merges_path) as f:
            merges = [(bytes(a), bytes(b)) for a, b in json.load(f)]
        return cls(vocab, merges, special_tokens)