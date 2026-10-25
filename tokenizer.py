"""Part A: byte-level BPE tokenizer.

"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: list[tuple[int, int]] = []
        self.token_bytes: list[bytes] = [bytes([i]) for i in range(256)]

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        result: list[int] = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    def train(self, text: str, vocab_size: int) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        if not isinstance(vocab_size, int) or not 256 <= vocab_size <= 65536:
            raise ValueError("vocab_size must be an integer from 256 to 65536")

        ids = list(text.encode("utf-8"))
        self.merges = []
        self.token_bytes = [bytes([i]) for i in range(256)]

        for _ in range(vocab_size - 256):
            counts = Counter(zip(ids, ids[1:]))
            if not counts:
                break
            best_count = max(counts.values())
            # Visible A4: smallest a, then smallest b.
            pair = min(pair for pair, count in counts.items() if count == best_count)
            new_id = 256 + len(self.merges)
            self.merges.append(pair)
            self.token_bytes.append(self.token_bytes[pair[0]] + self.token_bytes[pair[1]])
            ids = self._merge(ids, pair, new_id)

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        ids = list(text.encode("utf-8"))
        for rank, pair in enumerate(self.merges):
            ids = self._merge(ids, pair, 256 + rank)
        return ids

    def decode(self, ids: list[int]) -> str:
        data = bytearray()
        for token_id in ids:
            if not isinstance(token_id, int):
                raise TypeError("token IDs must be integers")
            if token_id < 0 or token_id >= len(self.token_bytes):
                raise ValueError(f"token ID out of range: {token_id}")
            data.extend(self.token_bytes[token_id])
        return bytes(data).decode("utf-8")

    def save(self, path: str) -> None:
        payload = {
            "format": "byte-level-bpe",
            "vocab_size": len(self.token_bytes),
            "merges": [[a, b] for a, b in self.merges],
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "byte-level-bpe":
            raise ValueError("Unrecognized tokenizer file format")
        raw_merges = payload.get("merges")
        if not isinstance(raw_merges, list):
            raise ValueError("Invalid merge table")

        tokenizer = cls()
        for index, raw_pair in enumerate(raw_merges):
            if (
                not isinstance(raw_pair, list)
                or len(raw_pair) != 2
                or not all(isinstance(x, int) for x in raw_pair)
            ):
                raise ValueError(f"Invalid merge at index {index}")
            a, b = raw_pair
            new_id = 256 + index
            if a < 0 or b < 0 or a >= new_id or b >= new_id:
                raise ValueError(f"Invalid token reference in merge {index}")
            tokenizer.merges.append((a, b))
            tokenizer.token_bytes.append(tokenizer.token_bytes[a] + tokenizer.token_bytes[b])

        if payload.get("vocab_size") != len(tokenizer.token_bytes):
            raise ValueError("Vocabulary size does not match merge table")
        return tokenizer


def main() -> None:
    dataset_path = Path("data/input.txt")
    if not dataset_path.is_file():
        raise FileNotFoundError("Dataset not found: data/input.txt")

    text = dataset_path.read_text(encoding="utf-8")
    split = int(len(text) * 0.9)
    train_text, validation_text = text[:split], text[split:]

    print(f"Dataset characters: {len(text):,}")
    print(f"Training characters: {len(train_text):,}")
    print(f"Validation characters: {len(validation_text):,}")

    tokenizer = BPETokenizer()
    tokenizer.train(train_text, vocab_size=1024)
    print(f"Vocabulary size: {len(tokenizer.token_bytes)}")
    print(f"Merge count: {len(tokenizer.merges)}")

    sample = "Hello, Shakespeare!\nCafé 🙂"
    assert tokenizer.decode(tokenizer.encode(sample)) == sample
    assert tokenizer.decode(tokenizer.encode(validation_text)) == validation_text

    tokenizer.save("tokenizer.json")
    loaded = BPETokenizer.load("tokenizer.json")
    assert loaded.merges == tokenizer.merges
    assert loaded.decode(loaded.encode(sample)) == sample
    print("Sample round-trip: PASS")
    print("Validation round-trip: PASS")
    print("Save/load test: PASS")
    print("Saved tokenizer to tokenizer.json")


if __name__ == "__main__":
    main()
