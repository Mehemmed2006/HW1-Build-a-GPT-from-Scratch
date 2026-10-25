"""A1-A6 checks for tokenizer.py.

Run from the project folder:
    python test_tokenizer_a1_a6.py

This test file does not modify tokenizer.py or tokenizer.json.
"""
from pathlib import Path
from tokenizer import BPETokenizer


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main():
    # A1: 256 base byte tokens, with IDs equal to byte values.
    tok = BPETokenizer()
    check(len(tok.token_bytes) == 256, "A1: initial vocabulary has 256 tokens")
    check(all(tok.token_bytes[i] == bytes([i]) for i in range(256)),
          "A1: base token ID b represents byte b")
    sample = "Salam, dünya! Café 🙂"
    check(tok.decode(tok.encode(sample)) == sample,
          "A1: UTF-8 round-trip works for Unicode")

    # A2: no pre-tokenization; merges may include a space and adjacent text.
    tok_space = BPETokenizer()
    tok_space.train(" a a a ", vocab_size=257)
    check((32, 97) in tok_space.merges,
          "A2: training can merge across whitespace/text boundaries (no pre-tokenization)")

    # A3: overlapping pair occurrences are merged left-to-right without overlap.
    tok_overlap = BPETokenizer()
    tok_overlap.train("aaa", vocab_size=257)
    check(tok_overlap.merges == [(97, 97)],
          "A3: most frequent pair is selected in the overlap test")
    check(tok_overlap.encode("aaa") == [256, 97],
          "A3: overlapping 'aa' in 'aaa' merges once, left-to-right")
    check(tok_overlap.decode(tok_overlap.encode("aaa")) == "aaa",
          "A3: overlap test round-trip works")

    # A4: ties choose the lexicographically smallest (a, b) token-ID pair.
    tok_tie = BPETokenizer()
    tok_tie.train("abac", vocab_size=257)
    check(tok_tie.merges[0] == (97, 98),
          "A4: tie selects smallest (a, b) pair; not first pair by text position")

    # A5: run the real dataset training entry point's expected configuration.
    dataset = Path("data/input.txt")
    check(dataset.is_file(), "A5: data/input.txt exists")
    text = dataset.read_text(encoding="utf-8")
    train_text = text[: int(len(text) * 0.9)]
    tok_full = BPETokenizer()
    tok_full.train(train_text, vocab_size=1024)
    check(len(tok_full.merges) == 768,
          f"A5: exactly 768 merges (actual: {len(tok_full.merges)})")
    check(len(tok_full.token_bytes) == 1024,
          f"A5: vocabulary is exactly 1024 tokens (actual: {len(tok_full.token_bytes)})")

    # A6: encode matches an explicit replay of the learned merges in order.
    test_text = "To be, or not to be — café 🙂"
    ids = list(test_text.encode("utf-8"))
    for rank, pair in enumerate(tok_full.merges):
        ids = BPETokenizer._merge(ids, pair, 256 + rank)
    check(tok_full.encode(test_text) == ids,
          "A6: encode applies learned merges in training order")
    check(tok_full.decode(tok_full.encode(test_text)) == test_text,
          "A6: round-trip works after full training")

    print("\nAll A1-A6 checks passed.")


if __name__ == "__main__":
    main()
