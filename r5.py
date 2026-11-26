from tokenizer import BPETokenizer

tok = BPETokenizer.load("tokenizer.json")

print("First 10 merges:")
for k in range(10):
    a, b = tok.merges[k]
    print(k, (a, b), repr(tok.token_bytes[256 + k].decode("utf-8", errors="replace")))

text = open("data/input.txt", encoding="utf-8").read()
val_text = text[int(len(text) * 0.9):]
n_bytes = len(val_text.encode("utf-8"))
n_tokens = len(tok.encode(val_text))
print("val bytes:", n_bytes, "| val tokens:", n_tokens, "| bytes per token:", n_bytes / n_tokens)