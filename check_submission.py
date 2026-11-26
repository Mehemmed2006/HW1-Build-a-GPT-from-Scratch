import json, csv, pathlib, sys
root = pathlib.Path(".")
need = ["tokenizer.py","model.py","train.py","bigram.py","generate.py","tokenizer.json",
        "runs.csv","samples.txt","git_log.txt","report.pdf","ai_disclosure.md"]
for f in need:
    print("OK     " if (root/f).exists() else "MISSING", f)

print("configs:", sorted(p.name for p in (root/"configs").glob("*.json")))
print("logs   :", sorted(p.name for p in (root/"logs").glob("*.csv")))

bad = ["MultiheadAttention","nn.Transformer","scaled_dot_product_attention",
       "import transformers","tiktoken","sentencepiece","minbpe","from tokenizers","import tokenizers"]
for p in root.glob("*.py"):
    if p.name == pathlib.Path(sys.argv[0]).name:
        continue
    t = p.read_text(encoding="utf-8")
    for b in bad:
        if b in t:
            print("FORBIDDEN", p.name, b)
    if "hw1-build" in t:
        print("TRAP-COMMENT", p.name)

with open("runs.csv", encoding="utf-8") as f:
    print("runs.csv:", [r["run_id"] for r in csv.DictReader(f)])

tok = json.load(open("tokenizer.json", encoding="utf-8"))
print("tokenizer: merges =", len(tok["merges"]), "| vocab_size =", tok["vocab_size"])