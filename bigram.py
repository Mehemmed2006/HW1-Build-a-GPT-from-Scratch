import csv
from pathlib import Path

import torch
from tokenizer import BPETokenizer


def main():
    train_path = Path("data/train_ids.pt")
    val_path = Path("data/validation_ids.pt")
    runs_path = Path("runs.csv")

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Token ID cache files were not found. Run the training pipeline first."
        )

    if not runs_path.exists():
        raise FileNotFoundError(
            "runs.csv was not found. Run the training pipeline first."
        )

    tokenizer = BPETokenizer.load("tokenizer.json")
    vocab_size = len(tokenizer.token_bytes)

    train_ids = torch.load(
        train_path, map_location="cpu", weights_only=True
    ).long()

    val_ids = torch.load(
        val_path, map_location="cpu", weights_only=True
    ).long()

    if train_ids.numel() < 2 or val_ids.numel() < 2:
        raise ValueError("Training and validation data must contain at least 2 tokens.")

    if train_ids.min() < 0 or train_ids.max() >= vocab_size:
        raise ValueError("Training token IDs are outside the vocabulary range.")

    if val_ids.min() < 0 or val_ids.max() >= vocab_size:
        raise ValueError("Validation token IDs are outside the vocabulary range.")

    # Count consecutive token pairs in the training data.
    pair_ids = (
        train_ids[:-1] * vocab_size + train_ids[1:]
    )

    counts = torch.bincount(
        pair_ids,
        minlength=vocab_size * vocab_size
    ).reshape(vocab_size, vocab_size)

    # Add-one smoothing.
    context_counts = counts.sum(dim=1)

    # Calculate validation negative log-likelihood in nats per token.
    previous_tokens = val_ids[:-1]
    next_tokens = val_ids[1:]

    log_probs = (
        torch.log(counts[previous_tokens, next_tokens].double() + 1)
        - torch.log(context_counts[previous_tokens].double() + vocab_size)
    )

    val_loss = -log_probs.mean().item()

    print(f"Bigram validation loss: {val_loss:.6f} nats per token")

    # Add the Bigram result to runs.csv.
    with runs_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or "run_id" not in fieldnames:
        raise ValueError("runs.csv has an invalid header.")

    if "best_val_loss" not in fieldnames:
        raise ValueError("runs.csv does not contain the best_val_loss column.")

    # Avoid duplicate Bigram rows if the script is run again.
    rows = [row for row in rows if row["run_id"] != "bigram"]

    bigram_row = {column: "" for column in fieldnames}
    bigram_row["run_id"] = "bigram"
    bigram_row["best_val_loss"] = f"{val_loss:.8f}"

    rows.append(bigram_row)

    with runs_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Bigram result added to runs.csv")


if __name__ == "__main__":
    main()