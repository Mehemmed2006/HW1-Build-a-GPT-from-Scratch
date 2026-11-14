
"""Part C: GPT training pipeline."""

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer


# ==================================================
# 1. CONFIGURATION
# ==================================================

REQUIRED_KEYS = [
    "run_id",
    "n_layer",
    "n_head",
    "n_embd",
    "block_size",
    "dropout",
    "batch_size",
    "max_steps",
    "max_lr",
    "warmup_steps",
    "eval_interval",
    "seed",
    "ablation",
]

def load_config(config_path: str) -> dict:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    missing = [
        key for key in REQUIRED_KEYS
        if key not in config
    ]

    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    integer_keys = [
        "n_layer",
        "n_head",
        "n_embd",
        "block_size",
        "batch_size",
        "max_steps",
        "warmup_steps",
        "eval_interval",
        "seed",
    ]

    for key in integer_keys:
        if not isinstance(config[key], int) or isinstance(config[key], bool):
            raise ValueError(f"{key} must be an integer.")

    if config["n_layer"] <= 0:
        raise ValueError("n_layer must be positive.")

    if config["n_head"] <= 0:
        raise ValueError("n_head must be positive.")

    if config["n_embd"] <= 0:
        raise ValueError("n_embd must be positive.")

    if config["n_embd"] % config["n_head"] != 0:
        raise ValueError("n_embd must be divisible by n_head.")

    if config["block_size"] <= 0:
        raise ValueError("block_size must be positive.")

    if config["batch_size"] <= 0:
        raise ValueError("batch_size must be positive.")

    if config["max_steps"] <= 0:
        raise ValueError("max_steps must be positive.")

    if not 0 <= config["warmup_steps"] < config["max_steps"]:
        raise ValueError("Invalid warmup_steps.")

    if config["eval_interval"] <= 0:
        raise ValueError("eval_interval must be positive.")

    if config["max_lr"] <= 0:
        raise ValueError("max_lr must be positive.")

    if not 0.0 <= config["dropout"] < 1.0:
        raise ValueError("dropout must be in [0, 1).")

    if not isinstance(config["run_id"], str) or not config["run_id"].strip():
        raise ValueError("run_id must be a non-empty string.")

    if not isinstance(config["ablation"], str):
        raise ValueError("ablation must be a string.")

    # Optional ablation settings.
    config.setdefault("use_positional_embeddings", True)
    config.setdefault("norm_type", "pre")

    if not isinstance(config["use_positional_embeddings"], bool):
        raise ValueError("use_positional_embeddings must be boolean.")

    if config["norm_type"] not in ("pre", "post"):
        raise ValueError("norm_type must be 'pre' or 'post'.")

    return config


# ==================================================
# 2. REPRODUCIBILITY AND DEVICE
# ==================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested_device: str = "auto") -> torch.device:
    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")

        return torch.device("cuda")

    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    raise ValueError("Device must be auto, cpu, or cuda.")


# ==================================================
# 3. DATASET AND TOKEN CACHE
# ==================================================

def load_dataset(tokenizer: BPETokenizer):
    dataset_path = Path("data/input.txt")
    tokenizer_path = Path("tokenizer.json")

    if not dataset_path.exists():
        raise FileNotFoundError("data/input.txt not found.")

    if not tokenizer_path.exists():
        raise FileNotFoundError("tokenizer.json not found.")

    text = dataset_path.read_text(encoding="utf-8")

    # First 90% for training, last 10% for validation.
    split_index = int(len(text) * 0.9)

    train_text = text[:split_index]
    validation_text = text[split_index:]

    fingerprint_source = (
        text + tokenizer_path.read_text(encoding="utf-8")
    )

    fingerprint = hashlib.sha256(
        fingerprint_source.encode("utf-8")
    ).hexdigest()

    train_cache = Path("data/train_ids.pt")
    validation_cache = Path("data/validation_ids.pt")
    metadata_path = Path("data/token_cache_metadata.json")

    train_ids = None
    validation_ids = None

    # Reuse cache only when the dataset and tokenizer match.
    if (
        train_cache.exists()
        and validation_cache.exists()
        and metadata_path.exists()
    ):
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        if metadata.get("fingerprint") == fingerprint:
            print("Loading cached token IDs...")

            train_ids = torch.load(
                train_cache,
                map_location="cpu",
                weights_only=True,
            )

            validation_ids = torch.load(
                validation_cache,
                map_location="cpu",
                weights_only=True,
            )

    if train_ids is None or validation_ids is None:
        print("Encoding training text: 0%")

        train_ids = torch.tensor(
            tokenizer.encode(train_text),
            dtype=torch.long,
        )

        print("Encoding training text: 100%")
        print("Encoding validation text: 0%")

        validation_ids = torch.tensor(
            tokenizer.encode(validation_text),
            dtype=torch.long,
        )

        print("Encoding validation text: 100%")

        train_cache.parent.mkdir(parents=True, exist_ok=True)

        torch.save(train_ids, train_cache)
        torch.save(validation_ids, validation_cache)

        metadata_path.write_text(
            json.dumps(
                {"fingerprint": fingerprint},
                indent=4,
            ),
            encoding="utf-8",
        )

    if len(train_ids) <= 1 or len(validation_ids) <= 1:
        raise ValueError("Dataset split is too small.")

    print(f"Training tokens: {len(train_ids):,}")
    print(f"Validation tokens: {len(validation_ids):,}")

    return train_ids, validation_ids


# ==================================================
# 4. RANDOM BATCH GENERATION
# ==================================================

def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
):
    if len(data) <= block_size:
        raise ValueError("Dataset is too small for block_size.")

    starts = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,),
    )

    x = torch.stack([
        data[start:start + block_size]
        for start in starts
    ])

    y = torch.stack([
        data[start + 1:start + block_size + 1]
        for start in starts
    ])

    return x.to(device), y.to(device)


# ==================================================
# 5. LEARNING RATE SCHEDULE
# ==================================================

def get_learning_rate(step: int, config: dict) -> float:
    max_lr = config["max_lr"]
    warmup_steps = config["warmup_steps"]
    max_steps = config["max_steps"]

    min_lr = max_lr * 0.1

    # Linear warm-up.
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Cosine decay.
    decay_steps = max_steps - warmup_steps

    if decay_steps <= 1:
        return min_lr

    progress = (
        (step - warmup_steps) / (decay_steps - 1)
    )

    progress = min(max(progress, 0.0), 1.0)

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr + cosine * (max_lr - min_lr)


# ==================================================
# 6. ADAMW OPTIMIZER
# ==================================================

def create_optimizer(model: GPT, max_lr: float):
    decay_parameters = []
    no_decay_parameters = []

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay_parameters,
                "weight_decay": 0.1,
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            },
        ],
        lr=max_lr,
        betas=(0.9, 0.95),
    )

    return optimizer


# ==================================================
# 7. VALIDATION
# ==================================================

@torch.no_grad()
def estimate_validation_loss(
    model: GPT,
    validation_ids: torch.Tensor,
    config: dict,
    device: torch.device,
    number_of_batches: int = 50,
):
    was_training = model.training
    model.eval()

    losses = []
    start_time = time.time()

    try:
        for batch_index in range(number_of_batches):
            x, y = get_batch(
                validation_ids,
                config["batch_size"],
                config["block_size"],
                device,
            )

            _, loss = model(x, y)

            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Invalid validation loss.")

            losses.append(loss.item())

            completed = batch_index + 1

            if completed % 5 == 0 or completed == number_of_batches:
                elapsed = time.time() - start_time
                percentage = completed / number_of_batches * 100

                remaining = (
                    elapsed / completed
                ) * (number_of_batches - completed)

                print(
                    f"Validation: {completed}/{number_of_batches} "
                    f"({percentage:.0f}%) | "
                    f"Elapsed: {elapsed:.1f}s | "
                    f"Remaining: {remaining:.1f}s"
                )

    finally:
        model.train(was_training)

    return sum(losses) / len(losses)


# ==================================================
# 8. CSV LOGGING
# ==================================================

LOG_COLUMNS = [
    "step",
    "train_loss",
    "val_loss",
    "lr",
    "tokens_seen",
]

RUN_COLUMNS = [
    "run_id",
    "n_layer",
    "n_head",
    "n_embd",
    "block_size",
    "max_lr",
    "max_steps",
    "ablation",
    "params",
    "best_val_loss",
    "final_train_loss",
    "tokens_per_sec",
    "wall_time_min",
]


def append_log(log_path: Path, row: dict) -> None:
    file_exists = log_path.exists()

    with log_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=LOG_COLUMNS,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def update_runs_csv(config: dict, results: dict) -> None:
    runs_path = Path("runs.csv")
    rows = []

    if runs_path.exists():
        with runs_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)
            rows = list(reader)

    new_row = {
        "run_id": config["run_id"],
        "n_layer": config["n_layer"],
        "n_head": config["n_head"],
        "n_embd": config["n_embd"],
        "block_size": config["block_size"],
        "max_lr": config["max_lr"],
        "max_steps": config["max_steps"],
        "ablation": config["ablation"],
        "params": results["params"],
        "best_val_loss": results["best_val_loss"],
        "final_train_loss": results["final_train_loss"],
        "tokens_per_sec": results["tokens_per_sec"],
        "wall_time_min": results["wall_time_min"],
    }

    # Replace only the row with the same run_id.
    rows = [
        row for row in rows
        if row.get("run_id") != config["run_id"]
    ]

    rows.append(new_row)

    with runs_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RUN_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(rows)


# ==================================================
# 9. MODEL CREATION
# ==================================================

def create_model(config: dict, vocab_size: int) -> GPT:
    model_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config["dropout"],
        use_positional_embeddings=config["use_positional_embeddings"],
        norm_type=config["norm_type"],
    )

    return GPT(model_config)


# ==================================================
# 10. TRAINING
# ==================================================

def train(config_path: str, requested_device: str = "auto") -> None:
    config = load_config(config_path)

    set_seed(config["seed"])

    device = get_device(requested_device)

    print("=" * 55)
    print("GPT TRAINING")
    print("=" * 55)

    print(f"Run ID: {config['run_id']}")
    print(f"Ablation: {config['ablation']}")
    print(f"Device: {device}")
    print(f"Learning rate: {config['max_lr']}")
    print(f"Warmup steps: {config['warmup_steps']}")
    print(f"Positional embeddings: {config['use_positional_embeddings']}")
    print(f"Normalization: {config['norm_type']}")

    # Load tokenizer.
    print("\nLoading tokenizer...")

    tokenizer = BPETokenizer.load("tokenizer.json")
    vocab_size = len(tokenizer.token_bytes)

    print(f"Vocabulary size: {vocab_size}")

    # Load dataset.
    print("\nLoading dataset...")

    train_ids, validation_ids = load_dataset(tokenizer)

    # Create model.
    print("\nCreating GPT model...")

    model = create_model(config, vocab_size).to(device)

    parameter_count = model.num_params()

    print(f"Parameters: {parameter_count:,}")

    # Create optimizer.
    optimizer = create_optimizer(
        model,
        config["max_lr"],
    )

    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    log_path = Path("logs") / f"{config['run_id']}.csv"

    # Start a fresh log for this run.
    if log_path.exists():
        log_path.unlink()

    best_val_loss = float("inf")
    final_train_loss = float("nan")
    tokens_seen = 0

    start_time = time.time()

    print("\nStarting training...\n")

    for step in range(config["max_steps"]):

        # Update learning rate.
        learning_rate = get_learning_rate(step, config)

        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        model.train()

        # Get training batch.
        x, y = get_batch(
            train_ids,
            config["batch_size"],
            config["block_size"],
            device,
        )

        optimizer.zero_grad(set_to_none=True)

        _, loss = model(x, y)

        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(
                f"Invalid training loss at step {step + 1}."
            )

        loss.backward()

        # Gradient clipping.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        final_train_loss = loss.item()

        tokens_seen += (
            config["batch_size"] * config["block_size"]
        )

        current_step = step + 1

        # Training progress: every 1%.
        progress_interval = max(
            1,
            config["max_steps"] // 100,
        )

        if (
            current_step % progress_interval == 0
            or current_step == config["max_steps"]
        ):
            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.time() - start_time

            percentage = (
                current_step / config["max_steps"] * 100
            )

            average_step_time = elapsed / current_step

            remaining = average_step_time * (
                config["max_steps"] - current_step
            )

            print(
                f"Training: {current_step}/{config['max_steps']} "
                f"({percentage:.0f}%) | "
                f"Loss: {final_train_loss:.4f} | "
                f"Elapsed: {elapsed / 60:.1f} min | "
                f"Remaining: {remaining / 60:.1f} min"
            )

        # Evaluate at the configured interval and final step.
        should_evaluate = (
            current_step % config["eval_interval"] == 0
            or current_step == config["max_steps"]
        )

        if should_evaluate:
            print(
                f"\nEvaluating model at step {current_step}..."
            )

            validation_loss = estimate_validation_loss(
                model,
                validation_ids,
                config,
                device,
                number_of_batches=50,
            )

            append_log(
                log_path,
                {
                    "step": current_step,
                    "train_loss": final_train_loss,
                    "val_loss": validation_loss,
                    "lr": learning_rate,
                    "tokens_seen": tokens_seen,
                },
            )

            print(
                f"Train loss: {final_train_loss:.4f} | "
                f"Validation loss: {validation_loss:.4f}"
            )

            # Save the best checkpoint.
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss

                checkpoint_path = (
                    Path("checkpoints")
                    / f"{config['run_id']}_best.pt"
                )

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": vars(model.config),
                        "step": current_step,
                        "best_val_loss": best_val_loss,
                    },
                    checkpoint_path,
                )

                print(
                    f"Best checkpoint saved: {checkpoint_path}"
                )

            print("-" * 55)

    # Final statistics.
    if device.type == "cuda":
        torch.cuda.synchronize()

    wall_time = time.time() - start_time

    tokens_per_sec = (
        tokens_seen / wall_time
        if wall_time > 0
        else 0.0
    )

    results = {
        "params": parameter_count,
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
        "tokens_per_sec": tokens_per_sec,
        "wall_time_min": wall_time / 60,
    }

    update_runs_csv(config, results)

    print("\n" + "=" * 55)
    print("TRAINING COMPLETED: 100%")
    print("=" * 55)

    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final training loss: {final_train_loss:.4f}")
    print(f"Tokens per second: {tokens_per_sec:.2f}")
    print(f"Wall time: {wall_time / 60:.2f} minutes")
    print(f"Log saved: {log_path}")
    print("Run information saved: runs.csv")


# ==================================================
# 11. COMMAND-LINE INTERFACE
# ==================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train a GPT model from scratch."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file.",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device. Default: auto.",
    )

    args = parser.parse_args()

    train(
        args.config,
        requested_device=args.device,
    )


if __name__ == "__main__":
    main()
