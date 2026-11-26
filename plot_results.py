
"""Create the two figures required for Part E."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
OUTPUT_DIR = ROOT / "figures"

OUTPUT_DIR.mkdir(exist_ok=True)

# Figure 1: Holdout shard loss curves
run_names = ["S", "M", "L", "A1", "A2", "A3", "A4", "A5"]

plt.figure(figsize=(10, 6))

for run_name in run_names:
    log_path = LOG_DIR / f"{run_name}.csv"

    with log_path.open("r", newline="") as file:
        rows = list(csv.DictReader(file))

    steps = [int(row["step"]) for row in rows]
    val_losses = [float(row["val_loss"]) for row in rows]

    plt.plot(steps, val_losses, marker="o", markersize=3,
             linewidth=1.5, label=run_name)

plt.xlabel("Training Steps")
plt.ylabel("Holdout Shard Loss")
plt.title("Holdout Shard Loss Curves")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "validation_loss_curves.png", dpi=300)
plt.close()

# Figure 2: Model scaling
scaling_results = []

with (ROOT / "runs.csv").open("r", newline="") as file:
    for row in csv.DictReader(file):
        if row["run_id"] in ["S", "M", "L"]:
            scaling_results.append({
                "run_id": row["run_id"],
                "params": int(row["params"]),
                "best_val_loss": float(row["best_val_loss"]),
            })

scaling_results.sort(key=lambda item: item["params"])

params = [item["params"] for item in scaling_results]
losses = [item["best_val_loss"] for item in scaling_results]
labels = [item["run_id"] for item in scaling_results]

plt.figure(figsize=(8, 5))
plt.plot(params, losses, marker="o", linewidth=2)

for x, y, label in zip(params, losses, labels):
    plt.annotate(label, (x, y), xytext=(6, 8),
                 textcoords="offset points")

plt.xlabel("Number of Parameters")
plt.ylabel("Best Holdout Shard Loss")
plt.title("Model Scaling: Parameters vs. Best Holdout Shard Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_scaling.png", dpi=300)
plt.close()

print("Both figures created successfully.")
print(f"Output directory: {OUTPUT_DIR}")
print("1. validation_loss_curves.png")
print("2. model_scaling.png")
