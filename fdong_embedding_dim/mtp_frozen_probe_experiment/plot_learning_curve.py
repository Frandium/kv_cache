from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def maybe_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    groups = defaultdict(list)
    for row in rows:
        groups[row["run_name"]].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    for run_name, group in sorted(groups.items()):
        group = sorted(group, key=lambda item: int(item["step"]))
        steps = [int(item["step"]) for item in group]
        mtp = int(group[0]["mtp"])
        label_base = run_name.replace("_seed", "_s")

        offset1 = [maybe_float(item["train_offset1_ce"]) for item in group]
        axes[0].plot(steps, offset1, label=f"{label_base} offset1")

        total = [maybe_float(item["train_total_ce"]) for item in group]
        axes[1].plot(steps, total, label=f"{label_base} total", linewidth=2.0)
        if mtp > 1:
            for offset in range(1, mtp + 1):
                values = [maybe_float(item[f"train_offset{offset}_ce"]) for item in group]
                axes[1].plot(steps, values, linestyle="--", label=f"{label_base} offset{offset}")

    axes[0].set_title("Same-task comparison: train offset-1 CE")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("cross entropy")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)

    axes[1].set_title("MTP diagnostic: total CE and per-offset CE")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("cross entropy")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
