from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["backbone_kind"], int(row["hidden_size"]), int(row["mtp"]))].append(row)

    backbones = sorted({key[0] for key in groups})
    hidden_sizes = sorted({key[1] for key in groups})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.36
    x = np.arange(len(backbones) * len(hidden_sizes))
    labels = [f"{b}\nd={d}" for b in backbones for d in hidden_sizes]

    for mtp, shift, color in [(1, -width / 2, "#4C78A8"), (3, width / 2, "#F58518")]:
        probe_means, rank_means = [], []
        for backbone in backbones:
            for hidden in hidden_sizes:
                group = groups.get((backbone, hidden, mtp), [])
                probe_means.append(np.mean([float(r["linear_probe_test_accuracy"]) for r in group]))
                rank_means.append(np.mean([float(r["effective_rank"]) for r in group]))
        axes[0].bar(x + shift, probe_means, width, label=f"MTP={mtp}", color=color)
        axes[1].bar(x + shift, rank_means, width, label=f"MTP={mtp}", color=color)

    axes[0].set_ylabel("Frozen linear probe suffix accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel("Centered prefix effective rank")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()

