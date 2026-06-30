#!/usr/bin/env python3
"""Select LRs on seeds 0--4 and analyze disjoint held-out seed runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, wilcoxon


def read_rows(paths: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with open(path, newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    keys = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def select_lr(tuning: Sequence[Dict[str, str]], dim: int, variant: str) -> float:
    candidates = []
    for lr in sorted({float(row["lr"]) for row in tuning}):
        values = [
            int(row["first_stable_tail_step"])
            for row in tuning
            if int(row["dim"]) == dim
            and row["variant"] == variant
            and float(row["lr"]) == lr
            and int(row["seed"]) < 5
            and row["first_stable_tail_step"]
        ]
        candidates.append((-len(values), np.median(values) if values else np.inf, lr))
    return float(min(candidates)[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-summary", required=True)
    parser.add_argument("--heldout-d8", nargs="+", required=True)
    parser.add_argument("--heldout-d16", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=50000)
    args = parser.parse_args()

    tuning = read_rows([args.tuning_summary])
    heldout_by_dim = {8: read_rows(args.heldout_d8), 16: read_rows(args.heldout_d16)}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_rows: List[Dict[str, object]] = []
    statistics: List[Dict[str, object]] = []
    rng = np.random.default_rng(0)

    for dim in (8, 16):
        common_lr = select_lr(tuning, dim, "common_oracle")
        tail_lr = select_lr(tuning, dim, "residual_oracle")
        rows = heldout_by_dim[dim]
        common = {
            int(row["seed"]): int(row["first_stable_tail_step"])
            for row in rows
            if row["variant"] == "common_oracle"
            and float(row["lr"]) == common_lr
            and int(row["seed"]) >= 5
        }
        tail = {
            int(row["seed"]): int(row["first_stable_tail_step"])
            for row in rows
            if row["variant"] == "residual_oracle"
            and float(row["lr"]) == tail_lr
            and int(row["seed"]) >= 5
        }
        seeds = sorted(set(common) & set(tail))
        common_values = np.array([common[seed] for seed in seeds], dtype=float)
        tail_values = np.array([tail[seed] for seed in seeds], dtype=float)
        differences = tail_values - common_values
        for seed, common_step, tail_step in zip(seeds, common_values, tail_values):
            selected_rows.append(
                {
                    "dim": dim,
                    "seed": seed,
                    "common_lr": common_lr,
                    "spectral_tail_lr": tail_lr,
                    "common_stable_step": int(common_step),
                    "spectral_tail_stable_step": int(tail_step),
                    "tail_minus_common_step": int(tail_step - common_step),
                }
            )
        bootstrap_means = np.array(
            [
                differences[rng.integers(0, len(differences), len(differences))].mean()
                for _ in range(args.bootstrap_samples)
            ]
        )
        wins = int((differences < 0).sum())
        ties = int((differences == 0).sum())
        losses = int((differences > 0).sum())
        statistics.append(
            {
                "dim": dim,
                "heldout_seeds": len(seeds),
                "common_lr_selected_on_seeds_0_4": common_lr,
                "spectral_tail_lr_selected_on_seeds_0_4": tail_lr,
                "common_median_step": float(np.median(common_values)),
                "spectral_tail_median_step": float(np.median(tail_values)),
                "common_mean_step": float(common_values.mean()),
                "spectral_tail_mean_step": float(tail_values.mean()),
                "mean_tail_minus_common_step": float(differences.mean()),
                "bootstrap_mean_diff_ci_low": float(np.quantile(bootstrap_means, 0.025)),
                "bootstrap_mean_diff_ci_high": float(np.quantile(bootstrap_means, 0.975)),
                "tail_wins": wins,
                "ties": ties,
                "tail_losses": losses,
                "wilcoxon_one_sided_p_tail_faster": float(wilcoxon(differences, alternative="less").pvalue),
                "sign_test_one_sided_p_tail_faster": float(
                    binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue
                ),
            }
        )

    write_csv(output / "heldout_seed_results.csv", selected_rows)
    write_csv(output / "heldout_statistics.csv", statistics)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, dim in zip(axes, (8, 16)):
        rows = [row for row in selected_rows if int(row["dim"]) == dim]
        common = [int(row["common_stable_step"]) for row in rows]
        tail = [int(row["spectral_tail_stable_step"]) for row in rows]
        for x, y in zip(common, tail):
            ax.scatter(x, y, color="#4C78A8", alpha=0.7)
        limit = max(common + tail) * 1.05
        ax.plot([0, limit], [0, limit], linestyle="--", color="black", linewidth=1)
        ax.set_xlabel("common-subspace stable step")
        ax.set_ylabel("spectral-tail stable step")
        ax.set_title(f"dim={dim}, held-out n={len(rows)}")
        ax.grid(alpha=0.25)
    fig.suptitle("Paired held-out comparison (below diagonal favors spectral tail)")
    fig.tight_layout()
    fig.savefig(output / "heldout_paired_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
