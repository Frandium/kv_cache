#!/usr/bin/env python3
"""Held-out analysis for common, spectral-tail, and unrestricted output branches."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, wilcoxon


VARIANTS = ("common_oracle", "residual_oracle", "full_output_oracle")
LABELS = {
    "common_oracle": "top-2 common",
    "residual_oracle": "bottom-2 spectral tail",
    "full_output_oracle": "unrestricted full output",
}


def read_rows(paths: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with open(path, newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_lr(tuning: Sequence[Dict[str, str]], dim: int, variant: str) -> float:
    candidates: List[Tuple[int, float, float]] = []
    for lr in sorted({float(row["lr"]) for row in tuning}):
        values = [
            int(row["first_stable_tail_step"])
            for row in tuning
            if int(row["dim"]) == dim
            and row["variant"] == variant
            and int(row["seed"]) < 5
            and float(row["lr"]) == lr
            and row["first_stable_tail_step"]
        ]
        candidates.append((-len(values), float(np.median(values)) if values else np.inf, lr))
    return float(min(candidates)[2])


def bootstrap_ci(differences: np.ndarray, rng: np.random.Generator, samples: int) -> Tuple[float, float]:
    means = np.array(
        [differences[rng.integers(0, len(differences), len(differences))].mean() for _ in range(samples)]
    )
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-summary", required=True)
    parser.add_argument("--heldout-d8", nargs="+", required=True)
    parser.add_argument("--heldout-d16", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=50000)
    args = parser.parse_args()

    tuning = read_rows([args.tuning_summary])
    heldout = {8: read_rows(args.heldout_d8), 16: read_rows(args.heldout_d16)}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    seed_rows: List[Dict[str, object]] = []
    variant_rows: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []

    values_by_dim: Dict[int, Dict[str, Dict[int, int]]] = {}
    lrs_by_dim: Dict[int, Dict[str, float]] = {}
    for dim in (8, 16):
        values_by_dim[dim] = {}
        lrs_by_dim[dim] = {}
        for variant in VARIANTS:
            lr = selected_lr(tuning, dim, variant)
            lrs_by_dim[dim][variant] = lr
            values = {
                int(row["seed"]): int(row["first_stable_tail_step"])
                for row in heldout[dim]
                if row["variant"] == variant
                and float(row["lr"]) == lr
                and int(row["seed"]) >= 5
                and row["first_stable_tail_step"]
            }
            values_by_dim[dim][variant] = values
            array = np.array(list(values.values()), dtype=float)
            variant_rows.append(
                {
                    "dim": dim,
                    "variant": variant,
                    "label": LABELS[variant],
                    "lr_selected_on_seeds_0_4": lr,
                    "heldout_seeds": len(array),
                    "median_stable_step": float(np.median(array)),
                    "mean_stable_step": float(array.mean()),
                    "std_stable_step": float(array.std(ddof=1)),
                }
            )

        common_seeds = sorted(set.intersection(*(set(values_by_dim[dim][v]) for v in VARIANTS)))
        for seed in common_seeds:
            seed_rows.append(
                {
                    "dim": dim,
                    "seed": seed,
                    "common_step": values_by_dim[dim]["common_oracle"][seed],
                    "spectral_tail_step": values_by_dim[dim]["residual_oracle"][seed],
                    "full_output_step": values_by_dim[dim]["full_output_oracle"][seed],
                }
            )

        for left, right in (
            ("full_output_oracle", "common_oracle"),
            ("full_output_oracle", "residual_oracle"),
            ("residual_oracle", "common_oracle"),
        ):
            seeds = sorted(set(values_by_dim[dim][left]) & set(values_by_dim[dim][right]))
            left_values = np.array([values_by_dim[dim][left][seed] for seed in seeds], dtype=float)
            right_values = np.array([values_by_dim[dim][right][seed] for seed in seeds], dtype=float)
            differences = left_values - right_values
            ci_low, ci_high = bootstrap_ci(differences, rng, args.bootstrap_samples)
            wins = int((differences < 0).sum())
            ties = int((differences == 0).sum())
            losses = int((differences > 0).sum())
            pair_rows.append(
                {
                    "dim": dim,
                    "left_variant": left,
                    "right_variant": right,
                    "comparison": f"{LABELS[left]} minus {LABELS[right]}",
                    "paired_seeds": len(seeds),
                    "mean_step_difference": float(differences.mean()),
                    "bootstrap_mean_diff_ci_low": ci_low,
                    "bootstrap_mean_diff_ci_high": ci_high,
                    "left_wins": wins,
                    "ties": ties,
                    "left_losses": losses,
                    "wilcoxon_one_sided_p_left_faster": float(wilcoxon(differences, alternative="less").pvalue),
                    "sign_test_one_sided_p_left_faster": float(
                        binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue
                    ),
                }
            )

    write_csv(output / "three_way_heldout_seed_results.csv", seed_rows)
    write_csv(output / "three_way_heldout_variant_summary.csv", variant_rows)
    write_csv(output / "three_way_heldout_pairwise.csv", pair_rows)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for row_index, dim in enumerate((8, 16)):
        full = values_by_dim[dim]["full_output_oracle"]
        for column_index, constrained in enumerate(("common_oracle", "residual_oracle")):
            other = values_by_dim[dim][constrained]
            seeds = sorted(set(full) & set(other))
            x = np.array([other[seed] for seed in seeds])
            y = np.array([full[seed] for seed in seeds])
            ax = axes[row_index, column_index]
            ax.scatter(x, y, alpha=0.65)
            limit = max(float(x.max()), float(y.max())) * 1.05
            ax.plot([0, limit], [0, limit], "k--", linewidth=1)
            ax.set_xlabel(f"{LABELS[constrained]} stable step")
            ax.set_ylabel("unrestricted full-output stable step")
            ax.set_title(f"dim={dim}, n={len(seeds)}")
            ax.grid(alpha=0.25)
    fig.suptitle("Held-out paired comparison: below diagonal favors unrestricted output")
    fig.tight_layout()
    fig.savefig(output / "three_way_heldout_paired.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
