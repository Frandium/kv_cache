from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "main_swapmoe" / "assets"
OUT.mkdir(exist_ok=True)
COLORS = {"baseline": "#6b7280", "proposed": "#2563eb"}
LABELS = {"baseline": "Residual Routing", "proposed": "Attention-Mean Routing"}
MARKERS = {"M": "o", "L": "s"}


def read(name: str):
    with (ROOT / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, name: str):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def linear_fit(xs, ys):
    return np.polyfit(np.asarray(xs), np.asarray(ys), 1)


def scaling():
    rows = read("scaling_points.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), constrained_layout=True)
    panels = (
        ("downstream_macro_accuracy", lambda value: 100 * value, "Downstream macro accuracy (%)"),
        ("test_loss", lambda value: math.exp(value), "Held-out perplexity"),
    )
    for ax, (metric, transform, ylabel) in zip(axes, panels):
        for method in ("baseline", "proposed"):
            selected = [row for row in rows if row["metric"] == metric and row["method"] == method]
            xs = [float(row["log10_training_flops"]) for row in selected]
            ys = [transform(float(row["value"])) for row in selected]
            grid = np.linspace(min(xs), max(xs), 200)
            if metric == "test_loss":
                loss = [float(row["value"]) for row in selected]
                slope, intercept = linear_fit(xs, loss)
                curve = np.exp(slope * grid + intercept)
            else:
                slope, intercept = linear_fit(xs, ys)
                curve = slope * grid + intercept
            ax.plot(grid, curve, lw=2.4, color=COLORS[method], label=LABELS[method])
            for size in ("M", "L"):
                points = [row for row in selected if row["size"] == size]
                ax.scatter(
                    [float(row["log10_training_flops"]) for row in points],
                    [transform(float(row["value"])) for row in points],
                    marker=MARKERS[size], s=52, color=COLORS[method], edgecolor="white",
                    linewidth=0.7, zorder=3,
                )
        ax.set_xlabel(r"$\log_{10}$ training FLOPs")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False)
    axes[0].text(0.02, 0.03, "circle: M   square: L", transform=axes[0].transAxes, fontsize=9)
    fig.suptitle("Capability scaling across M and L checkpoints", fontsize=14)
    save(fig, "0814_scaling_law.png")


def load_heatmaps():
    rows = read("expert_load_per_layer.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4), constrained_layout=True)
    image = None
    for i, size in enumerate(("M", "L")):
        for j, method in enumerate(("baseline", "proposed")):
            selected = [row for row in rows if row["size"] == size and row["method"] == method]
            layers = max(int(row["layer"]) for row in selected) + 1
            matrix = np.zeros((layers, 8))
            for row in selected:
                matrix[int(row["layer"]), int(row["expert"])] = 100 * float(row["activation_share"])
            ax = axes[i, j]
            image = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=40)
            ax.set_title(f"{size} — {LABELS[method]}")
            ax.set_xlabel("Expert ID")
            ax.set_ylabel("Layer (0-indexed)")
            ax.set_xticks(range(8))
    fig.colorbar(image, ax=axes, label="Activation share within layer (%)", shrink=0.85)
    fig.suptitle("All layer-expert pairs remain active", fontsize=14)
    save(fig, "0814_expert_load_heatmaps.png")


def continuity():
    rows = read("continuity_by_budget.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    for ax, size in zip(axes, ("M", "L")):
        for method in ("baseline", "proposed"):
            selected = sorted(
                [
                    row for row in rows
                    if row["size"] == size and row["method"] == method
                    and int(row["cache_capacity"]) in (1, 2, 4)
                ],
                key=lambda row: int(row["cache_capacity"]),
            )
            ax.plot(
                [int(row["cache_capacity"]) for row in selected],
                [float(row["loads_per_100_token_layer"]) for row in selected],
                marker="o", lw=2.3, color=COLORS[method], label=LABELS[method],
            )
        ax.set_title(f"{size} model")
        ax.set_xlabel("Resident tail experts per layer (K)")
        ax.set_ylabel("Loads per 100 token-layer accesses")
        ax.set_xticks([1, 2, 4])
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False)
    fig.suptitle("Expert-cache loads on 64 × 2048-token sequences", fontsize=14)
    save(fig, "0814_continuity_by_budget.png")


def predictability_heatmaps():
    rows = [row for row in read("predictability.csv") if row["recall_k"] == "1"]
    combinations = (
        ("M", "baseline"), ("M", "proposed"),
        ("L", "baseline"), ("L", "proposed"),
    )
    fig, axes = plt.subplots(4, 2, figsize=(10.5, 16), constrained_layout=True)
    image = None
    for row_index, (size, method) in enumerate(combinations):
        for column, task in enumerate(("same_token", "next_token")):
            selected = [
                row for row in rows
                if row["size"] == size and row["method"] == method and row["task"] == task
            ]
            layers = max(
                max(int(row["source_layer"]), int(row["target_layer"])) for row in selected
            ) + 1
            matrix = np.full((layers, layers), np.nan)
            for row in selected:
                matrix[int(row["source_layer"]), int(row["target_layer"])] = 100 * float(row["recall"])
            ax = axes[row_index, column]
            image = ax.imshow(
                matrix, origin="lower", aspect="equal", cmap="viridis", vmin=10, vmax=75
            )
            task_name = "Same token (i < j)" if task == "same_token" else "Next token (i ≥ j)"
            short_method = "Residual" if method == "baseline" else "Attention-Mean"
            ax.set_title(f"{size} {short_method} — {task_name}")
            ax.set_xlabel("Target layer j")
            ax.set_ylabel("Source layer i")
    fig.colorbar(image, ax=axes, label="Recall@1 (%)", shrink=0.62)
    fig.suptitle("Layer-to-layer expert-activation predictability", fontsize=14)
    save(fig, "0814_predictability_heatmaps.png")


def latency():
    decode = read("ppu_latency_by_budget.csv")
    ttft = [
        row for row in read("ppu_ttft_by_budget.csv")
        if row["cache_state"] == "warm" and row["prompt_length"] == "2048"
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8), constrained_layout=True)
    for column, size in enumerate(("M", "L")):
        for ax, rows, field, ylabel, title in (
            (axes[0, column], decode, "milliseconds_per_token", "Decode latency (ms/token)", "2048-token decode"),
            (axes[1, column], ttft, "ttft_milliseconds", "TTFT (ms)", "2048-token prefill to first output"),
        ):
            for method in ("baseline", "proposed"):
                selected = sorted(
                    [
                        row for row in rows
                        if row["size"] == size and row["method"] == method
                        and int(row["cache_capacity"]) in (1, 2, 4)
                    ],
                    key=lambda row: int(row["cache_capacity"]),
                )
                ax.plot(
                    [int(row["cache_capacity"]) for row in selected],
                    [float(row[field]) for row in selected],
                    marker="o", lw=2.3, color=COLORS[method], label=LABELS[method],
                )
            ax.set_title(f"{size} — {title}")
            ax.set_xlabel("Resident tail experts per layer (K)")
            ax.set_ylabel(ylabel)
            ax.set_xticks([1, 2, 4])
            ax.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Measured PPU-ZW810 latency", fontsize=14)
    save(fig, "0814_ppu_latency.png")


if __name__ == "__main__":
    scaling()
    load_heatmaps()
    continuity()
    predictability_heatmaps()
    latency()
    print(OUT)
