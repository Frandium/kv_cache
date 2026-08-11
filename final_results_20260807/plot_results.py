from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

COLORS = {"baseline": "#6b7280", "proposed": "#2563eb"}
MARKERS = {"baseline": "o", "proposed": "s"}


def read(name: str):
    with (ROOT / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, name: str):
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def scaling():
    rows = read("scaling_points.csv")
    wanted = [
        ("downstream_macro_accuracy", "Downstream macro accuracy", lambda x: 100 * x, "%"),
        ("train_loss", "Smoothed train loss", float, "loss"),
        ("test_loss", "Held-out test loss", float, "loss"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, (metric, title, transform, ylabel) in zip(axes, wanted):
        for method in ("baseline", "proposed"):
            values = sorted(
                [r for r in rows if r["metric"] == metric and r["method"] == method],
                key=lambda r: float(r["log10_training_flops"]),
            )
            x = [float(r["log10_training_flops"]) for r in values]
            y = [transform(float(r["value"])) for r in values]
            ax.plot(x, y, marker=MARKERS[method], lw=2, ms=7, color=COLORS[method], label=method)
            for r, px, py in zip(values, x, y):
                ax.annotate(r["size"], (px, py), xytext=(5, 5), textcoords="offset points")
        ax.set_title(title)
        ax.set_xlabel(r"$\log_{10}$ training FLOPs")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Scaling trend at matched checkpoints (two model sizes)", fontsize=14)
    save(fig, "01_scaling.png")


def load_heatmaps():
    rows = read("expert_load_per_layer.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    image = None
    for row_index, size in enumerate(("M", "L")):
        for col_index, method in enumerate(("baseline", "proposed")):
            selected = [r for r in rows if r["size"] == size and r["method"] == method]
            layers = max(int(r["layer"]) for r in selected) + 1
            experts = max(int(r["expert"]) for r in selected) + 1
            matrix = np.zeros((layers, experts))
            for r in selected:
                matrix[int(r["layer"]), int(r["expert"])] = 100 * float(r["activation_share"])
            ax = axes[row_index, col_index]
            image = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=40)
            ax.set_title(f"{size} {method}")
            ax.set_xlabel("Expert")
            ax.set_ylabel("Layer (0-indexed)")
            ax.set_xticks(range(experts))
    fig.colorbar(image, ax=axes, label="Activation share within layer (%)", shrink=0.85)
    fig.suptitle("Per-layer expert load", fontsize=14)
    save(fig, "02_expert_load_heatmaps.png")


def continuity():
    rows = read("continuity_by_budget.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, size in zip(axes, ("M", "L")):
        for method in ("baseline", "proposed"):
            selected = sorted(
                [r for r in rows if r["size"] == size and r["method"] == method],
                key=lambda r: int(r["cache_capacity"]),
            )
            ax.plot(
                [int(r["cache_capacity"]) for r in selected],
                [float(r["loads_per_100_token_layer"]) for r in selected],
                marker=MARKERS[method], lw=2, color=COLORS[method], label=method,
            )
        ax.set_title(f"{size} model")
        ax.set_xlabel("Resident tail experts per layer (K)")
        ax.set_ylabel("Loads per 100 token-layer accesses")
        ax.set_xticks([1, 2, 4, 8])
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Expert-cache misses on 64 × 2048-token sequences", fontsize=14)
    save(fig, "03_continuity_by_budget.png")


def latency():
    rows = read("ppu_latency_by_budget.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, size in zip(axes, ("M", "L")):
        baseline_infinite = float(next(
            r["milliseconds_per_token"] for r in rows
            if r["size"] == size and r["method"] == "baseline" and r["cache_capacity"] == "8"
        ))
        for method in ("baseline", "proposed"):
            selected = sorted(
                [r for r in rows if r["size"] == size and r["method"] == method],
                key=lambda r: int(r["cache_capacity"]),
            )
            ax.plot(
                [int(r["cache_capacity"]) for r in selected],
                [float(r["milliseconds_per_token"]) for r in selected],
                marker=MARKERS[method], lw=2, color=COLORS[method], label=method,
            )
        ax.axhline(baseline_infinite, ls="--", lw=1.3, color="black", alpha=0.6, label="baseline K=8 reference")
        ax.set_title(f"{size} model")
        ax.set_xlabel("Resident tail experts per layer (K)")
        ax.set_ylabel("Decode latency (ms/token)")
        ax.set_xticks([1, 2, 4, 8])
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("PPU-ZW810 autoregressive decode latency (2048 tokens)", fontsize=14)
    save(fig, "04_ppu_latency.png")


def predictability_summary():
    rows = read("predictability.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    styles = {("M", "baseline"): "--", ("M", "proposed"): "-", ("L", "baseline"): "--", ("L", "proposed"): "-"}
    for ax, task in zip(axes, ("same_token", "next_token")):
        for size in ("M", "L"):
            for method in ("baseline", "proposed"):
                means = []
                for k in (1, 2, 4):
                    values = [float(r["recall"]) for r in rows if r["size"] == size and r["method"] == method and r["task"] == task and int(r["recall_k"]) == k]
                    means.append(100 * sum(values) / len(values))
                ax.plot(
                    (1, 2, 4), means, marker=MARKERS[method], lw=2,
                    ls=styles[(size, method)], color=COLORS[method],
                    label=f"{size} {method}", alpha=1.0 if size == "L" else 0.65,
                )
        ax.set_title("Same token: i < j" if task == "same_token" else "Current token → next token: i ≥ j")
        ax.set_xlabel("Recall@K")
        ax.set_ylabel("Mean recall (%)")
        ax.set_xticks([1, 2, 4])
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Expert-activation predictability", fontsize=14)
    save(fig, "05_predictability_summary.png")


def predictability_heatmaps():
    rows = [r for r in read("predictability.csv") if r["recall_k"] == "1"]
    fig, axes = plt.subplots(4, 2, figsize=(11, 17), constrained_layout=True)
    image = None
    for row_index, (size, method) in enumerate(
        (("M", "baseline"), ("M", "proposed"), ("L", "baseline"), ("L", "proposed"))
    ):
        for col_index, task in enumerate(("same_token", "next_token")):
            selected = [r for r in rows if r["size"] == size and r["method"] == method and r["task"] == task]
            layers = max(max(int(r["source_layer"]), int(r["target_layer"])) for r in selected) + 1
            matrix = np.full((layers, layers), np.nan)
            for r in selected:
                matrix[int(r["source_layer"]), int(r["target_layer"])] = 100 * float(r["recall"])
            ax = axes[row_index, col_index]
            image = ax.imshow(matrix, origin="lower", aspect="equal", cmap="viridis", vmin=10, vmax=75)
            ax.set_title(f"{size} {method} — {'same token' if task == 'same_token' else 'next token'}")
            ax.set_xlabel("Target layer j")
            ax.set_ylabel("Source layer i")
    fig.colorbar(image, ax=axes, label="Recall@1 (%)", shrink=0.6)
    fig.suptitle("Layer-to-layer expert prediction", fontsize=14)
    save(fig, "06_predictability_recall1_heatmaps.png")


def downstream():
    rows = [r for r in read("downstream_accuracy.csv") if r["task"] != "macro_average"]
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    x = np.arange(len(tasks))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True, sharey=True)
    width = 0.36
    for ax, size in zip(axes, ("M", "L")):
        for offset, method in ((-width / 2, "baseline"), (width / 2, "proposed")):
            values = [100 * float(next(r["value"] for r in rows if r["size"] == size and r["method"] == method and r["task"] == task)) for task in tasks]
            ax.bar(x + offset, values, width, color=COLORS[method], label=method)
        ax.set_title(f"{size} model")
        ax.set_xticks(x, tasks, rotation=35, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Downstream task accuracy", fontsize=14)
    save(fig, "07_downstream_accuracy.png")


if __name__ == "__main__":
    scaling()
    load_heatmaps()
    continuity()
    latency()
    predictability_summary()
    predictability_heatmaps()
    downstream()
    print(FIGURES)
