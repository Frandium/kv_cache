#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECTIONS = ("q", "k", "v", "o")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument(
        "--layers-by-projection",
        help="Comma-separated mapping such as q:11,k:0,v:12,o:25",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected_layers = {projection: args.layer for projection in PROJECTIONS}
    if args.layers_by_projection:
        for entry in args.layers_by_projection.split(","):
            projection, layer = entry.split(":", maxsplit=1)
            selected_layers[projection.strip()] = int(layer)

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    by_projection = {projection: [] for projection in PROJECTIONS}
    for item in results["matrices"]:
        projection = item["projection"]
        if projection in by_projection:
            by_projection[projection].append(item)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, projection in zip(axes.flat, PROJECTIONS):
        ax.set_facecolor("white")
        items = by_projection[projection]
        selected_layer = selected_layers[projection]
        selected = next(item for item in items if item["layer"] == selected_layer)
        curves = [item["frequency_sorted"] for item in items]
        mean_curve = [sum(values) / len(values) for values in zip(*curves)]
        x = [100.0 * index / (len(mean_curve) - 1) for index in range(len(mean_curve))]

        ax.plot(x, mean_curve, color="#9aa3ad", linewidth=2.0, linestyle="--", label="28-layer mean")
        ax.plot(
            x,
            selected["frequency_sorted"],
            color="#1565a8",
            linewidth=2.6,
            label=f"Representative layer L{selected_layer}",
        )
        ax.set_title(f"{projection}_proj")
        ax.grid(True, alpha=0.22)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.02)

    axes[0, 0].legend(frameon=False)
    for ax in axes[1, :]:
        ax.set_xlabel("Feature percentile ranked by activation frequency")
    for ax in axes[:, 0]:
        ax.set_ylabel("Activation frequency")

    fig.suptitle("Representative sharp feature-activation layers in Qwen3-0.6B", fontsize=15)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
