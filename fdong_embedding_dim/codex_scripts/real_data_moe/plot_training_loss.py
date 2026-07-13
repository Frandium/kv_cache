from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--proposed-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smooth-window", type=int, default=50)
    return parser.parse_args()


def read_metrics(path: str) -> List[Dict[str, float]]:
    by_step: Dict[int, Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                step = int(record["step"])
                by_step[step] = {"step": float(step), "loss": float(record["loss"])}
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    if not by_step:
        raise RuntimeError(f"no valid loss records in {path}")
    return [by_step[step] for step in sorted(by_step)]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    smoothed = (cumulative[window:] - cumulative[:-window]) / window
    return np.concatenate((np.full(window - 1, np.nan), smoothed))


def main() -> None:
    args = parse_args()
    if args.smooth_window < 1:
        raise ValueError("smooth-window must be positive")
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "baseline": read_metrics(args.baseline_metrics),
        "proposed": read_metrics(args.proposed_metrics),
    }
    common_max_step = min(records[-1]["step"] for records in results.values())

    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    colors = {"baseline": "#777777", "proposed": "#2a6fbb"}
    for name, records in results.items():
        records = [record for record in records if record["step"] <= common_max_step]
        steps = np.asarray([record["step"] for record in records])
        losses = np.asarray([record["loss"] for record in records])
        axis.plot(steps, losses, color=colors[name], alpha=0.12, linewidth=0.7)
        axis.plot(
            steps,
            moving_average(losses, args.smooth_window),
            color=colors[name],
            linewidth=2,
            label=f"{name} ({args.smooth_window}-record mean)",
        )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Training next-token loss")
    axis.set_title(f"Training loss through common step {int(common_max_step)}")
    axis.grid(alpha=0.25)
    axis.legend()
    output_path = os.path.join(args.output_dir, "training_loss.png")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    with open(os.path.join(args.output_dir, "training_loss_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "common_max_step": int(common_max_step),
                "smooth_window": args.smooth_window,
                "baseline_records": len(results["baseline"]),
                "proposed_records": len(results["proposed"]),
            },
            handle,
            indent=2,
        )
    print(output_path)


if __name__ == "__main__":
    main()
