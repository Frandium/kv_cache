from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read(path: str) -> tuple[np.ndarray, np.ndarray]:
    records = {}
    for line in Path(path).read_text().splitlines():
        item = json.loads(line)
        records[int(item["step"])] = float(item["loss"])
    steps = np.asarray(sorted(records))
    return steps, np.asarray([records[int(step)] for step in steps])


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values
    # Use all available history before a full window exists. This preserves
    # the initial loss descent and always returns one value per input step.
    cumulative = np.cumsum(values, dtype=np.float64)
    totals = cumulative.copy()
    if values.size > window:
        totals[window:] -= cumulative[:-window]
    counts = np.minimum(np.arange(1, values.size + 1), window)
    return totals / counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="NAME=METRICS_JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smooth-window", type=int, default=50)
    parser.add_argument(
        "--truncate-common",
        action="store_true",
        help="truncate every curve to the largest step available in every run",
    )
    args = parser.parse_args()

    series = []
    for item in args.run:
        name, path = item.split("=", 1)
        steps, losses = read(path)
        if steps.size == 0:
            raise ValueError(f"no metrics found for {name}: {path}")
        series.append((name, steps, losses))
    common_max_step = min(int(steps[-1]) for _, steps, _ in series)

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for name, steps, losses in series:
        if args.truncate_common:
            keep = steps <= common_max_step
            steps = steps[keep]
            losses = losses[keep]
        axis.plot(steps, smooth(losses, args.smooth_window), linewidth=2, label=name)
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Training next-token loss")
    axis.set_title(f"Training loss ({args.smooth_window}-record moving mean)")
    axis.grid(alpha=0.25)
    axis.legend()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
