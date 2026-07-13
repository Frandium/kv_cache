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
    if window <= 1:
        return values
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    means = (cumulative[window:] - cumulative[:-window]) / window
    return np.concatenate((np.full(window - 1, np.nan), means))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="NAME=METRICS_JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smooth-window", type=int, default=50)
    args = parser.parse_args()

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for item in args.run:
        name, path = item.split("=", 1)
        steps, losses = read(path)
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
