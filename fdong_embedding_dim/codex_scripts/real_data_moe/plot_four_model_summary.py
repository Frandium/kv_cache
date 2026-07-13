from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


MODELS = (
    "baseline",
    "proposed",
    "dense_active_matched",
    "proposed_baseline_active_matched",
)
LABELS = {
    "baseline": "Baseline MoE",
    "proposed": "F-MoE (active 33.54M)",
    "dense_active_matched": "Dense (active 33.53M)",
    "proposed_baseline_active_matched": "F-MoE (active 36.20M)",
}
COLORS = {
    "baseline": "#666666",
    "proposed": "#2A6FBB",
    "dense_active_matched": "#2E8B57",
    "proposed_baseline_active_matched": "#C05A32",
}


def read_metrics(path: Path) -> List[Dict[str, float]]:
    records = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        records[int(record["step"])] = float(record["loss"])
    return [{"step": step, "loss": records[step]} for step in sorted(records)]


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    averaged = (cumulative[window:] - cumulative[:-window]) / window
    return np.concatenate((np.full(window - 1, np.nan), averaged))


def normalized_entropy(shares: List[float]) -> float:
    return -sum(value * math.log(value) for value in shares if value > 0) / math.log(4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smooth-window", type=int, default=50)
    args = parser.parse_args()

    root = Path(args.experiment_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {name: read_metrics(root / name / "metrics.jsonl") for name in MODELS}
    test = json.loads((output_dir / "test_loss.json").read_text())
    continuity_standard = json.loads((root / "analysis" / "route_continuity.json").read_text())
    continuity_matched = json.loads(
        (root / "analysis_proposed_baseline_active_matched" / "route_continuity.json").read_text()
    )
    continuity = {
        "baseline": continuity_standard["baseline"],
        "proposed": continuity_standard["proposed"],
        "dense_active_matched": None,
        "proposed_baseline_active_matched": continuity_matched["proposed"],
    }

    summary = {}
    for name in MODELS:
        trailing = metrics[name][-50:]
        record = {
            "training_loss_last_500_steps_mean": float(
                np.mean([point["loss"] for point in trailing])
            ),
            "test_loss": test[name]["test_loss"],
            "perplexity": test[name]["perplexity"],
        }
        if continuity[name] is None:
            record.update(
                {
                    "expert_switches_per_100_tokens_4_layers": 0.0,
                    "stay_probability": 1.0,
                    "max_expert_share": None,
                    "mean_normalized_routing_entropy": None,
                    "swap_note": "no routed experts; all dense FFN parameters are resident",
                }
            )
        else:
            shares = continuity[name]["expert_shares_by_layer"]
            record.update(
                {
                    "expert_switches_per_100_tokens_4_layers": continuity[name]["mean_total_switches"],
                    "stay_probability": continuity[name]["stay_probability"],
                    "max_expert_share": max(max(layer) for layer in shares),
                    "mean_normalized_routing_entropy": float(
                        np.mean([normalized_entropy(layer) for layer in shares])
                    ),
                }
            )
        summary[name] = record

    (output_dir / "four_model_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for name in MODELS:
        steps = np.asarray([point["step"] for point in metrics[name]])
        losses = np.asarray([point["loss"] for point in metrics[name]])
        axes[0, 0].plot(
            steps,
            smooth(losses, args.smooth_window),
            color=COLORS[name],
            linewidth=2,
            label=LABELS[name],
        )
    axes[0, 0].set_title("Training loss (50-record moving mean)")
    axes[0, 0].set_xlabel("Optimizer step")
    axes[0, 0].set_ylabel("Next-token loss")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    x = np.arange(len(MODELS))
    test_losses = [summary[name]["test_loss"] for name in MODELS]
    axes[0, 1].bar(x, test_losses, color=[COLORS[name] for name in MODELS])
    axes[0, 1].set_title("Final checkpoint test loss (4 x 256 tokens)")
    axes[0, 1].set_ylabel("Next-token loss")
    axes[0, 1].set_xticks(x, [LABELS[name] for name in MODELS], rotation=18, ha="right")
    axes[0, 1].set_ylim(min(test_losses) - 0.03, max(test_losses) + 0.04)
    for index, value in enumerate(test_losses):
        axes[0, 1].text(index, value + 0.004, f"{value:.4f}", ha="center", fontsize=9)

    switches = [summary[name]["expert_switches_per_100_tokens_4_layers"] for name in MODELS]
    axes[1, 0].bar(x, switches, color=[COLORS[name] for name in MODELS])
    axes[1, 0].set_title("Dynamic expert switches per 100 tokens, 4 layers")
    axes[1, 0].set_ylabel("Route changes")
    axes[1, 0].set_xticks(x, [LABELS[name] for name in MODELS], rotation=18, ha="right")
    for index, value in enumerate(switches):
        label = "0 (dense resident)" if MODELS[index] == "dense_active_matched" else f"{value:.1f}"
        axes[1, 0].text(index, value + 4, label, ha="center", fontsize=9)

    moe_models = [name for name in MODELS if name != "dense_active_matched"]
    entropy = [summary[name]["mean_normalized_routing_entropy"] for name in moe_models]
    x_moe = np.arange(len(moe_models))
    axes[1, 1].bar(x_moe, entropy, color=[COLORS[name] for name in moe_models])
    axes[1, 1].set_title("Routing balance across four experts")
    axes[1, 1].set_ylabel("Mean normalized entropy (1 = balanced)")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_xticks(
        x_moe, [LABELS[name] for name in moe_models], rotation=18, ha="right"
    )
    for index, value in enumerate(entropy):
        axes[1, 1].text(index, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)

    figure.savefig(output_dir / "four_model_summary.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
