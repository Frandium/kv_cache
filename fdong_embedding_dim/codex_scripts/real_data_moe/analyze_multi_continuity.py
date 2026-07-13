from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from .analysis_utils import fixed_token_sequences
from .analyze_route_continuity import analyze_model
from transformers import AutoTokenizer
import torch


def entropy(shares: list[float]) -> float:
    return -sum(value * math.log(value) for value in shares if value > 0) / math.log(len(shares))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="NAME=CHECKPOINT")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-sequences", type=int, default=16)
    parser.add_argument("--num-tokens", type=int, default=100)
    args = parser.parse_args()

    runs = dict(item.split("=", 1) for item in args.run)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    sequences, _ = fixed_token_sequences(
        args.data_dir, tokenizer, args.num_sequences, args.num_tokens
    )
    device = torch.device(args.device)
    results = {}
    for name, checkpoint in runs.items():
        raw = analyze_model(checkpoint, sequences, device)
        shares = raw["expert_shares_by_layer"]
        results[name] = {
            "checkpoint": str(Path(checkpoint).resolve()),
            "step": raw["step"],
            "mean_switches_per_layer": raw["mean_switches_per_layer"],
            "mean_total_switches": raw["mean_total_switches"],
            "mean_total_loads_including_initial": raw["mean_total_loads_including_initial"],
            "stay_probability": raw["stay_probability"],
            "expert_shares_by_layer": shares,
            "mean_normalized_entropy": sum(entropy(layer) for layer in shares) / len(shares),
        }
        print(name, json.dumps(results[name]), flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "continuity.json").write_text(json.dumps(results, indent=2) + "\n")
    names = list(results)
    switches = [results[name]["mean_total_switches"] for name in names]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bars = axis.bar(names, switches)
    axis.set_ylabel(f"Expert switches per {args.num_tokens} tokens across all layers")
    axis.set_title("Expert activation continuity")
    for bar, value in zip(bars, switches):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom")
    figure.savefig(output_dir / "continuity.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
