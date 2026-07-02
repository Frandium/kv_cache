from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

from .analysis_utils import fixed_token_sequences, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--proposed-checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-sequences", type=int, default=16)
    parser.add_argument("--num-tokens", type=int, default=100)
    return parser.parse_args()


@torch.no_grad()
def analyze_model(
    checkpoint: str,
    sequences: List[torch.Tensor],
    device: torch.device,
) -> Dict[str, object]:
    model, step = load_model(checkpoint, device)
    all_routes = []
    all_switches = []
    for sequence in sequences:
        _, diagnostics = model(sequence.unsqueeze(0).to(device))
        routes = np.stack(
            [diagnostics[f"layer_{layer}"]["route_indices"][0].cpu().numpy()
             for layer in range(model.config.num_hidden_layers)]
        )
        switches = (routes[:, 1:] != routes[:, :-1]).sum(axis=1)
        all_routes.append(routes)
        all_switches.append(switches)
    route_array = np.stack(all_routes)
    switch_array = np.stack(all_switches)
    expert_shares_by_layer = [
        [float((route_array[:, layer] == expert).mean()) for expert in range(model.config.num_tail_experts)]
        for layer in range(model.config.num_hidden_layers)
    ]
    result: Dict[str, object] = {
        "step": step,
        "routes": route_array,
        "switches": switch_array,
        "mean_switches_per_layer": switch_array.mean(axis=0).tolist(),
        "mean_total_switches": float(switch_array.sum(axis=1).mean()),
        "mean_total_loads_including_initial": float(
            (switch_array.sum(axis=1) + model.config.num_hidden_layers).mean()
        ),
        "expert_shares_by_layer": expert_shares_by_layer,
        "stay_probability": float(1.0 - switch_array.sum() / (
            switch_array.shape[0] * switch_array.shape[1] * (route_array.shape[2] - 1)
        )),
    }
    del model
    gc.collect()
    return result


def plot_results(results: Dict[str, Dict[str, object]], output_path: str) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    variants = ["baseline", "proposed"]
    for column, variant in enumerate(variants):
        routes = results[variant]["routes"][0]  # type: ignore[index]
        image = axes[0, column].imshow(
            routes, aspect="auto", interpolation="nearest", vmin=-0.5, vmax=3.5,
            cmap="tab10"
        )
        axes[0, column].set_title(
            f"{variant}: 100-token route trace (step {results[variant]['step']})"
        )
        axes[0, column].set_xlabel("Token position")
        axes[0, column].set_ylabel("Layer")
        axes[0, column].set_yticks(range(routes.shape[0]))
        figure.colorbar(image, ax=axes[0, column], ticks=range(4), label="Tail expert")

    x = np.arange(4)
    width = 0.36
    baseline = np.asarray(results["baseline"]["mean_switches_per_layer"])
    proposed = np.asarray(results["proposed"]["mean_switches_per_layer"])
    axes[1, 0].bar(x - width / 2, baseline, width, label="baseline")
    axes[1, 0].bar(x + width / 2, proposed, width, label="proposed")
    axes[1, 0].set_xticks(x, [f"Layer {i}" for i in x])
    axes[1, 0].set_ylabel("Expert switches / 99 transitions")
    axes[1, 0].set_title("Mean switches by layer")
    axes[1, 0].legend()

    totals = [
        results[variant]["mean_total_switches"] for variant in variants
    ]
    axes[1, 1].bar(variants, totals, color=["#777777", "#2a6fbb"])
    axes[1, 1].set_ylabel("Switches across all 4 layers")
    axes[1, 1].set_title("Mean dynamic swaps for each 100-token sequence")
    for index, value in enumerate(totals):
        axes[1, 1].text(index, float(value) + 1, f"{float(value):.1f}", ha="center")

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.num_tokens < 2:
        raise ValueError("num-tokens must be at least 2")
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    sequences, texts = fixed_token_sequences(
        args.data_dir, tokenizer, args.num_sequences, args.num_tokens
    )
    device = torch.device(args.device)
    results = {
        "baseline": analyze_model(args.baseline_checkpoint, sequences, device),
        "proposed": analyze_model(args.proposed_checkpoint, sequences, device),
    }
    serializable = {
        name: {key: value for key, value in result.items() if key not in {"routes", "switches"}}
        for name, result in results.items()
    }
    serializable["definition"] = {
        "switch": "route[t] != route[t-1] within the same layer",
        "initial_load": "one initial tail expert load per layer",
        "common_expert": "always resident and excluded from swap counts",
        "num_sequences": args.num_sequences,
        "num_tokens": args.num_tokens,
        "sample_preview": texts[0][:500],
    }
    with open(os.path.join(args.output_dir, "route_continuity.json"), "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)
    plot_results(results, os.path.join(args.output_dir, "route_continuity.png"))
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
