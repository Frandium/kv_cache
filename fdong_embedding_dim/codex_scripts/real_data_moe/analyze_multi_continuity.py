from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from .analysis_utils import fixed_token_sequences
from .analyze_route_continuity import analyze_model
from transformers import AutoTokenizer
import torch


def entropy(shares: list[float]) -> float:
    return -sum(value * math.log(value) for value in shares if value > 0) / math.log(len(shares))


def parse_cache_capacities(value: str) -> tuple[int, ...]:
    try:
        capacities = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "cache capacities must be comma-separated integers"
        ) from exc
    if not capacities or any(capacity < 1 for capacity in capacities):
        raise argparse.ArgumentTypeError("cache capacities must be positive")
    return capacities


def simulate_lru_cache(
    routes: np.ndarray,
    capacities: Iterable[int],
) -> dict[str, dict[str, object]]:
    """Simulate an independent per-layer LRU expert cache for each sequence."""
    if routes.ndim != 3:
        raise ValueError("routes must have shape [sequences, layers, tokens]")
    num_sequences, num_layers, num_tokens = routes.shape
    if num_sequences < 1 or num_layers < 1 or num_tokens < 1:
        raise ValueError("routes dimensions must be positive")

    capacities = tuple(capacities)
    num_experts = int(routes.max()) + 1
    # Each sequence-layer pair is an independent cache stream. For LRU, an
    # access misses a K-slot cache iff at least K distinct experts have been
    # accessed since the previous access to the requested expert.
    streams = routes.reshape(num_sequences * num_layers, num_tokens)
    num_streams = streams.shape[0]
    stream_indices = np.arange(num_streams)
    last_seen = np.full((num_streams, num_experts), -1, dtype=np.int32)
    loads_by_capacity = {
        capacity: np.zeros(num_streams, dtype=np.int64) for capacity in capacities
    }
    evictions_by_capacity = {
        capacity: np.zeros(num_streams, dtype=np.int64) for capacity in capacities
    }

    for token_index in range(num_tokens):
        experts = streams[:, token_index]
        previous = last_seen[stream_indices, experts]
        first_access = previous < 0
        reuse_distance = (last_seen > previous[:, None]).sum(axis=1)
        unique_seen = (last_seen >= 0).sum(axis=1)
        for capacity in capacities:
            misses = first_access | (reuse_distance >= capacity)
            loads_by_capacity[capacity] += misses
            evictions_by_capacity[capacity] += misses & (unique_seen >= capacity)
        last_seen[stream_indices, experts] = token_index

    summaries: dict[str, dict[str, object]] = {}
    for capacity in capacities:
        loads = loads_by_capacity[capacity].reshape(num_sequences, num_layers)
        evictions = evictions_by_capacity[capacity].reshape(
            num_sequences, num_layers
        )
        total_accesses = num_sequences * num_layers * num_tokens
        total_transitions = num_sequences * num_layers * max(num_tokens - 1, 1)
        summaries[str(capacity)] = {
            "capacity_experts_per_layer": capacity,
            "policy": "LRU",
            "cache_reset": "empty at the start of each sequence",
            "mean_total_loads_including_cold": float(loads.sum(axis=1).mean()),
            "mean_total_evictions": float(evictions.sum(axis=1).mean()),
            "loads_per_100_token_layer_including_cold": float(
                100.0 * loads.sum() / total_accesses
            ),
            "evictions_per_100_transitions": float(
                100.0 * evictions.sum() / total_transitions
            ),
            "mean_loads_per_layer_including_cold": loads.mean(axis=0).tolist(),
            "mean_evictions_per_layer": evictions.mean(axis=0).tolist(),
        }

    if "1" in summaries:
        reference_loads = float(summaries["1"]["mean_total_loads_including_cold"])
        reference_evictions = float(summaries["1"]["mean_total_evictions"])
        for summary in summaries.values():
            loads = float(summary["mean_total_loads_including_cold"])
            evictions = float(summary["mean_total_evictions"])
            summary["load_reduction_vs_k1"] = (
                1.0 - loads / reference_loads if reference_loads else 0.0
            )
            summary["eviction_reduction_vs_k1"] = (
                1.0 - evictions / reference_evictions if reference_evictions else 0.0
            )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="NAME=CHECKPOINT")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-sequences", type=int, default=16)
    parser.add_argument("--num-tokens", type=int, default=100)
    parser.add_argument(
        "--cache-capacities",
        type=parse_cache_capacities,
        default=(1, 2, 3, 4),
        help="comma-separated per-layer LRU expert cache capacities",
    )
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
        num_experts = len(shares[0])
        if any(capacity > num_experts for capacity in args.cache_capacities):
            raise ValueError(
                f"cache capacity exceeds the model's {num_experts} tail experts"
            )
        cache_simulation = simulate_lru_cache(raw["routes"], args.cache_capacities)
        results[name] = {
            "checkpoint": str(Path(checkpoint).resolve()),
            "step": raw["step"],
            "mean_switches_per_layer": raw["mean_switches_per_layer"],
            "mean_total_switches": raw["mean_total_switches"],
            "mean_total_loads_including_initial": raw["mean_total_loads_including_initial"],
            "stay_probability": raw["stay_probability"],
            "switch_probability": 1.0 - raw["stay_probability"],
            "maximum_switches": args.num_tokens - 1,
            "maximum_total_switches": (args.num_tokens - 1) * len(shares),
            "expert_shares_by_layer": shares,
            "mean_normalized_entropy": sum(entropy(layer) for layer in shares) / len(shares),
            "cache_simulation": cache_simulation,
        }
        print(name, json.dumps(results[name]), flush=True)

    print("CACHE_BUDGET_SUMMARY_BEGIN", flush=True)
    for name, result in results.items():
        cache_simulation = result["cache_simulation"]
        for capacity in args.cache_capacities:
            cache = cache_simulation[str(capacity)]
            load_reduction = float(cache.get("load_reduction_vs_k1", 0.0)) * 100.0
            eviction_reduction = float(
                cache.get("eviction_reduction_vs_k1", 0.0)
            ) * 100.0
            print(
                "CACHE_BUDGET "
                f"run={name} policy=LRU k={capacity} "
                f"mean_loads_including_cold="
                f"{float(cache['mean_total_loads_including_cold']):.6f} "
                f"mean_evictions={float(cache['mean_total_evictions']):.6f} "
                f"loads_per_100_token_layer_including_cold="
                f"{float(cache['loads_per_100_token_layer_including_cold']):.6f} "
                f"evictions_per_100_transitions="
                f"{float(cache['evictions_per_100_transitions']):.6f} "
                f"load_reduction_vs_k1_pct={load_reduction:.6f} "
                f"eviction_reduction_vs_k1_pct={eviction_reduction:.6f}",
                flush=True,
            )
    print("CACHE_BUDGET_SUMMARY_END", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "continuity.json").write_text(json.dumps(results, indent=2) + "\n")
    names = list(results)
    switches = [results[name]["mean_total_switches"] for name in names]
    rates = [results[name]["switch_probability"] * 100 for name in names]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bars = axis.bar(names, switches)
    axis.set_ylabel(f"Expert switches per {args.num_tokens} tokens across all layers")
    axis.set_title("Expert activation continuity")
    for bar, value, rate in zip(bars, switches, rates):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}\n({rate:.1f}%)",
            ha="center",
            va="bottom",
        )
    figure.savefig(output_dir / "continuity.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
