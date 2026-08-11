from __future__ import annotations

import argparse
import math
import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
from .common import write_csv


def setup() -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world, torch.device("cuda", local_rank)
    return 0, 1, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def lru_counts(routes: torch.Tensor, capacities: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    # routes: [layers, tokens], one independent cache per layer.
    layers = routes.shape[0]
    loads = torch.zeros(len(capacities), layers, dtype=torch.long)
    evictions = torch.zeros_like(loads)
    for layer in range(layers):
        stream = routes[layer].tolist()
        for capacity_index, capacity in enumerate(capacities):
            cache: OrderedDict[int, None] = OrderedDict()
            for expert in stream:
                if expert in cache:
                    cache.move_to_end(expert)
                    continue
                loads[capacity_index, layer] += 1
                if len(cache) >= capacity:
                    cache.popitem(last=False)
                    evictions[capacity_index, layer] += 1
                cache[expert] = None
    return loads, evictions


def reduce_sum(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    reduced = value.to(device)
    if dist.is_initialized():
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced.cpu()


@torch.no_grad()
def route_only_forward(model: torch.nn.Module, input_ids: torch.Tensor) -> list[torch.Tensor]:
    """Collect routes without allocating the 152k-vocabulary logits tensor."""
    x = model.embed_tokens(input_ids)
    routes = []
    for layer in model.layers:
        x, diagnostics = layer(x)
        routes.append(diagnostics["route_indices"])
    return routes


def main() -> None:
    from transformers import AutoTokenizer
    try:
        from moe.analysis_utils import fixed_token_sequences, load_model
    except ModuleNotFoundError:  # local source-tree execution
        from fdong_embedding_dim.codex_scripts.real_data_moe.analysis_utils import (
            fixed_token_sequences,
            load_model,
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--size", required=True, choices=("L", "M"))
    parser.add_argument("--method", required=True, choices=("baseline", "proposed"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-sequences", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--cache-capacities", default="1,2,4,8")
    args = parser.parse_args()

    capacities = list(dict.fromkeys(int(item) for item in args.cache_capacities.split(",")))
    if not capacities or min(capacities) < 1:
        raise ValueError("cache capacities must be positive")
    rank, world, device = setup()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    sequences, _ = fixed_token_sequences(
        args.data_dir, tokenizer, args.num_sequences, args.sequence_length
    )
    local_sequences = sequences[rank::world]
    model, step = load_model(args.checkpoint, device)
    layers = model.config.num_hidden_layers
    experts = model.config.num_tail_experts
    if max(capacities) > experts:
        raise ValueError(f"cache capacity exceeds expert count {experts}")

    expert_counts = torch.zeros(layers, experts, dtype=torch.long)
    loads = torch.zeros(len(capacities), layers, dtype=torch.long)
    evictions = torch.zeros_like(loads)
    switches = torch.zeros(layers, dtype=torch.long)
    local_sequence_count = 0
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad()
    with torch.inference_mode(), autocast:
        for sequence in local_sequences:
            route_tensors = route_only_forward(model, sequence.unsqueeze(0).to(device))
            routes = torch.stack(
                [route_tensors[layer][0].cpu() for layer in range(layers)]
            )
            for layer in range(layers):
                expert_counts[layer] += torch.bincount(routes[layer], minlength=experts)
            switches += (routes[:, 1:] != routes[:, :-1]).sum(dim=1)
            sequence_loads, sequence_evictions = lru_counts(routes, capacities)
            loads += sequence_loads
            evictions += sequence_evictions
            local_sequence_count += 1

    expert_counts = reduce_sum(expert_counts, device)
    loads = reduce_sum(loads, device)
    evictions = reduce_sum(evictions, device)
    switches = reduce_sum(switches, device)
    sequence_count = reduce_sum(torch.tensor([local_sequence_count]), device).item()

    if rank == 0:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        load_rows = []
        summary_rows = []
        total_per_layer = sequence_count * args.sequence_length
        entropies = []
        cvs = []
        for layer in range(layers):
            shares = expert_counts[layer].double() / max(total_per_layer, 1)
            positive = shares[shares > 0]
            entropy = float(-(positive * positive.log()).sum() / math.log(experts))
            cv = float(shares.std(unbiased=False) / shares.mean().clamp_min(1e-12))
            entropies.append(entropy)
            cvs.append(cv)
            for expert in range(experts):
                load_rows.append(
                    {
                        "size": args.size,
                        "method": args.method,
                        "step": step,
                        "layer": layer,
                        "expert": expert,
                        "activation_count": int(expert_counts[layer, expert]),
                        "activation_share": float(shares[expert]),
                    }
                )
        all_shares = expert_counts.double() / expert_counts.sum(dim=1, keepdim=True).clamp_min(1)
        summary_rows.append(
            {
                "size": args.size,
                "method": args.method,
                "step": step,
                "num_sequences": int(sequence_count),
                "sequence_length": args.sequence_length,
                "mean_normalized_entropy": sum(entropies) / len(entropies),
                "mean_coefficient_of_variation": sum(cvs) / len(cvs),
                "minimum_expert_share": float(all_shares.min()),
                "maximum_expert_share": float(all_shares.max()),
                "max_to_min_share_ratio": float(all_shares.max() / all_shares.min().clamp_min(1e-12)),
                "zero_load_expert_layer_pairs": int((expert_counts == 0).sum()),
            }
        )
        continuity_rows = []
        total_accesses = sequence_count * layers * args.sequence_length
        total_transitions = sequence_count * layers * (args.sequence_length - 1)
        k1_loads = int(loads[capacities.index(1)].sum()) if 1 in capacities else None
        for index, capacity in enumerate(capacities):
            total_loads = int(loads[index].sum())
            total_evictions = int(evictions[index].sum())
            continuity_rows.append(
                {
                    "size": args.size,
                    "method": args.method,
                    "step": step,
                    "evaluation_mode": "teacher_forced_sequence",
                    "cache_capacity": capacity,
                    "is_unlimited": int(capacity >= experts),
                    "policy": "LRU",
                    "num_sequences": int(sequence_count),
                    "sequence_length": args.sequence_length,
                    "total_expert_accesses": int(total_accesses),
                    "total_loads_including_cold": total_loads,
                    "total_evictions": total_evictions,
                    "loads_per_100_token_layer": 100.0 * total_loads / total_accesses,
                    "evictions_per_100_transitions": 100.0 * total_evictions / total_transitions,
                    "cache_hit_rate": 1.0 - total_loads / total_accesses,
                    "load_reduction_vs_k1": (1.0 - total_loads / k1_loads) if k1_loads else 0.0,
                    "switch_probability": float(switches.sum() / total_transitions),
                }
            )
        write_csv(output / "expert_load.csv", load_rows)
        write_csv(output / "expert_load_summary.csv", summary_rows)
        write_csv(output / "continuity_by_budget.csv", continuity_rows)
        print(f"[routing] completed {args.size}/{args.method} step={step}", flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
