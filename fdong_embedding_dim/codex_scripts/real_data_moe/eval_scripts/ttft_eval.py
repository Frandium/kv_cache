from __future__ import annotations

import argparse
import gc
import time

import torch

from .common import write_csv
from .swap_latency_eval import SwappingDecodeRuntime


def main() -> None:
    from transformers import AutoTokenizer
    try:
        from moe.analysis_utils import fixed_token_sequences
    except ModuleNotFoundError:
        from fdong_embedding_dim.codex_scripts.real_data_moe.analysis_utils import fixed_token_sequences

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--size", required=True, choices=("L", "M"))
    parser.add_argument("--method", required=True, choices=("baseline", "proposed"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-capacity", type=int, required=True)
    parser.add_argument("--prompt-lengths", default="32,128,512,1024,2048")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hardware-name", default="PPU-ZW810")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("real TTFT benchmark requires CUDA/PPU")
    lengths = sorted({int(item) for item in args.prompt_lengths.split(",") if item})
    if not lengths or min(lengths) < 1:
        raise ValueError("prompt lengths must be positive")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    prompts, _ = fixed_token_sequences(args.data_dir, tokenizer, args.repeats, max(lengths))
    runtime = SwappingDecodeRuntime(args.checkpoint, args.cache_capacity, device, max(lengths))
    rows: list[dict[str, object]] = []

    def timed_prefill(prompt: torch.Tensor, cache_state: str, repeat: int) -> None:
        if cache_state == "cold":
            runtime.reset(preserve_expert_cache=False)
        else:
            runtime.reset(preserve_expert_cache=False)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                runtime.prompt_forward(prompt.to(device, non_blocking=False))
            torch.cuda.synchronize(device)
            runtime.reset(preserve_expert_cache=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = runtime.prompt_forward(prompt.to(device, non_blocking=False))
            _ = logits[0, -1].argmax()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        stats = runtime.stats()
        row = {
            "size": args.size,
            "method": args.method,
            "step": runtime.step,
            "hardware": args.hardware_name,
            "device_name": torch.cuda.get_device_name(device),
            "evaluation_mode": "batch1_full_causal_prefill_to_first_output",
            "cache_state": cache_state,
            "cache_capacity": args.cache_capacity,
            "is_unlimited": int(args.cache_capacity >= runtime.config.num_tail_experts),
            "repeat": repeat,
            "prompt_length": int(prompt.numel()),
            "ttft_seconds": elapsed,
            "ttft_milliseconds": 1000.0 * elapsed,
            **stats,
            "loaded_gib": stats["bytes_loaded"] / 2**30,
        }
        rows.append(row)
        print(
            f"[ttft] {args.size}/{args.method} state={cache_state} "
            f"k={args.cache_capacity} prompt={prompt.numel()} repeat={repeat} "
            f"ms={row['ttft_milliseconds']:.6f} loads={stats['expert_loads']}",
            flush=True,
        )

    for length in lengths:
        for repeat, sequence in enumerate(prompts):
            prompt = sequence[:length].contiguous()
            timed_prefill(prompt, "cold", repeat)
            timed_prefill(prompt, "warm", repeat)

    for cache_state in ("cold", "warm"):
        for length in lengths:
            group = [
                row for row in rows
                if row["cache_state"] == cache_state and row["prompt_length"] == length
            ]
            aggregate = dict(group[0])
            aggregate["repeat"] = "mean"
            for field in (
                "ttft_seconds", "ttft_milliseconds", "expert_accesses", "expert_loads",
                "expert_evictions", "bytes_loaded", "loaded_gib",
            ):
                aggregate[field] = sum(float(row[field]) for row in group) / len(group)
            rows.append(aggregate)
    write_csv(args.output, rows)
    del runtime
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
