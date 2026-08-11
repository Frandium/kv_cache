from __future__ import annotations

import argparse
import gc
import math
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
try:
    from moe.model import ModelConfig, RealDataMoEForCausalLM, rotate_half
except ModuleNotFoundError:  # local source-tree execution
    from fdong_embedding_dim.codex_scripts.real_data_moe.model import (
        ModelConfig,
        RealDataMoEForCausalLM,
        rotate_half,
    )
from .common import write_csv


class ExpertWeights(NamedTuple):
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


class FixedSlotExpertCache:
    def __init__(self, weights: list[ExpertWeights], capacity: int, device: torch.device) -> None:
        if not 1 <= capacity <= len(weights):
            raise ValueError("invalid expert cache capacity")
        self.weights = weights
        self.capacity = capacity
        self.device = device
        prototype = weights[0]
        self.slots = [
            ExpertWeights(
                torch.empty_like(prototype.gate, device=device),
                torch.empty_like(prototype.up, device=device),
                torch.empty_like(prototype.down, device=device),
            )
            for _ in range(capacity)
        ]
        self.mapping: dict[int, int] = {}
        self.lru: OrderedDict[int, None] = OrderedDict()
        self.loads = 0
        self.evictions = 0
        self.hits = 0
        self.bytes_loaded = 0

    def reset(self) -> None:
        self.mapping.clear()
        self.lru.clear()
        self.reset_stats()
        if self.capacity == len(self.weights):
            for expert in range(len(self.weights)):
                self._load(expert)
            self.reset_stats()

    def reset_stats(self) -> None:
        self.loads = self.evictions = self.hits = self.bytes_loaded = 0

    def _load(self, expert: int) -> ExpertWeights:
        if len(self.mapping) < self.capacity:
            slot_index = next(index for index in range(self.capacity) if index not in self.mapping.values())
        else:
            evicted, _ = self.lru.popitem(last=False)
            slot_index = self.mapping.pop(evicted)
            self.evictions += 1
        source = self.weights[expert]
        target = self.slots[slot_index]
        target.gate.copy_(source.gate, non_blocking=False)
        target.up.copy_(source.up, non_blocking=False)
        target.down.copy_(source.down, non_blocking=False)
        self.mapping[expert] = slot_index
        self.lru[expert] = None
        self.loads += 1
        self.bytes_loaded += sum(tensor.numel() * tensor.element_size() for tensor in source)
        return target

    def get(self, expert: int) -> ExpertWeights:
        if expert in self.mapping:
            self.hits += 1
            self.lru.move_to_end(expert)
            return self.slots[self.mapping[expert]]
        return self._load(expert)


def pinned_bfloat16(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    return value.pin_memory() if torch.cuda.is_available() else value


class SwappingDecodeRuntime:
    def __init__(self, checkpoint: str, capacity: int, device: torch.device, max_positions: int) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        self.step = int(payload["step"])
        config = ModelConfig(**payload["model_config"])
        model = RealDataMoEForCausalLM(config)
        model.load_state_dict(payload["model"])
        del payload
        self.model = model.eval()
        self.config = config
        self.device = device
        self.capacity = capacity
        self.max_positions = max_positions
        self.expert_caches: list[FixedSlotExpertCache] = []

        for layer in self.model.layers:
            weights = [
                ExpertWeights(
                    pinned_bfloat16(expert.gate_proj.weight),
                    pinned_bfloat16(expert.up_proj.weight),
                    pinned_bfloat16(expert.down_proj.weight),
                )
                for expert in layer.moe.tail_experts
            ]
            self.expert_caches.append(FixedSlotExpertCache(weights, capacity, device))
            layer.moe.tail_experts = torch.nn.ModuleList()
            layer.input_layernorm.to(device=device, dtype=torch.bfloat16)
            layer.self_attn.to(device=device, dtype=torch.bfloat16)
            layer.post_attention_layernorm.to(device=device, dtype=torch.bfloat16)
            layer.moe.common_expert.to(device=device, dtype=torch.bfloat16)
            layer.moe.router.to(device=device, dtype=torch.bfloat16)
            if config.orthogonalize_tail:
                # The constraint basis is model state derived from the common
                # expert. Build it before request timing, as a production load
                # path would, rather than charging its SVD to TTFT/decode.
                layer.moe._refresh_common_basis(device, torch.bfloat16)
        self.model.embed_tokens.to(device=device, dtype=torch.bfloat16)
        self.model.norm.to(device=device, dtype=torch.bfloat16)

        positions = torch.arange(max_positions, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, device=device).float() / config.head_dim)
        )
        angles = torch.outer(positions, inv_freq)
        embedding = torch.repeat_interleave(angles, repeats=2, dim=-1)
        self.cos = embedding.cos().to(torch.bfloat16)
        self.sin = embedding.sin().to(torch.bfloat16)
        self.reset()

    def reset(self, preserve_expert_cache: bool = False) -> None:
        self.position = 0
        self.kv = [
            (
                torch.empty(
                    1,
                    self.config.num_key_value_heads,
                    self.max_positions,
                    self.config.head_dim,
                    device=self.device,
                    dtype=torch.bfloat16,
                ),
                torch.empty(
                    1,
                    self.config.num_key_value_heads,
                    self.max_positions,
                    self.config.head_dim,
                    device=self.device,
                    dtype=torch.bfloat16,
                ),
            )
            for _ in self.model.layers
        ]
        self.attention_history = [deque(maxlen=self.config.router_window) for _ in self.model.layers]
        self.route_counts = torch.zeros(
            self.config.num_hidden_layers, self.config.num_tail_experts, dtype=torch.long
        )
        for cache in self.expert_caches:
            if preserve_expert_cache:
                cache.reset_stats()
            else:
                cache.reset()

    def reset_decode_stats(self) -> None:
        self.route_counts.zero_()
        for cache in self.expert_caches:
            cache.reset_stats()

    def cached_attention(self, layer_index: int, x: torch.Tensor) -> torch.Tensor:
        attention = self.model.layers[layer_index].self_attn
        batch, tokens, _ = x.shape
        if tokens != 1:
            raise ValueError("cached decoder accepts one token at a time")
        q = attention.q_proj(x).view(batch, 1, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)
        k = attention.k_proj(x).view(batch, 1, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2)
        v = attention.v_proj(x).view(batch, 1, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2)
        q = attention.q_norm(q)
        k = attention.k_norm(k)
        cos = self.cos[self.position][None, None, None]
        sin = self.sin[self.position][None, None, None]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        k_cache, v_cache = self.kv[layer_index]
        k_cache[:, :, self.position : self.position + 1].copy_(k)
        v_cache[:, :, self.position : self.position + 1].copy_(v)
        k_full = k_cache[:, :, : self.position + 1]
        v_full = v_cache[:, :, : self.position + 1]
        repeats = self.config.num_attention_heads // self.config.num_key_value_heads
        if repeats > 1:
            k_compute = k_full.repeat_interleave(repeats, dim=1)
            v_compute = v_full.repeat_interleave(repeats, dim=1)
        else:
            k_compute, v_compute = k_full, v_full
        output = F.scaled_dot_product_attention(q, k_compute, v_compute, is_causal=False)
        return attention.o_proj(output.transpose(1, 2).reshape(batch, 1, -1))

    def full_attention(self, layer_index: int, x: torch.Tensor) -> torch.Tensor:
        """Full causal attention using the runtime-sized RoPE table."""
        attention = self.model.layers[layer_index].self_attn
        batch, tokens, _ = x.shape
        q = attention.q_proj(x).view(
            batch, tokens, self.config.num_attention_heads, self.config.head_dim
        ).transpose(1, 2)
        k = attention.k_proj(x).view(
            batch, tokens, self.config.num_key_value_heads, self.config.head_dim
        ).transpose(1, 2)
        v = attention.v_proj(x).view(
            batch, tokens, self.config.num_key_value_heads, self.config.head_dim
        ).transpose(1, 2)
        q = attention.q_norm(q)
        k = attention.k_norm(k)
        cos = self.cos[:tokens][None, None]
        sin = self.sin[:tokens][None, None]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        repeats = self.config.num_attention_heads // self.config.num_key_value_heads
        if repeats > 1:
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return attention.o_proj(output.transpose(1, 2).reshape(batch, tokens, -1))

    @torch.inference_mode()
    def token_step(self, token: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(token.view(1, 1))
        for layer_index, layer in enumerate(self.model.layers):
            attention_output = self.cached_attention(layer_index, layer.input_layernorm(x))
            residual = x + attention_output
            expert_input = layer.post_attention_layernorm(residual)
            if self.config.router_input == "residual":
                router_input = expert_input
            elif self.config.router_input == "attention":
                router_input = attention_output
            else:
                history = self.attention_history[layer_index]
                history.append(attention_output)
                router_input = (
                    torch.stack(tuple(history), dim=0)
                    .float()
                    .mean(dim=0)
                    .to(attention_output.dtype)
                )
            route = int(layer.moe.router(router_input).argmax(dim=-1).item())
            self.route_counts[layer_index, route] += 1
            common = layer.moe.common_expert(expert_input)
            weights = self.expert_caches[layer_index].get(route)
            tail = F.linear(
                F.silu(F.linear(expert_input, weights.gate))
                * F.linear(expert_input, weights.up),
                weights.down,
            )
            if self.config.orthogonalize_tail:
                tail = layer.moe._orthogonalize(tail)
            x = residual + common + tail
        self.position += 1
        x = self.model.norm(x)
        return F.linear(x, self.model.embed_tokens.weight)

    @torch.inference_mode()
    def prompt_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full causal prefill with batched tokens and demand-loaded experts."""
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.shape[0] != 1:
            raise ValueError("TTFT runtime currently requires batch size one")
        if input_ids.shape[1] > self.max_positions:
            raise ValueError("prompt exceeds allocated positions")
        x = self.model.embed_tokens(input_ids)
        for layer_index, layer in enumerate(self.model.layers):
            attention_output = self.full_attention(layer_index, layer.input_layernorm(x))
            residual = x + attention_output
            expert_input = layer.post_attention_layernorm(residual)
            if self.config.router_input == "residual":
                router_input = expert_input
            elif self.config.router_input == "attention":
                router_input = attention_output
            else:
                # Same causal moving mean as the training/evaluation model.
                prefix = torch.cat(
                    (
                        torch.zeros_like(attention_output[:, :1], dtype=torch.float32),
                        torch.cumsum(attention_output.float(), dim=1),
                    ),
                    dim=1,
                )
                positions = torch.arange(attention_output.shape[1], device=self.device)
                starts = (positions + 1 - self.config.router_window).clamp_min(0)
                totals = prefix[:, positions + 1] - prefix[:, starts]
                counts = (positions + 1 - starts).view(1, -1, 1)
                router_input = (totals / counts).to(attention_output.dtype)
            routes = layer.moe.router(router_input).argmax(dim=-1)
            common = layer.moe.common_expert(expert_input)
            tail = torch.zeros_like(expert_input)
            for expert_index in range(self.config.num_tail_experts):
                mask = routes == expert_index
                count = int(mask.sum().item())
                if count == 0:
                    continue
                self.route_counts[layer_index, expert_index] += count
                weights = self.expert_caches[layer_index].get(expert_index)
                selected = expert_input[mask]
                output = F.linear(
                    F.silu(F.linear(selected, weights.gate))
                    * F.linear(selected, weights.up),
                    weights.down,
                )
                if self.config.orthogonalize_tail:
                    output = layer.moe._orthogonalize(output)
                tail[mask] = output
            x = residual + common + tail
        self.position = input_ids.shape[1]
        return F.linear(self.model.norm(x[:, -1:]), self.model.embed_tokens.weight)

    def stats(self) -> dict[str, int]:
        return {
            "expert_accesses": sum(cache.loads + cache.hits for cache in self.expert_caches),
            "expert_loads": sum(cache.loads for cache in self.expert_caches),
            "expert_evictions": sum(cache.evictions for cache in self.expert_caches),
            "bytes_loaded": sum(cache.bytes_loaded for cache in self.expert_caches),
        }


def main() -> None:
    from transformers import AutoTokenizer
    try:
        from moe.analysis_utils import fixed_token_sequences
    except ModuleNotFoundError:  # local source-tree execution
        from fdong_embedding_dim.codex_scripts.real_data_moe.analysis_utils import fixed_token_sequences

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--size", required=True, choices=("L", "M"))
    parser.add_argument("--method", required=True, choices=("baseline", "proposed"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-capacity", type=int, required=True)
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hardware-name", default="PPU-ZW810")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("real swapping benchmark requires CUDA/PPU")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    prompts, _ = fixed_token_sequences(args.data_dir, tokenizer, args.repeats, args.prompt_length)
    runtime = SwappingDecodeRuntime(
        args.checkpoint,
        args.cache_capacity,
        device,
        args.prompt_length + args.decode_tokens + 1,
    )
    rows = []
    for repeat, prompt in enumerate(prompts):
        runtime.reset()
        logits = None
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for token in prompt.to(device):
                logits = runtime.token_step(token)
            runtime.reset_decode_stats()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            for _ in range(args.decode_tokens):
                assert logits is not None
                next_token = logits[0, -1].argmax()
                logits = runtime.token_step(next_token)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
        stats = runtime.stats()
        rows.append(
            {
                "size": args.size,
                "method": args.method,
                "step": runtime.step,
                "hardware": args.hardware_name,
                "device_name": torch.cuda.get_device_name(device),
                "evaluation_mode": "autoregressive_kv_cached_decode",
                "cache_capacity": args.cache_capacity,
                "is_unlimited": int(args.cache_capacity >= runtime.config.num_tail_experts),
                "repeat": repeat,
                "prompt_length": args.prompt_length,
                "decode_tokens": args.decode_tokens,
                "elapsed_seconds": elapsed,
                "milliseconds_per_token": 1000.0 * elapsed / args.decode_tokens,
                "tokens_per_second": args.decode_tokens / elapsed,
                **stats,
                "loads_per_100_token_layer": 100.0 * stats["expert_loads"] / (args.decode_tokens * runtime.config.num_hidden_layers),
                "cache_hit_rate": 1.0 - stats["expert_loads"] / max(stats["expert_accesses"], 1),
                "loaded_gib": stats["bytes_loaded"] / 2**30,
            }
        )
        print(
            f"[swap] {args.size}/{args.method} k={args.cache_capacity} "
            f"repeat={repeat} ms/token={rows[-1]['milliseconds_per_token']:.6f} "
            f"loads={stats['expert_loads']}",
            flush=True,
        )
    aggregate = dict(rows[0])
    aggregate["repeat"] = "mean"
    for field in (
        "elapsed_seconds",
        "milliseconds_per_token",
        "tokens_per_second",
        "expert_accesses",
        "expert_loads",
        "expert_evictions",
        "bytes_loaded",
        "loads_per_100_token_layer",
        "cache_hit_rate",
        "loaded_gib",
    ):
        aggregate[field] = sum(float(row[field]) for row in rows) / len(rows)
    rows.append(aggregate)
    write_csv(args.output, rows)
    del runtime
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
