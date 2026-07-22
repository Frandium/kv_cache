from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
import random
import sys
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from .data import DCLMTokenStream
from .model import ModelConfig, RealDataMoEForCausalLM, parameter_counts
from .train import Tee, append_metric, format_duration, prepare_metrics_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "proposed", "routing_only"), required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--data-pattern", default="*.txt")
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=0, help="0 trains indefinitely")
    parser.add_argument("--save-every", type=int, default=4_000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4, help="per-GPU micro batch")
    parser.add_argument("--sequence-length", type=int, default=1_024)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--decay-steps", type=int, default=50_000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--load-balance-weight",
        type=float,
        default=0.0,
        help="coefficient for Switch-style router load-balance auxiliary loss",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--hidden-size", type=int, default=1_536)
    parser.add_argument("--vocab-size", type=int, default=0, help="0 uses tokenizer length")
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--num-heads", type=int, default=24)
    parser.add_argument("--num-kv-heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--common-intermediate-size", type=int, required=True)
    parser.add_argument("--tail-intermediate-size", type=int, required=True)
    parser.add_argument("--num-tail-experts", type=int, default=4)
    parser.add_argument("--num-experts-per-token", type=int, default=1)
    parser.add_argument("--router-window", type=int, default=16)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16", "none"), default="bfloat16")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("distributed trainer requires CUDA")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state(device: torch.device) -> Dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
    }


def restore_rng(state: Dict[str, object], device: torch.device) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
    torch.cuda.set_rng_state(state["cuda"], device)  # type: ignore[arg-type]


def learning_rate(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.learning_rate * (step + 1) / max(args.warmup_steps, 1)
    if step >= args.decay_steps:
        return args.min_learning_rate
    progress = (step - args.warmup_steps) / max(args.decay_steps - args.warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_learning_rate + (args.learning_rate - args.min_learning_rate) * cosine


def build_config(args: argparse.Namespace, vocab_size: int) -> ModelConfig:
    shared = {
        "vocab_size": vocab_size,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_layers,
        "num_attention_heads": args.num_heads,
        "num_key_value_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "max_position_embeddings": max(2_048, args.sequence_length),
        "num_tail_experts": args.num_tail_experts,
        "num_experts_per_token": args.num_experts_per_token,
        "common_intermediate_size": args.common_intermediate_size,
        "tail_intermediate_size": args.tail_intermediate_size,
        "router_window": args.router_window,
        "orthogonalize_tail": False,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    if args.variant == "baseline":
        return ModelConfig.baseline(**shared)
    return ModelConfig.proposed(**shared)


def resolve_resume(output_dir: str, resume: str) -> Optional[str]:
    if resume == "none":
        return None
    if resume != "auto":
        return resume
    latest = os.path.join(output_dir, "latest.pt")
    return latest if os.path.exists(latest) else None


def model_module(model: DDP) -> RealDataMoEForCausalLM:
    return model.module  # type: ignore[return-value]


def current_route_counts(model: RealDataMoEForCausalLM) -> torch.Tensor:
    counts = torch.zeros(model.config.num_tail_experts, device=next(model.parameters()).device)
    for layer in model.layers:
        if hasattr(layer, "moe") and layer.moe.last_route_counts is not None:
            counts += layer.moe.last_route_counts.to(counts.device)
    return counts


def load_balance_loss(
    diagnostics: Dict[str, Dict[str, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    losses = [
        layer_stats["load_balance_loss"]
        for layer_stats in diagnostics.values()
        if "load_balance_loss" in layer_stats
    ]
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def save_checkpoint(
    model: DDP,
    optimizer: torch.optim.Optimizer,
    stream: DCLMTokenStream,
    step: int,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    local_runtime = {"stream": stream.state_dict(), "rng": rng_state(device)}
    runtime_states = [None for _ in range(world_size)]
    dist.all_gather_object(runtime_states, local_runtime)
    if rank == 0:
        module = model_module(model)
        path = os.path.join(args.output_dir, f"checkpoint-{step:07d}.pt")
        payload = {
            "step": step,
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "runtime_states": runtime_states,
            "world_size": world_size,
            "model_config": module.config_dict(),
            "train_args": vars(args),
        }
        torch.save(payload, path)
        latest = os.path.join(args.output_dir, "latest.pt")
        temporary = latest + ".tmp"
        if os.path.lexists(temporary):
            os.unlink(temporary)
        os.symlink(os.path.basename(path), temporary)
        os.replace(temporary, latest)
        print(f"[checkpoint] {path}", flush=True)
    dist.barrier()


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = setup_distributed()
    is_main = rank == 0
    os.makedirs(args.output_dir, exist_ok=True)
    if is_main:
        log_handle = open(os.path.join(args.output_dir, "train.log"), "a", encoding="utf-8", buffering=1)
        sys.stdout = Tee(sys.stdout, log_handle)  # type: ignore[assignment]
        sys.stderr = Tee(sys.stderr, log_handle)  # type: ignore[assignment]

    # Identical model initialization on every rank; rank-specific data streams
    # are independent because DCLMTokenStream uses a local Random instance.
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    model_vocab_size = args.vocab_size if args.vocab_size > 0 else len(tokenizer)
    if max(tokenizer.get_vocab().values()) >= model_vocab_size:
        raise ValueError("model vocab size does not cover every tokenizer id")
    config = build_config(args, model_vocab_size)
    module = RealDataMoEForCausalLM(config).to(device)
    model = DDP(
        module,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    stream = DCLMTokenStream(
        args.data_dir,
        tokenizer,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
        file_pattern=args.data_pattern,
    )
    step = 0
    resume_path = resolve_resume(args.output_dir, args.resume)
    if resume_path is not None:
        payload = torch.load(resume_path, map_location="cpu", weights_only=False, mmap=True)
        if payload["world_size"] != world_size:
            raise RuntimeError("checkpoint world size differs from current world size")
        checkpoint_config = ModelConfig(**payload["model_config"])
        if asdict(checkpoint_config) != module.config_dict():
            raise RuntimeError("checkpoint model config does not match current config")
        module.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        stream.load_state_dict(payload["runtime_states"][rank]["stream"])
        restore_rng(payload["runtime_states"][rank]["rng"], device)
        step = int(payload["step"])
        del payload
        if is_main:
            print(f"[resume] {resume_path} at step {step}")

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    if is_main:
        prepare_metrics_file(metrics_path, step)
        counts = parameter_counts(module)
        global_batch = args.batch_size * world_size * args.gradient_accumulation
        tokens_per_step = global_batch * args.sequence_length
        print("[config]", json.dumps(asdict(config), sort_keys=True))
        print("[parameters]", json.dumps(counts, sort_keys=True))
        print(
            f"[distributed] world_size={world_size} global_batch={global_batch} "
            f"tokens_per_step={tokens_per_step:,} rank_files={len(stream.files)}"
        )
        print(f"[train] start={step} target={'unbounded' if args.max_steps == 0 else args.max_steps}")
    dist.barrier()

    amp_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "none": None,
    }[args.amp_dtype]
    recent_step_seconds = deque(maxlen=50)
    model.train()

    try:
        while args.max_steps == 0 or step < args.max_steps:
            started = time.perf_counter()
            module.set_training_step(step)
            optimizer.zero_grad(set_to_none=True)
            local_ce_loss = 0.0
            local_aux_loss = 0.0
            route_totals = torch.zeros(config.num_tail_experts, device=device)
            for micro_step in range(args.gradient_accumulation):
                input_ids, labels = stream.next_batch(args.batch_size, args.sequence_length, device)
                sync_context = model.no_sync() if micro_step + 1 < args.gradient_accumulation else nullcontext()
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if amp_dtype is not None
                    else nullcontext()
                )
                with sync_context, autocast_context:
                    logits, diagnostics = model(input_ids)
                    ce_loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
                    )
                    aux_loss = load_balance_loss(diagnostics, device)
                    loss = ce_loss + args.load_balance_weight * aux_loss
                    scaled_loss = loss / args.gradient_accumulation
                route_totals += current_route_counts(module)
                scaled_loss.backward()
                local_ce_loss += ce_loss.detach().float().item() / args.gradient_accumulation
                local_aux_loss += aux_loss.detach().float().item() / args.gradient_accumulation

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            lr = learning_rate(step, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            step += 1

            loss_tensor = torch.tensor(
                [
                    local_ce_loss + args.load_balance_weight * local_aux_loss,
                    local_ce_loss,
                    local_aux_loss,
                ],
                device=device,
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor = loss_tensor / world_size
            total_loss_value = loss_tensor[0].item()
            ce_loss_value = loss_tensor[1].item()
            aux_loss_value = loss_tensor[2].item()
            dist.all_reduce(route_totals, op=dist.ReduceOp.SUM)
            step_seconds = time.perf_counter() - started
            recent_step_seconds.append(step_seconds)

            if is_main and (step % args.log_every == 0 or step == 1):
                shares = route_totals / route_totals.sum().clamp_min(1)
                mean_seconds = sum(recent_step_seconds) / len(recent_step_seconds)
                remaining = (
                    mean_seconds * (args.max_steps - step) if args.max_steps > 0 else None
                )
                steps_to_checkpoint = args.save_every - (step % args.save_every)
                if steps_to_checkpoint == args.save_every and step % args.save_every == 0:
                    steps_to_checkpoint = 0
                checkpoint_remaining = mean_seconds * steps_to_checkpoint
                lr_floor_remaining = mean_seconds * max(0, args.decay_steps - step)
                record = {
                    "step": step,
                    "tokens": step * args.batch_size * world_size * args.gradient_accumulation * args.sequence_length,
                    "loss": ce_loss_value,
                    "total_loss": total_loss_value,
                    "ce_loss": ce_loss_value,
                    "load_balance_loss": aux_loss_value,
                    "load_balance_weight": args.load_balance_weight,
                    "perplexity": math.exp(min(ce_loss_value, 20)),
                    "learning_rate": lr,
                    "route_shares": shares.cpu().tolist(),
                    "step_seconds": step_seconds,
                    "mean_step_seconds_50": mean_seconds,
                    "next_checkpoint_remaining_seconds": checkpoint_remaining,
                    "lr_floor_remaining_seconds": lr_floor_remaining,
                }
                if remaining is not None:
                    record["estimated_remaining_seconds"] = remaining
                append_metric(metrics_path, record)
                remaining_text = format_duration(remaining) if remaining is not None else "unbounded"
                print(
                    f"[step {step:07d}] tokens={record['tokens']:,} loss={ce_loss_value:.4f} "
                    f"total={total_loss_value:.4f} "
                    f"ce={ce_loss_value:.4f} lb={aux_loss_value:.4f} "
                    f"lr={lr:.3e} routes={[round(x, 4) for x in record['route_shares']]} "
                    f"step_time={step_seconds:.2f}s avg50={mean_seconds:.2f}s "
                    f"remaining={remaining_text} next_ckpt={format_duration(checkpoint_remaining)} "
                    f"lr_floor={format_duration(lr_floor_remaining)}",
                    flush=True,
                )

            if step % args.save_every == 0:
                save_checkpoint(model, optimizer, stream, step, args, rank, world_size, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
