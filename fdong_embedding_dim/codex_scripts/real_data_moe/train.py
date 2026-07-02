from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from .data import DCLMTokenStream
from .model import ModelConfig, RealDataMoEForCausalLM, parameter_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "proposed"), required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-steps", type=int, default=5_000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--keep-last", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="auto", help="auto, none, or checkpoint path")
    parser.add_argument("--orthogonalize-tail", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--orthogonal-rank", type=int, default=16)
    parser.add_argument("--orthogonal-refresh-steps", type=int, default=50)
    parser.add_argument("--router-window", type=int, default=16)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state() -> Dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_rng(state: Dict[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]


def learning_rate(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.learning_rate * (step + 1) / max(args.warmup_steps, 1)
    progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return args.min_learning_rate + (args.learning_rate - args.min_learning_rate) * cosine


def checkpoint_path(output_dir: str, step: int) -> str:
    return os.path.join(output_dir, f"checkpoint-{step:07d}.pt")


def resolve_resume(output_dir: str, resume: str) -> Optional[str]:
    if resume == "none":
        return None
    if resume != "auto":
        return resume
    latest = os.path.join(output_dir, "latest.pt")
    return latest if os.path.exists(latest) else None


def save_checkpoint(
    path: str,
    model: RealDataMoEForCausalLM,
    optimizer: torch.optim.Optimizer,
    stream: DCLMTokenStream,
    step: int,
    args: argparse.Namespace,
) -> None:
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "stream": stream.state_dict(),
        "rng": rng_state(),
        "model_config": model.config_dict(),
        "train_args": vars(args),
    }
    torch.save(payload, path)
    latest = os.path.join(args.output_dir, "latest.pt")
    tmp = latest + ".tmp"
    if os.path.lexists(tmp):
        os.unlink(tmp)
    os.symlink(os.path.basename(path), tmp)
    os.replace(tmp, latest)

    checkpoints = sorted(Path(args.output_dir).glob("checkpoint-*.pt"))
    for old in checkpoints[: max(0, len(checkpoints) - args.keep_last)]:
        old.unlink()


def build_config(args: argparse.Namespace, vocab_size: int) -> ModelConfig:
    shared = {
        "vocab_size": vocab_size,
        "max_position_embeddings": max(1_024, args.sequence_length),
        "orthogonalize_tail": args.orthogonalize_tail,
        "orthogonal_rank": args.orthogonal_rank,
        "orthogonal_refresh_steps": args.orthogonal_refresh_steps,
        "router_window": args.router_window,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    return ModelConfig.baseline(**shared) if args.variant == "baseline" else ModelConfig.proposed(**shared)


def main() -> None:
    args = parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if args.max_steps < 1 or args.save_every < 1 or args.gradient_accumulation < 1:
        raise ValueError("step and accumulation arguments must be positive")
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    config = build_config(args, len(tokenizer))
    model = RealDataMoEForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    stream = DCLMTokenStream(args.data_dir, tokenizer, seed=args.seed)
    step = 0

    resume_path = resolve_resume(args.output_dir, args.resume)
    if resume_path is not None:
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload["model_config"] != model.config_dict():
            raise RuntimeError("checkpoint model config does not match current config")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        stream.load_state_dict(payload["stream"])
        restore_rng(payload["rng"])
        step = int(payload["step"])
        print(f"[resume] {resume_path} at step {step}")

    counts = parameter_counts(model)
    print("[config]", json.dumps(asdict(config), sort_keys=True))
    print("[parameters]", json.dumps(counts, sort_keys=True))
    print(f"[train] device={device} start={step} target={args.max_steps}")
    model.train()

    while step < args.max_steps:
        model.set_training_step(step)
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        route_totals = torch.zeros(config.num_tail_experts, dtype=torch.long)
        for _ in range(args.gradient_accumulation):
            input_ids, labels = stream.next_batch(
                args.batch_size, args.sequence_length, device
            )
            logits, diagnostics = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
            )
            (loss / args.gradient_accumulation).backward()
            accumulated_loss += loss.detach().item() / args.gradient_accumulation
            for layer_stats in diagnostics.values():
                route_totals += layer_stats["route_counts"].cpu()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        current_lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.step()
        step += 1

        if step % args.log_every == 0 or step == 1:
            shares = route_totals.float() / route_totals.sum().clamp_min(1)
            print(
                f"[step {step:05d}] loss={accumulated_loss:.4f} "
                f"ppl={math.exp(min(accumulated_loss, 20)):.2f} lr={current_lr:.3e} "
                f"routes={[round(value, 4) for value in shares.tolist()]}",
                flush=True,
            )

        if step % args.save_every == 0 or step == args.max_steps:
            path = checkpoint_path(args.output_dir, step)
            save_checkpoint(path, model, optimizer, stream, step, args)
            print(f"[checkpoint] {path}", flush=True)


if __name__ == "__main__":
    main()
