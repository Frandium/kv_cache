from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[4]
FDONG_SCRIPTS_DIR = REPO_ROOT / "fdong" / "scripts"
if str(FDONG_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(FDONG_SCRIPTS_DIR))

from eval_unet_synthetic_retrieval import (  # noqa: E402
    KNOWN_28_LAYER_PATTERNS,
    SyntheticBatch,
    fallback_pattern_for_name,
    generate_variant_a,
    generate_variant_b,
    parse_int_list,
    str2bool,
    validate_token_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MyQwen3 on synthetic retrieval with answer-only loss."
    )
    parser.add_argument("--config_dir", default="/mnt/workspace/Qwen3-0.6B")
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "ymluo/projects/qwen3_unet_synthetic_retrieval/outputs/train"),
    )
    parser.add_argument("--run_name", default="unet-4-variant-b-answer-only")
    parser.add_argument(
        "--model_name",
        choices=sorted(KNOWN_28_LAYER_PATTERNS),
        default="unet-4",
        help="Fallback stride schedule to use when --attention_stride_pattern is not set.",
    )
    parser.add_argument("--variant", choices=["A", "B", "mix"], default="B")
    parser.add_argument("--total_steps", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_bf16", type=str2bool, default=True)
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--attention_stride_pattern", type=parse_int_list, default=None)
    parser.add_argument("--residual_source_pattern", type=parse_int_list, default=None)
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--resume_optimizer", default="")
    parser.add_argument(
        "--train_mode",
        choices=["full_sequence", "full_kv_decode", "anchor_kv_decode"],
        default="anchor_kv_decode",
        help=(
            "full_sequence uses model(source).logits[:, -1]. "
            "full_kv_decode and anchor_kv_decode prefill source[:, :-1], "
            "then decode source[:, -1:] and train only that answer prediction."
        ),
    )
    parser.add_argument("--placeholder_token_id", type=int, default=0)
    parser.add_argument("--content_token_min", type=int, default=1)
    parser.add_argument("--content_token_max", type=int, default=1024)
    parser.add_argument("--query_token_start", type=int, default=1025)
    parser.add_argument("--anchor_token_id", type=int, default=2045)
    args = parser.parse_args()

    if args.total_steps < 1:
        raise ValueError("--total_steps must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be >= 1")
    return args


def resolve_patterns(config: Any, args: argparse.Namespace) -> tuple[list[int], list[int]]:
    fallback_attention = fallback_pattern_for_name(args.model_name, config.num_hidden_layers)
    if fallback_attention is None:
        fallback_attention = [1 for _ in range(config.num_hidden_layers)]
    attention_stride_pattern = args.attention_stride_pattern or fallback_attention
    residual_source_pattern = args.residual_source_pattern or [-1 for _ in range(config.num_hidden_layers)]
    return list(attention_stride_pattern), list(residual_source_pattern)


def prepare_model(args: argparse.Namespace, device: torch.device):
    from transformers import AutoConfig

    from models import MyQwen3ForCausalLM

    config = AutoConfig.from_pretrained(args.config_dir, trust_remote_code=True)
    config._attn_implementation = args.attn_implementation
    config.attention_stride_pattern, config.residual_source_pattern = resolve_patterns(config, args)

    print(f"attention_stride_pattern={config.attention_stride_pattern}", flush=True)
    print(f"residual_source_pattern={config.residual_source_pattern}", flush=True)

    model = MyQwen3ForCausalLM(config).to(device)
    if args.init_checkpoint:
        print(f"loading init checkpoint: {args.init_checkpoint}", flush=True)
        state_dict = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    model.train()
    validate_token_ids(args, model.config.vocab_size)
    return model


def prepare_optimizer(model, args: argparse.Namespace):
    from transformers import get_cosine_schedule_with_warmup

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
    )
    if args.resume_optimizer:
        print(f"loading optimizer state: {args.resume_optimizer}", flush=True)
        payload = torch.load(args.resume_optimizer, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
    return optimizer, scheduler


def make_batch(
    args: argparse.Namespace,
    generator: torch.Generator,
    device: torch.device,
    variant: str | None = None,
) -> tuple[SyntheticBatch, str]:
    actual_variant = variant or args.variant
    if actual_variant == "mix":
        actual_variant = "A" if int(torch.randint(0, 2, (1,), generator=generator).item()) == 0 else "B"
    if actual_variant == "A":
        batch = generate_variant_a(args.batch_size, args, generator)
    elif actual_variant == "B":
        batch = generate_variant_b(args.batch_size, args, generator)
    else:
        raise ValueError(f"Unsupported variant: {actual_variant}")

    batch.source = batch.source.to(device)
    batch.answer = batch.answer.to(device)
    return batch, actual_variant


def autocast_context(args: argparse.Namespace, device: torch.device):
    if args.use_bf16 and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    return nullcontext()


def answer_only_forward(
    model,
    source: torch.Tensor,
    answer: torch.Tensor,
    train_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if train_mode == "full_sequence":
        output = model(input_ids=source, use_cache=False, output_hidden_states=False)
    elif train_mode in {"full_kv_decode", "anchor_kv_decode"}:
        anchor_only = train_mode == "anchor_kv_decode"
        prefill_output = model(
            input_ids=source[:, :-1],
            use_cache=True,
            anchor_only_kv_cache=anchor_only,
            output_hidden_states=False,
        )
        output = model(
            input_ids=source[:, -1:],
            past_key_values=prefill_output.past_key_values,
            use_cache=True,
            anchor_only_kv_cache=anchor_only,
            output_hidden_states=False,
        )
    else:
        raise ValueError(f"Unsupported train_mode: {train_mode}")

    answer_logits = output.logits[:, -1, :]
    loss = F.cross_entropy(answer_logits.float(), answer)
    return loss, answer_logits


@torch.no_grad()
def evaluate_answer_only(
    model,
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    variant_counts: dict[str, int] = {"A": 0, "B": 0}

    for _ in range(args.eval_batches):
        batch, actual_variant = make_batch(args, generator, device)
        with autocast_context(args, device):
            loss, logits = answer_only_forward(model, batch.source, batch.answer, args.train_mode)
        total_loss += float(loss.detach().cpu()) * batch.answer.numel()
        total_correct += int(logits.argmax(dim=-1).eq(batch.answer).sum().detach().cpu())
        total_count += int(batch.answer.numel())
        variant_counts[actual_variant] = variant_counts.get(actual_variant, 0) + int(batch.answer.numel())

    if was_training:
        model.train()

    loss = total_loss / max(total_count, 1)
    return {
        "loss": loss,
        "accuracy": total_correct / max(total_count, 1),
        "ppl": math.exp(min(loss, 80.0)),
        "count": total_count,
        "variant_A_count": variant_counts.get("A", 0),
        "variant_B_count": variant_counts.get("B", 0),
    }


def write_runtime_config(ckpt_dir: Path, model, args: argparse.Namespace) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "attention_stride_pattern": list(model.model.attention_stride_pattern),
        "residual_source_pattern": list(model.model.residual_source_pattern),
        "training_objective": "answer_only_last_logit_cross_entropy",
        "train_mode": args.train_mode,
        "synthetic_variant": args.variant,
    }
    (ckpt_dir / "runtime_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint(
    ckpt_dir: Path,
    step: int,
    model,
    optimizer,
    scheduler,
    args: argparse.Namespace,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"{step}.pth")
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
        },
        ckpt_dir / f"{step}.optim.pth",
    )
    write_runtime_config(ckpt_dir, model, args)
    print(f"saved checkpoint: {ckpt_dir / f'{step}.pth'}", flush=True)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)

    run_dir = Path(args.output_dir) / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Synthetic answer-only training", flush=True)
    print(f"run_dir={run_dir}", flush=True)
    print(f"device={device}", flush=True)
    print(f"variant={args.variant}", flush=True)
    print(f"train_mode={args.train_mode}", flush=True)
    print(f"batch_size={args.batch_size}", flush=True)

    model = prepare_model(args, device)
    optimizer, scheduler = prepare_optimizer(model, args)
    train_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    eval_generator = torch.Generator(device="cpu").manual_seed(args.seed + 99_999)

    optimizer.zero_grad(set_to_none=True)
    rolling_loss = 0.0
    rolling_correct = 0
    rolling_count = 0
    rolling_variant_counts: dict[str, int] = {"A": 0, "B": 0}
    started_at = time.monotonic()

    for step in range(1, args.total_steps + 1):
        step_loss = 0.0
        step_correct = 0
        step_count = 0
        variant_seen: dict[str, int] = {}

        for _ in range(args.gradient_accumulation_steps):
            batch, actual_variant = make_batch(args, train_generator, device)
            with autocast_context(args, device):
                loss, logits = answer_only_forward(model, batch.source, batch.answer, args.train_mode)
            (loss / args.gradient_accumulation_steps).backward()

            count = int(batch.answer.numel())
            step_loss += float(loss.detach().cpu()) * count
            step_correct += int(logits.argmax(dim=-1).eq(batch.answer).sum().detach().cpu())
            step_count += count
            variant_seen[actual_variant] = variant_seen.get(actual_variant, 0) + count

        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        rolling_loss += step_loss
        rolling_correct += step_correct
        rolling_count += step_count
        for variant_name, count in variant_seen.items():
            rolling_variant_counts[variant_name] = rolling_variant_counts.get(variant_name, 0) + count

        if step % args.log_interval == 0 or step == 1:
            elapsed = max(time.monotonic() - started_at, 1e-6)
            train_loss = rolling_loss / max(rolling_count, 1)
            train_acc = rolling_correct / max(rolling_count, 1)
            row = {
                "type": "train",
                "step": step,
                "loss": train_loss,
                "accuracy": train_acc,
                "count": rolling_count,
                "lr": scheduler.get_last_lr()[0],
                "answers_per_second": rolling_count / elapsed,
                "variant_A_count": rolling_variant_counts.get("A", 0),
                "variant_B_count": rolling_variant_counts.get("B", 0),
            }
            print(
                f"step {step}: train_answer_loss={train_loss:.6f} "
                f"train_answer_acc={train_acc:.4f} lr={row['lr']:.3e}",
                flush=True,
            )
            append_jsonl(metrics_path, row)
            rolling_loss = 0.0
            rolling_correct = 0
            rolling_count = 0
            rolling_variant_counts = {"A": 0, "B": 0}
            started_at = time.monotonic()

        if args.eval_interval > 0 and step % args.eval_interval == 0:
            metrics = evaluate_answer_only(model, args, device, eval_generator)
            row = {"type": "eval", "step": step, **metrics}
            print(
                f"step {step}: eval_answer_loss={metrics['loss']:.6f} "
                f"eval_answer_acc={metrics['accuracy']:.4f}",
                flush=True,
            )
            append_jsonl(metrics_path, row)

        if args.save_interval > 0 and step % args.save_interval == 0:
            save_checkpoint(ckpt_dir, step, model, optimizer, scheduler, args)

    save_checkpoint(ckpt_dir, args.total_steps, model, optimizer, scheduler, args)
    print(f"finished. metrics={metrics_path}", flush=True)


if __name__ == "__main__":
    main()
