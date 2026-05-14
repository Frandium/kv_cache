from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[4]
FDONG_SCRIPTS_DIR = REPO_ROOT / "fdong" / "scripts"
if str(FDONG_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(FDONG_SCRIPTS_DIR))


DEFAULT_CHECKPOINTS = {
    "baseline": "/mnt/workspace/df-unet-transformer/fdong/checkpoints/baseline/145000.pth",
    "unet-4": "/mnt/workspace/df-unet-transformer/fdong/checkpoints/unet-4/103000.pth",
    "unet-4-8-4": "/mnt/workspace/df-unet-transformer/fdong/checkpoints/unet-4-8-4/70000.pth",
    "unet-4-8-16-8-4": "/mnt/workspace/df-unet-transformer/fdong/checkpoints/unet-4-8-16-8-4/24000.pth",
}

KNOWN_28_LAYER_PATTERNS = {
    "baseline": [1] * 28,
    "unet-4": [1] * 9 + [4] * 9 + [1] * 10,
    "unet-4-8-4": [1] * 6 + [4] * 6 + [8] * 4 + [4] * 6 + [1] * 6,
    "unet-4-8-16-8-4": [1] * 4
    + [4] * 4
    + [8] * 4
    + [16] * 4
    + [8] * 4
    + [4] * 4
    + [1] * 4,
}


@dataclass
class CheckpointSpec:
    name: str
    path: str


@dataclass
class SyntheticBatch:
    source: torch.Tensor
    answer: torch.Tensor
    answer_position: torch.Tensor
    block_id: torch.Tensor
    offset: torch.Tensor
    query_token: torch.Tensor


@dataclass
class ModeMetrics:
    loss: float | None
    accuracy: float | None
    count: int
    cache_total_tokens_mean: float | None = None
    cache_per_layer_tokens_mean: dict[str, float] | None = None


@dataclass
class LogitComparison:
    full_vs_anchor_top1_match: float | None
    mean_abs_diff: float | None
    max_abs_diff: float | None


@dataclass
class EvalResult:
    model_name: str
    checkpoint_path: str
    variant: str
    num_samples: int
    full_sequence: ModeMetrics | None
    full_kv_decode: ModeMetrics
    anchor_kv_decode: ModeMetrics
    anchor_vs_full_cache_ratio: float | None
    logit_comparison: LogitComparison
    anchor_accuracy_by_offset: dict[str, dict[str, float | int | None]]
    anchor_accuracy_by_distance_bucket: dict[str, dict[str, float | int | None]]
    attention_stride_pattern: list[int]
    residual_source_pattern: list[int]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", default="/mnt/workspace/Qwen3-0.6B")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "ymluo/projects/qwen3_unet_synthetic_retrieval/outputs/synthetic_eval"))
    parser.add_argument("--checkpoint", action="append", default=[], help="Repeatable NAME=PATH override.")
    parser.add_argument("--model_names", default="", help="Comma-separated subset of checkpoint names to run.")
    parser.add_argument("--variants", default="A,B", help="Comma-separated list from A,B.")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_bf16", type=str2bool, default=True)
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--attention_stride_pattern", type=parse_int_list, default=None)
    parser.add_argument("--residual_source_pattern", type=parse_int_list, default=None)
    parser.add_argument("--skip_full_sequence", type=str2bool, default=False)
    parser.add_argument("--skip_missing_checkpoints", type=str2bool, default=False)
    parser.add_argument("--sample_dump_path", default="")
    parser.add_argument("--placeholder_token_id", type=int, default=0)
    parser.add_argument("--content_token_min", type=int, default=1)
    parser.add_argument("--content_token_max", type=int, default=1024)
    parser.add_argument("--query_token_start", type=int, default=1025)
    parser.add_argument("--anchor_token_id", type=int, default=2045)
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num_samples must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    variants = [item.strip().upper() for item in args.variants.split(",") if item.strip()]
    invalid = [variant for variant in variants if variant not in {"A", "B"}]
    if invalid:
        raise ValueError(f"Unsupported variants: {invalid}")
    args.variants = variants
    args.model_names = [item.strip() for item in args.model_names.split(",") if item.strip()]
    return args


def parse_checkpoint_specs(args: argparse.Namespace) -> list[CheckpointSpec]:
    if args.checkpoint:
        specs = []
        for raw in args.checkpoint:
            if "=" not in raw:
                raise ValueError(f"--checkpoint must be NAME=PATH, got {raw!r}")
            name, path = raw.split("=", 1)
            specs.append(CheckpointSpec(name=name.strip(), path=path.strip()))
    else:
        specs = [CheckpointSpec(name=name, path=path) for name, path in DEFAULT_CHECKPOINTS.items()]

    if args.model_names:
        wanted = set(args.model_names)
        specs = [spec for spec in specs if spec.name in wanted]
        missing = wanted - {spec.name for spec in specs}
        if missing:
            raise ValueError(f"--model_names requested unknown checkpoint names: {sorted(missing)}")
    return specs


def load_runtime_config(ckpt_path: str) -> dict[str, Any]:
    runtime_path = Path(ckpt_path).parent / "runtime_config.json"
    if runtime_path.exists():
        return json.loads(runtime_path.read_text(encoding="utf-8"))
    return {}


def fallback_pattern_for_name(name: str, num_layers: int) -> list[int] | None:
    pattern = KNOWN_28_LAYER_PATTERNS.get(name)
    if pattern is None:
        return None
    if len(pattern) != num_layers:
        raise ValueError(
            f"Known fallback pattern for {name!r} has length {len(pattern)}, "
            f"but config has {num_layers} layers."
        )
    return list(pattern)


def resolve_pattern(
    cli_pattern: list[int] | None,
    runtime_config: dict[str, Any],
    key: str,
    fallback: list[int],
) -> list[int]:
    runtime_pattern = runtime_config.get(key)
    if cli_pattern is not None and runtime_pattern is not None and cli_pattern != runtime_pattern:
        raise ValueError(
            f"{key} mismatch between command line and runtime_config.json: "
            f"{cli_pattern} vs {runtime_pattern}"
        )
    return list(cli_pattern or runtime_pattern or fallback)


def load_model(spec: CheckpointSpec, args: argparse.Namespace, device: torch.device):
    ckpt_path = Path(spec.path)
    if not ckpt_path.exists():
        if args.skip_missing_checkpoints:
            print(f"skip missing checkpoint: {ckpt_path}", flush=True)
            return None
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    from transformers import AutoConfig

    from models import MyQwen3ForCausalLM

    config = AutoConfig.from_pretrained(args.config_dir, trust_remote_code=True)
    config._attn_implementation = args.attn_implementation
    runtime_config = load_runtime_config(spec.path)

    fallback_attention = fallback_pattern_for_name(spec.name, config.num_hidden_layers)
    if fallback_attention is None:
        fallback_attention = [1 for _ in range(config.num_hidden_layers)]
    fallback_residual = [-1 for _ in range(config.num_hidden_layers)]

    config.attention_stride_pattern = resolve_pattern(
        args.attention_stride_pattern,
        runtime_config,
        "attention_stride_pattern",
        fallback_attention,
    )
    config.residual_source_pattern = resolve_pattern(
        args.residual_source_pattern,
        runtime_config,
        "residual_source_pattern",
        fallback_residual,
    )

    print(f"loading {spec.name}: {ckpt_path}", flush=True)
    print(f"  attention_stride_pattern={config.attention_stride_pattern}", flush=True)
    model = MyQwen3ForCausalLM(config).to(device)
    state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def validate_token_ids(args: argparse.Namespace, vocab_size: int) -> None:
    max_id = max(args.content_token_max, args.query_token_start + 1019, args.anchor_token_id)
    min_id = min(args.placeholder_token_id, args.content_token_min)
    if min_id < 0 or max_id >= vocab_size:
        raise ValueError(
            f"Synthetic token ids must fit vocab size {vocab_size}; got range [{min_id}, {max_id}]."
        )


def generate_variant_a(num_samples: int, args: argparse.Namespace, generator: torch.Generator) -> SyntheticBatch:
    sources = []
    answers = []
    answer_positions = []
    block_ids = []
    offsets = []
    query_tokens = []

    patterns = torch.arange(1, 1025, dtype=torch.long).view(256, 4)
    for _ in range(num_samples):
        selected = torch.randperm(256, generator=generator)[:255]
        selected = selected[torch.randperm(255, generator=generator)]
        prefix = patterns[selected].reshape(-1)
        query_pos = int(torch.randint(0, 1020, (1,), generator=generator).item())
        query_token = args.query_token_start + query_pos
        sequence = torch.empty(1025, dtype=torch.long)
        sequence[:1020] = prefix
        sequence[1020:1023] = args.placeholder_token_id
        sequence[1023] = query_token
        sequence[1024] = sequence[query_pos]

        sources.append(sequence[:1024])
        answers.append(sequence[1024])
        answer_positions.append(query_pos)
        block_ids.append(query_pos // 4)
        offsets.append(query_pos % 4)
        query_tokens.append(query_token)

    return SyntheticBatch(
        source=torch.stack(sources),
        answer=torch.tensor(answers, dtype=torch.long),
        answer_position=torch.tensor(answer_positions, dtype=torch.long),
        block_id=torch.tensor(block_ids, dtype=torch.long),
        offset=torch.tensor(offsets, dtype=torch.long),
        query_token=torch.tensor(query_tokens, dtype=torch.long),
    )


def generate_variant_b(num_samples: int, args: argparse.Namespace, generator: torch.Generator) -> SyntheticBatch:
    sources = []
    answers = []
    answer_positions = []
    block_ids = []
    offsets = []
    query_tokens = []

    content_low = args.content_token_min
    content_high_exclusive = args.content_token_max + 1
    for _ in range(num_samples):
        content = torch.randint(
            low=content_low,
            high=content_high_exclusive,
            size=(255, 3),
            generator=generator,
            dtype=torch.long,
        )
        blocks = torch.empty((255, 4), dtype=torch.long)
        blocks[:, :3] = content
        blocks[:, 3] = args.anchor_token_id

        content_pos = int(torch.randint(0, 765, (1,), generator=generator).item())
        block_id = content_pos // 3
        offset = content_pos % 3
        source_pos = block_id * 4 + offset
        query_token = args.query_token_start + content_pos

        sequence = torch.empty(1025, dtype=torch.long)
        sequence[:1020] = blocks.reshape(-1)
        sequence[1020:1023] = args.placeholder_token_id
        sequence[1023] = query_token
        sequence[1024] = content[block_id, offset]

        sources.append(sequence[:1024])
        answers.append(sequence[1024])
        answer_positions.append(source_pos)
        block_ids.append(block_id)
        offsets.append(offset)
        query_tokens.append(query_token)

    return SyntheticBatch(
        source=torch.stack(sources),
        answer=torch.tensor(answers, dtype=torch.long),
        answer_position=torch.tensor(answer_positions, dtype=torch.long),
        block_id=torch.tensor(block_ids, dtype=torch.long),
        offset=torch.tensor(offsets, dtype=torch.long),
        query_token=torch.tensor(query_tokens, dtype=torch.long),
    )


def generate_dataset(variant: str, args: argparse.Namespace) -> SyntheticBatch:
    seed = args.seed + (0 if variant == "A" else 10_000)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if variant == "A":
        return generate_variant_a(args.num_samples, args, generator)
    if variant == "B":
        return generate_variant_b(args.num_samples, args, generator)
    raise ValueError(f"Unsupported variant: {variant}")


def cache_lengths(past_key_values) -> dict[int, int]:
    if past_key_values is None:
        return {}
    if hasattr(past_key_values, "get_cache_lengths"):
        return past_key_values.get_cache_lengths()
    if hasattr(past_key_values, "key_cache"):
        return {
            layer_idx: 0 if key_states is None else int(key_states.shape[-2])
            for layer_idx, key_states in enumerate(past_key_values.key_cache)
        }
    return {}


def total_cache_tokens(lengths: dict[int, int]) -> int:
    return sum(int(value) for value in lengths.values())


class MetricAccumulator:
    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.correct = 0
        self.count = 0
        self.cache_total_tokens_sum = 0.0
        self.cache_observations = 0
        self.cache_per_layer_sum: dict[int, float] = {}

    def update_logits(self, logits: torch.Tensor, answer: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        losses = F.cross_entropy(logits.float(), answer, reduction="none")
        pred = logits.argmax(dim=-1)
        correct = pred.eq(answer)
        self.loss_sum += float(losses.sum().detach().cpu())
        self.correct += int(correct.sum().detach().cpu())
        self.count += int(answer.numel())
        return losses.detach().cpu(), correct.detach().cpu()

    def update_cache(self, lengths: dict[int, int]) -> None:
        if not lengths:
            return
        self.cache_total_tokens_sum += total_cache_tokens(lengths)
        self.cache_observations += 1
        for layer_idx, length in lengths.items():
            self.cache_per_layer_sum[int(layer_idx)] = self.cache_per_layer_sum.get(int(layer_idx), 0.0) + int(length)

    def summarize(self) -> ModeMetrics:
        cache_total = None
        cache_per_layer = None
        if self.cache_observations:
            cache_total = self.cache_total_tokens_sum / self.cache_observations
            cache_per_layer = {
                str(layer_idx): value / self.cache_observations
                for layer_idx, value in sorted(self.cache_per_layer_sum.items())
            }
        return ModeMetrics(
            loss=None if self.count == 0 else self.loss_sum / self.count,
            accuracy=None if self.count == 0 else self.correct / self.count,
            count=self.count,
            cache_total_tokens_mean=cache_total,
            cache_per_layer_tokens_mean=cache_per_layer,
        )


class GroupAccumulator:
    def __init__(self) -> None:
        self.stats: dict[str, dict[str, float | int]] = {}

    def update(self, group_keys: torch.Tensor, losses: torch.Tensor, correct: torch.Tensor) -> None:
        keys = group_keys.detach().cpu().tolist()
        loss_values = losses.detach().cpu().tolist()
        correct_values = correct.detach().cpu().tolist()
        for key, loss, is_correct in zip(keys, loss_values, correct_values):
            stat = self.stats.setdefault(str(key), {"loss_sum": 0.0, "correct": 0, "count": 0})
            stat["loss_sum"] = float(stat["loss_sum"]) + float(loss)
            stat["correct"] = int(stat["correct"]) + int(bool(is_correct))
            stat["count"] = int(stat["count"]) + 1

    def summarize(self) -> dict[str, dict[str, float | int | None]]:
        summary = {}
        for key, stat in sorted(self.stats.items(), key=lambda item: item[0]):
            count = int(stat["count"])
            summary[key] = {
                "loss": None if count == 0 else float(stat["loss_sum"]) / count,
                "accuracy": None if count == 0 else int(stat["correct"]) / count,
                "count": count,
            }
        return summary


def distance_buckets(answer_position: torch.Tensor) -> torch.Tensor:
    distance = 1023 - answer_position
    buckets = torch.empty_like(distance)
    buckets[distance < 128] = 0
    buckets[(distance >= 128) & (distance < 256)] = 1
    buckets[(distance >= 256) & (distance < 512)] = 2
    buckets[distance >= 512] = 3
    return buckets


def bucket_labels(summary: dict[str, dict[str, float | int | None]]) -> dict[str, dict[str, float | int | None]]:
    labels = {
        "0": "distance_0000_0127",
        "1": "distance_0128_0255",
        "2": "distance_0256_0511",
        "3": "distance_0512_1023",
    }
    return {labels.get(key, key): value for key, value in summary.items()}


@torch.no_grad()
def full_sequence_answer_logits(model, source: torch.Tensor) -> torch.Tensor:
    output = model(input_ids=source, use_cache=False, output_hidden_states=False)
    return output.logits[:, -1, :]


@torch.no_grad()
def decode_answer_logits(
    model,
    source: torch.Tensor,
    anchor_only_kv_cache: bool,
) -> tuple[torch.Tensor, dict[int, int]]:
    prefill = source[:, :-1]
    query = source[:, -1:]
    prefill_output = model(
        input_ids=prefill,
        use_cache=True,
        anchor_only_kv_cache=anchor_only_kv_cache,
        output_hidden_states=False,
    )
    decode_output = model(
        input_ids=query,
        past_key_values=prefill_output.past_key_values,
        use_cache=True,
        anchor_only_kv_cache=anchor_only_kv_cache,
        output_hidden_states=False,
    )
    return decode_output.logits[:, -1, :], cache_lengths(decode_output.past_key_values)


def iter_batch_slices(num_samples: int, batch_size: int):
    for start in range(0, num_samples, batch_size):
        yield slice(start, min(start + batch_size, num_samples))


def maybe_autocast(args: argparse.Namespace, device: torch.device):
    enabled = args.use_bf16 and device.type == "cuda"
    return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)


def eval_model_on_variant(
    model,
    spec: CheckpointSpec,
    variant: str,
    dataset: SyntheticBatch,
    args: argparse.Namespace,
    device: torch.device,
) -> EvalResult:
    full_sequence_acc = MetricAccumulator() if not args.skip_full_sequence else None
    full_kv_acc = MetricAccumulator()
    anchor_kv_acc = MetricAccumulator()
    offset_acc = GroupAccumulator()
    distance_acc = GroupAccumulator()

    top1_match_sum = 0
    top1_match_count = 0
    abs_diff_sum = 0.0
    abs_diff_count = 0
    max_abs_diff = 0.0

    for batch_idx, batch_slice in enumerate(iter_batch_slices(dataset.source.shape[0], args.batch_size), start=1):
        source = dataset.source[batch_slice].to(device)
        answer = dataset.answer[batch_slice].to(device)
        offsets = dataset.offset[batch_slice]
        distance_bucket_ids = distance_buckets(dataset.answer_position[batch_slice])

        with maybe_autocast(args, device):
            if full_sequence_acc is not None:
                full_logits = full_sequence_answer_logits(model, source)
                full_sequence_acc.update_logits(full_logits, answer)

            full_kv_logits, full_lengths = decode_answer_logits(model, source, anchor_only_kv_cache=False)
            anchor_kv_logits, anchor_lengths = decode_answer_logits(model, source, anchor_only_kv_cache=True)

        full_kv_acc.update_logits(full_kv_logits, answer)
        anchor_losses, anchor_correct = anchor_kv_acc.update_logits(anchor_kv_logits, answer)
        full_kv_acc.update_cache(full_lengths)
        anchor_kv_acc.update_cache(anchor_lengths)

        offset_acc.update(offsets, anchor_losses, anchor_correct)
        distance_acc.update(distance_bucket_ids, anchor_losses, anchor_correct)

        diff = (full_kv_logits.float() - anchor_kv_logits.float()).abs()
        abs_diff_sum += float(diff.sum().detach().cpu())
        abs_diff_count += diff.numel()
        max_abs_diff = max(max_abs_diff, float(diff.max().detach().cpu()))
        top1_match_sum += int(
            full_kv_logits.argmax(dim=-1).eq(anchor_kv_logits.argmax(dim=-1)).sum().detach().cpu()
        )
        top1_match_count += int(answer.numel())

        print(
            f"{spec.name} variant {variant} batch {batch_idx}: "
            f"anchor_acc={anchor_kv_acc.correct}/{anchor_kv_acc.count}",
            flush=True,
        )

    full_summary = full_sequence_acc.summarize() if full_sequence_acc is not None else None
    full_kv_summary = full_kv_acc.summarize()
    anchor_kv_summary = anchor_kv_acc.summarize()
    cache_ratio = None
    if full_kv_summary.cache_total_tokens_mean and anchor_kv_summary.cache_total_tokens_mean:
        cache_ratio = anchor_kv_summary.cache_total_tokens_mean / full_kv_summary.cache_total_tokens_mean

    logit_comparison = LogitComparison(
        full_vs_anchor_top1_match=None
        if top1_match_count == 0
        else top1_match_sum / top1_match_count,
        mean_abs_diff=None if abs_diff_count == 0 else abs_diff_sum / abs_diff_count,
        max_abs_diff=None if abs_diff_count == 0 else max_abs_diff,
    )

    return EvalResult(
        model_name=spec.name,
        checkpoint_path=spec.path,
        variant=variant,
        num_samples=dataset.source.shape[0],
        full_sequence=full_summary,
        full_kv_decode=full_kv_summary,
        anchor_kv_decode=anchor_kv_summary,
        anchor_vs_full_cache_ratio=cache_ratio,
        logit_comparison=logit_comparison,
        anchor_accuracy_by_offset=offset_acc.summarize(),
        anchor_accuracy_by_distance_bucket=bucket_labels(distance_acc.summarize()),
        attention_stride_pattern=list(model.model.attention_stride_pattern),
        residual_source_pattern=list(model.model.residual_source_pattern),
    )


def write_sample_dump(path: str, datasets: dict[str, SyntheticBatch], limit: int = 8) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant, batch in datasets.items():
        for idx in range(min(limit, batch.source.shape[0])):
            rows.append(
                {
                    "variant": variant,
                    "index": idx,
                    "source": batch.source[idx].tolist(),
                    "answer": int(batch.answer[idx].item()),
                    "answer_position": int(batch.answer_position[idx].item()),
                    "block_id": int(batch.block_id[idx].item()),
                    "offset": int(batch.offset[idx].item()),
                    "query_token": int(batch.query_token[idx].item()),
                }
            )
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def write_outputs(output_dir: Path, results: list[EvalResult], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": {
            key: value
            for key, value in vars(args).items()
            if key not in {"attention_stride_pattern", "residual_source_pattern"}
        },
        "results": [asdict(result) for result in results],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rows = []
    for result in results:
        for mode_name, metrics in [
            ("full_sequence", result.full_sequence),
            ("full_kv_decode", result.full_kv_decode),
            ("anchor_kv_decode", result.anchor_kv_decode),
        ]:
            if metrics is None:
                continue
            rows.append(
                {
                    "model_name": result.model_name,
                    "checkpoint_path": result.checkpoint_path,
                    "variant": result.variant,
                    "mode": mode_name,
                    "loss": metrics.loss,
                    "accuracy": metrics.accuracy,
                    "count": metrics.count,
                    "cache_total_tokens_mean": metrics.cache_total_tokens_mean,
                    "anchor_vs_full_cache_ratio": result.anchor_vs_full_cache_ratio,
                    "full_vs_anchor_top1_match": result.logit_comparison.full_vs_anchor_top1_match,
                    "mean_abs_diff": result.logit_comparison.mean_abs_diff,
                    "max_abs_diff": result.logit_comparison.max_abs_diff,
                }
            )

    if rows:
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    specs = parse_checkpoint_specs(args)

    print("Synthetic U-Net retrieval evaluation", flush=True)
    print(f"repo_root={REPO_ROOT}", flush=True)
    print(f"fdong_scripts_dir={FDONG_SCRIPTS_DIR}", flush=True)
    print(f"device={device}", flush=True)
    print(f"variants={args.variants}", flush=True)
    print(f"num_samples={args.num_samples} batch_size={args.batch_size}", flush=True)

    datasets = {variant: generate_dataset(variant, args) for variant in args.variants}
    write_sample_dump(args.sample_dump_path, datasets)

    results: list[EvalResult] = []
    for spec in specs:
        model = load_model(spec, args, device)
        if model is None:
            continue
        validate_token_ids(args, model.config.vocab_size)

        for variant in args.variants:
            result = eval_model_on_variant(model, spec, variant, datasets[variant], args, device)
            results.append(result)
            anchor = result.anchor_kv_decode
            print(
                f"{spec.name} variant {variant}: "
                f"anchor_loss={anchor.loss} anchor_acc={anchor.accuracy} "
                f"cache_ratio={result.anchor_vs_full_cache_ratio}",
                flush=True,
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_outputs(Path(args.output_dir), results, args)
    metrics_path = Path(args.output_dir) / "metrics.json"
    csv_path = Path(args.output_dir) / "metrics.csv"
    print(f"saved metrics to {metrics_path}", flush=True)
    if csv_path.exists():
        print(f"saved csv to {csv_path}", flush=True)
    else:
        print("no evaluated checkpoint rows; metrics.csv was not written", flush=True)


if __name__ == "__main__":
    main()
