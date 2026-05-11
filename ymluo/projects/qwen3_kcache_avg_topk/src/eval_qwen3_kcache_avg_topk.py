from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen3_kcache_avg_topk import KCacheAvgTopKConfig, patch_qwen3_kcache_avg_topk


@dataclass
class EvalMetrics:
    name: str
    loss: float
    ppl: float
    mean_probability: float
    geometric_mean_probability: float
    tokens: int
    sequences: int
    mean_keep_block_ratio: float | None = None
    mean_keep_token_ratio: float | None = None
    mean_attention_energy: float | None = None


@dataclass
class HeadEnergyMetric:
    name: str
    layer: int
    head: int
    mean_energy: float
    samples: int


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument(
        "--dataset_path",
        default="/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10",
    )
    parser.add_argument("--output_dir", default="projects/qwen3_kcache_avg_topk/outputs/eval")
    parser.add_argument("--max_files", type=int, default=128)
    parser.add_argument("--max_sequences", type=int, default=128)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--min_tokens", type=int, default=32)
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_baseline", type=str2bool, default=True)
    parser.add_argument("--eval_sparse", type=str2bool, default=True)
    parser.add_argument("--block_size", type=int, default=10)
    parser.add_argument("--topk_ratio", type=float, default=0.30)
    parser.add_argument("--first_sparse_layer", type=int, default=3)
    parser.add_argument("--last_sparse_layer", type=int, default=27)
    parser.add_argument("--min_blocks_to_keep", type=int, default=1)
    return parser.parse_args()


def iter_text_files(dataset_path: str, max_files: int) -> Iterable[Path]:
    root = Path(dataset_path)
    if root.is_file():
        yield root
        return
    files = sorted(root.glob("*.txt"))
    if max_files > 0:
        files = files[:max_files]
    for path in files:
        yield path


def iter_token_sequences(
    tokenizer,
    dataset_path: str,
    max_files: int,
    max_sequences: int,
    seq_length: int,
    stride: int,
    min_tokens: int,
) -> Iterable[torch.Tensor]:
    produced = 0
    target_len = seq_length + 1
    for path in iter_text_files(dataset_path, max_files):
        text = path.read_text(encoding="utf-8", errors="ignore")
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) < min_tokens:
            continue
        if tokenizer.eos_token_id is not None:
            token_ids.append(tokenizer.eos_token_id)

        if len(token_ids) <= target_len:
            yield torch.tensor(token_ids, dtype=torch.long)
            produced += 1
        else:
            for start in range(0, len(token_ids) - min_tokens + 1, stride):
                chunk = token_ids[start : start + target_len]
                if len(chunk) < min_tokens:
                    continue
                yield torch.tensor(chunk, dtype=torch.long)
                produced += 1
                if 0 < max_sequences <= produced:
                    return
        if 0 < max_sequences <= produced:
            return


def _iter_sparse_modules(model) -> list[torch.nn.Module]:
    return [
        module
        for module in model.modules()
        if hasattr(module, "kcache_avg_topk_config")
    ]


def score_sequences_decode(
    model,
    sequences: list[torch.Tensor],
    device: torch.device,
    name: str,
) -> tuple[EvalMetrics, list[HeadEnergyMetric]]:
    total_nll = 0.0
    total_prob = 0.0
    total_tokens = 0
    keep_ratio_sum = 0.0
    keep_token_ratio_sum = 0.0
    keep_ratio_count = 0
    energy_sums: dict[int, torch.Tensor] = {}
    energy_counts: dict[int, int] = {}
    total_energy_sum = 0.0
    total_energy_count = 0
    sparse_modules = _iter_sparse_modules(model)

    model.eval()
    with torch.inference_mode():
        for seq in sequences:
            if seq.numel() < 2:
                continue
            input_ids = seq.to(device).view(1, -1)
            outputs = model(input_ids=input_ids[:, :1], use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            for pos in range(1, input_ids.shape[1]):
                target = input_ids[:, pos]
                log_probs = F.log_softmax(logits.float(), dim=-1)
                token_log_prob = log_probs.gather(-1, target.view(1, 1)).squeeze()
                total_nll += float(-token_log_prob)
                total_prob += float(token_log_prob.exp())
                total_tokens += 1

                if sparse_modules:
                    for module_idx, module in enumerate(sparse_modules):
                        num_blocks = getattr(module, "_kcache_avg_topk_last_num_blocks", 0)
                        keep_blocks = getattr(module, "_kcache_avg_topk_last_keep_blocks", 0)
                        kv_len = getattr(module, "_kcache_avg_topk_last_kv_len", 0)
                        if num_blocks > 0 and kv_len > 0:
                            keep_ratio_sum += keep_blocks / num_blocks
                            kept_tokens = min(
                                keep_blocks * module.kcache_avg_topk_config.block_size,
                                kv_len,
                            )
                            keep_token_ratio_sum += kept_tokens / kv_len
                            keep_ratio_count += 1

                            energy = getattr(
                                module,
                                "_kcache_avg_topk_last_attention_energy",
                                None,
                            )
                            if energy is not None:
                                energy_cpu = energy.float().detach().cpu()
                                if energy_cpu.ndim == 3:
                                    head_sum = energy_cpu.sum(dim=(0, 2))
                                    sample_count = int(energy_cpu.shape[0] * energy_cpu.shape[2])
                                    layer_idx = getattr(module, "layer_idx", None)
                                    if layer_idx is None:
                                        layer_idx = module_idx
                                    layer_idx = int(layer_idx)
                                    if layer_idx not in energy_sums:
                                        energy_sums[layer_idx] = torch.zeros_like(head_sum)
                                        energy_counts[layer_idx] = 0
                                    energy_sums[layer_idx] += head_sum
                                    energy_counts[layer_idx] += sample_count
                                    total_energy_sum += float(head_sum.sum())
                                    total_energy_count += sample_count * head_sum.numel()

                if pos + 1 < input_ids.shape[1]:
                    outputs = model(
                        input_ids=input_ids[:, pos : pos + 1],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]

    if total_tokens == 0:
        raise ValueError("No tokens were scored. Check dataset_path/min_tokens/max_sequences.")

    loss = total_nll / total_tokens
    head_energy_rows: list[HeadEnergyMetric] = []
    for layer_idx in sorted(energy_sums):
        mean_by_head = energy_sums[layer_idx] / energy_counts[layer_idx]
        for head_idx, mean_energy in enumerate(mean_by_head.tolist()):
            head_energy_rows.append(
                HeadEnergyMetric(
                    name=name,
                    layer=layer_idx,
                    head=head_idx,
                    mean_energy=float(mean_energy),
                    samples=energy_counts[layer_idx],
                )
            )

    metrics = EvalMetrics(
        name=name,
        loss=loss,
        ppl=math.exp(min(loss, 80.0)),
        mean_probability=total_prob / total_tokens,
        geometric_mean_probability=math.exp(-loss),
        tokens=total_tokens,
        sequences=len(sequences),
        mean_keep_block_ratio=(keep_ratio_sum / keep_ratio_count if keep_ratio_count else None),
        mean_keep_token_ratio=(keep_token_ratio_sum / keep_ratio_count if keep_ratio_count else None),
        mean_attention_energy=(
            total_energy_sum / total_energy_count if total_energy_count else None
        ),
    )
    return metrics, head_energy_rows


def write_outputs(
    output_dir: Path,
    rows: list[EvalMetrics],
    head_energy_rows: list[HeadEnergyMetric],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "metrics": [asdict(row) for row in rows],
        "head_energy": [asdict(row) for row in head_energy_rows],
    }
    if len(rows) == 2:
        base = rows[0]
        other = rows[1]
        payload["delta"] = {
            "loss": other.loss - base.loss,
            "ppl": other.ppl - base.ppl,
            "mean_probability": other.mean_probability - base.mean_probability,
            "geometric_mean_probability": (
                other.geometric_mean_probability - base.geometric_mean_probability
            ),
        }

    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    if head_energy_rows:
        head_energy_payload = [asdict(row) for row in head_energy_rows]
        (output_dir / "head_energy_by_layer_head.json").write_text(
            json.dumps(head_energy_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with (output_dir / "head_energy_by_layer_head.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(asdict(head_energy_rows[0]).keys()),
            )
            writer.writeheader()
            for row in head_energy_rows:
                writer.writerow(asdict(row))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    if device.type == "cpu":
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sequences = list(
        iter_token_sequences(
            tokenizer=tokenizer,
            dataset_path=args.dataset_path,
            max_files=args.max_files,
            max_sequences=args.max_sequences,
            seq_length=args.seq_length,
            stride=args.stride,
            min_tokens=args.min_tokens,
        )
    )
    if not sequences:
        raise ValueError(f"No usable .txt sequences found under {args.dataset_path}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.config.use_cache = True

    rows: list[EvalMetrics] = []
    head_energy_rows: list[HeadEnergyMetric] = []
    if args.eval_baseline:
        metrics, energy_rows = score_sequences_decode(model, sequences, device, "baseline")
        rows.append(metrics)
        head_energy_rows.extend(energy_rows)

    if args.eval_sparse:
        cfg = KCacheAvgTopKConfig(
            block_size=args.block_size,
            topk_ratio=args.topk_ratio,
            first_sparse_layer=args.first_sparse_layer,
            last_sparse_layer=args.last_sparse_layer,
            min_blocks_to_keep=args.min_blocks_to_keep,
        )
        patched_layers = patch_qwen3_kcache_avg_topk(model, cfg)
        print(f"patched sparse layers: {patched_layers}")
        metrics, energy_rows = score_sequences_decode(model, sequences, device, "kcache_avg_topk")
        rows.append(metrics)
        head_energy_rows.extend(energy_rows)

    for row in rows:
        print(
            f"{row.name}: loss={row.loss:.6f} ppl={row.ppl:.4f} "
            f"mean_prob={row.mean_probability:.8f} "
            f"geo_mean_prob={row.geometric_mean_probability:.8f} "
            f"tokens={row.tokens} sequences={row.sequences}"
        )
        if row.mean_keep_block_ratio is not None:
            print(
                f"{row.name}: mean_keep_block_ratio={row.mean_keep_block_ratio:.6f} "
                f"mean_keep_token_ratio={row.mean_keep_token_ratio:.6f}"
            )
        if row.mean_attention_energy is not None:
            print(f"{row.name}: mean_attention_energy={row.mean_attention_energy:.6f}")

    write_outputs(Path(args.output_dir), rows, head_energy_rows, args)
    print(f"saved metrics to {Path(args.output_dir) / 'metrics.json'}")
    if head_energy_rows:
        print(
            "saved head energy to "
            f"{Path(args.output_dir) / 'head_energy_by_layer_head.csv'}"
        )


if __name__ == "__main__":
    main()
