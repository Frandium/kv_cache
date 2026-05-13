from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = "/mnt/workspace/Qwen3-8B"
DEFAULT_TEXT_PATH = (
    "/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt"
)
DEFAULT_PERCENTILES = "0.1,1,5,10,25,50,75,90,95,99,99.9"
DEFAULT_PLOT_METRICS = (
    "k_value,abs_k_value,delta_value,abs_delta_value,"
    "k_l2_norm,delta_l2_norm,relative_delta_l2,cosine_prev"
)


METRIC_DESCRIPTIONS = {
    "k_value": "Signed scalar components of K vectors.",
    "abs_k_value": "Absolute scalar components of K vectors.",
    "delta_value": "Signed scalar components of delta K vectors: k_i - k_{i-1}.",
    "abs_delta_value": "Absolute scalar components of delta K vectors.",
    "k_l2_norm": "L2 norm of each K vector.",
    "k_l1_norm": "L1 norm of each K vector.",
    "k_linf_norm": "L-infinity norm of each K vector.",
    "k_mean_abs": "Mean absolute component value of each K vector.",
    "delta_l2_norm": "L2 norm of each delta K vector.",
    "delta_l1_norm": "L1 norm of each delta K vector.",
    "delta_linf_norm": "L-infinity norm of each delta K vector.",
    "delta_mean_abs": "Mean absolute component value of each delta K vector.",
    "relative_delta_l2": "delta_l2_norm / max(k_{i-1}_l2_norm, eps).",
    "cosine_prev": "Cosine similarity between k_i and k_{i-1}.",
}

SIGNED_METRICS = {"k_value", "delta_value", "cosine_prev"}
FIXED_HISTOGRAM_RANGES = {"cosine_prev": (-1.0, 1.0)}


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.min_value = math.inf
        self.max_value = -math.inf

    def update(self, values: torch.Tensor | Iterable[float]) -> None:
        if isinstance(values, torch.Tensor):
            flat = values.detach().float().reshape(-1)
            flat = flat[torch.isfinite(flat)]
            if flat.numel() == 0:
                return
            count = int(flat.numel())
            total = float(flat.sum().item())
            total_sq = float(flat.square().sum().item())
            min_value = float(flat.min().item())
            max_value = float(flat.max().item())
        else:
            materialized = [
                float(value)
                for value in values
                if math.isfinite(float(value))
            ]
            if not materialized:
                return
            count = len(materialized)
            total = sum(materialized)
            total_sq = sum(value * value for value in materialized)
            min_value = min(materialized)
            max_value = max(materialized)

        self.count += count
        self.total += total
        self.total_sq += total_sq
        self.min_value = min(self.min_value, min_value)
        self.max_value = max(self.max_value, max_value)

    def base_row(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "variance": 0.0,
                "min": 0.0,
                "max": 0.0,
                "rms": 0.0,
                "cv": 0.0,
            }

        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        std = math.sqrt(variance)
        rms = math.sqrt(max(self.total_sq / self.count, 0.0))
        return {
            "count": self.count,
            "mean": mean,
            "std": std,
            "variance": variance,
            "min": self.min_value,
            "max": self.max_value,
            "rms": rms,
            "cv": std / abs(mean) if abs(mean) > 1e-12 else 0.0,
        }


class SampleStore:
    def __init__(self, max_samples: int, seed: int) -> None:
        self.max_samples = max_samples
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.samples: dict[str, torch.Tensor] = {}

    def update(self, metric: str, values: torch.Tensor) -> None:
        if self.max_samples <= 0:
            return
        flat = values.detach().float().reshape(-1).cpu()
        flat = flat[torch.isfinite(flat)]
        if flat.numel() == 0:
            return

        per_update_cap = max(1, self.max_samples // 8)
        take = min(int(flat.numel()), per_update_cap)
        if take < flat.numel():
            indices = torch.randint(
                int(flat.numel()),
                (take,),
                generator=self.generator,
                dtype=torch.long,
            )
            flat = flat[indices]

        previous = self.samples.get(metric)
        if previous is None:
            merged = flat
        else:
            merged = torch.cat([previous, flat])

        if merged.numel() > self.max_samples:
            indices = torch.randint(
                int(merged.numel()),
                (self.max_samples,),
                generator=self.generator,
                dtype=torch.long,
            )
            merged = merged[indices]
        self.samples[metric] = merged

    def get(self, metric: str) -> torch.Tensor:
        return self.samples.get(metric, torch.empty(0, dtype=torch.float32))


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_percentiles(value: str) -> list[float]:
    percentiles = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not percentiles:
        raise ValueError("At least one percentile is required.")
    for percentile in percentiles:
        if percentile < 0.0 or percentile > 100.0:
            raise ValueError(f"Percentile must be in [0, 100], got {percentile}.")
    return sorted(percentiles)


def parse_metric_list(value: str) -> list[str]:
    metrics = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(metrics) - set(METRIC_DESCRIPTIONS))
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Qwen3 K-cache from a DCLM prefix and analyze K vector "
            "values plus adjacent-token deltas per layer/head."
        )
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/kcache_value_delta")
    parser.add_argument("--max_tokens", type=int, default=5000)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument(
        "--max_chars",
        type=int,
        default=8_000_000,
        help="Read at most this many characters from text_path. Use 0 to read the full file.",
    )
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device_map",
        default="auto",
        help='Use "auto" for accelerate placement, or "none" to move the full model to --device.',
    )
    parser.add_argument(
        "--attn_implementation",
        default="auto",
        help='Attention backend passed to from_pretrained. Use "auto" to leave it unset.',
    )
    parser.add_argument("--percentiles", default=DEFAULT_PERCENTILES)
    parser.add_argument("--histogram_bins", type=int, default=200)
    parser.add_argument(
        "--histogram_clip_percentile",
        type=float,
        default=99.9,
        help="Global histogram range is chosen from this sample percentile.",
    )
    parser.add_argument(
        "--global_sample_size",
        type=int,
        default=1_000_000,
        help="Sample size used for global percentile estimates and automatic histogram ranges.",
    )
    parser.add_argument("--sample_seed", type=int, default=1234)
    parser.add_argument(
        "--save_head_histograms",
        type=str2bool,
        default=True,
        help="Write exact histogram rows for every layer/head/metric.",
    )
    parser.add_argument(
        "--make_plots",
        type=str2bool,
        default=True,
        help="Generate PNG histograms and layer/head heatmaps with matplotlib.",
    )
    parser.add_argument("--plot_metrics", default=DEFAULT_PLOT_METRICS)
    parser.add_argument("--save_k_tensors", type=str2bool, default=False)
    parser.add_argument("--save_delta_tensors", type=str2bool, default=False)
    return parser.parse_args()


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if device.type == "cpu":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def read_text_prefix(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        if max_chars > 0:
            return handle.read(max_chars)
        return handle.read()


def pick_input_device(model: torch.nn.Module, fallback_device: torch.device) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback_device


def model_forward(model: torch.nn.Module, kwargs: dict[str, Any]) -> Any:
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "cache_position" in kwargs and "cache_position" in str(exc):
            kwargs = dict(kwargs)
            kwargs.pop("cache_position")
            return model(**kwargs)
        raise


def extract_key_tensors(past_key_values: Any) -> list[torch.Tensor]:
    if hasattr(past_key_values, "key_cache"):
        return list(past_key_values.key_cache)

    if hasattr(past_key_values, "to_legacy_cache"):
        legacy_cache = past_key_values.to_legacy_cache()
        return [layer_cache[0] for layer_cache in legacy_cache]

    if isinstance(past_key_values, (list, tuple)):
        if past_key_values and isinstance(past_key_values[0], (list, tuple)):
            return [layer_cache[0] for layer_cache in past_key_values]

    if hasattr(past_key_values, "layers"):
        key_tensors: list[torch.Tensor] = []
        for layer_cache in past_key_values.layers:
            for attr_name in ("keys", "key_cache", "key_states"):
                if hasattr(layer_cache, attr_name):
                    key_tensors.append(getattr(layer_cache, attr_name))
                    break
        if key_tensors:
            return key_tensors

    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def key_tensor_to_head_token_dim(
    key_tensor: torch.Tensor,
    expected_heads: int | None,
) -> torch.Tensor:
    key = key_tensor.detach()
    if key.ndim == 4:
        batch, dim1, dim2, head_dim = key.shape
        if expected_heads is not None and dim1 == expected_heads:
            key_by_head = key.permute(1, 0, 2, 3).reshape(dim1, batch * dim2, head_dim)
        elif expected_heads is not None and dim2 == expected_heads:
            key_by_head = key.permute(2, 0, 1, 3).reshape(dim2, batch * dim1, head_dim)
        elif dim1 <= dim2:
            key_by_head = key.permute(1, 0, 2, 3).reshape(dim1, batch * dim2, head_dim)
        else:
            key_by_head = key.permute(2, 0, 1, 3).reshape(dim2, batch * dim1, head_dim)
    elif key.ndim == 3:
        dim1, dim2, head_dim = key.shape
        if expected_heads is not None and dim1 == expected_heads:
            key_by_head = key
        elif expected_heads is not None and dim2 == expected_heads:
            key_by_head = key.permute(1, 0, 2)
        elif dim1 <= dim2:
            key_by_head = key
        else:
            key_by_head = key.permute(1, 0, 2)
    else:
        raise ValueError(f"Expected 3D or 4D key tensor, got shape {tuple(key.shape)}")

    return key_by_head.float().cpu()


def percentile_field(percentile: float) -> str:
    if float(percentile).is_integer():
        return f"p{int(percentile)}"
    return "p" + str(percentile).replace(".", "_")


def summarize_values(values: torch.Tensor, percentiles: list[float]) -> dict[str, float | int]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return empty_summary_row(percentiles)

    mean = flat.mean()
    centered = flat - mean
    variance = centered.square().mean()
    std = variance.sqrt()
    rms = flat.square().mean().sqrt()
    median = torch.quantile(flat, torch.tensor(0.5, dtype=torch.float32))
    mad = (flat - median).abs().median()
    if std.item() > 0.0:
        skewness = centered.pow(3).mean() / std.pow(3)
        excess_kurtosis = centered.pow(4).mean() / variance.square() - 3.0
    else:
        skewness = torch.tensor(0.0)
        excess_kurtosis = torch.tensor(0.0)

    quantiles = torch.quantile(
        flat,
        torch.tensor([p / 100.0 for p in percentiles], dtype=torch.float32),
    )

    row: dict[str, float | int] = {
        "count": int(flat.numel()),
        "mean": float(mean),
        "std": float(std),
        "variance": float(variance),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "rms": float(rms),
        "mad": float(mad),
        "cv": float(std / mean.abs()) if mean.abs().item() > 1e-12 else 0.0,
        "skewness": float(skewness),
        "excess_kurtosis": float(excess_kurtosis),
    }
    for percentile, quantile in zip(percentiles, quantiles):
        row[percentile_field(percentile)] = float(quantile)
    return row


def empty_summary_row(percentiles: list[float]) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "count": 0,
        "mean": 0.0,
        "std": 0.0,
        "variance": 0.0,
        "min": 0.0,
        "max": 0.0,
        "rms": 0.0,
        "mad": 0.0,
        "cv": 0.0,
        "skewness": 0.0,
        "excess_kurtosis": 0.0,
    }
    for percentile in percentiles:
        row[percentile_field(percentile)] = 0.0
    return row


def make_summary_row(
    metric: str,
    scope: str,
    layer: int | None,
    head: int | None,
    values: torch.Tensor,
    percentiles: list[float],
    percentile_source: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "metric": metric,
        "scope": scope,
        "layer": layer,
        "head": head,
        "description": METRIC_DESCRIPTIONS[metric],
        "percentile_source": percentile_source,
    }
    row.update(summarize_values(values, percentiles))
    return row


def make_global_summary_row(
    metric: str,
    stats: RunningStats,
    sample: torch.Tensor,
    percentiles: list[float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "metric": metric,
        "scope": "global",
        "layer": None,
        "head": None,
        "description": METRIC_DESCRIPTIONS[metric],
        "percentile_source": "sampled" if sample.numel() else "none",
    }
    row.update(stats.base_row())
    row["mad"] = 0.0
    row["skewness"] = 0.0
    row["excess_kurtosis"] = 0.0
    if sample.numel():
        sample = sample.float()
        median = torch.quantile(sample, torch.tensor(0.5, dtype=torch.float32))
        row["mad"] = float((sample - median).abs().median())
        quantiles = torch.quantile(
            sample,
            torch.tensor([p / 100.0 for p in percentiles], dtype=torch.float32),
        )
        for percentile, quantile in zip(percentiles, quantiles):
            row[percentile_field(percentile)] = float(quantile)
    else:
        for percentile in percentiles:
            row[percentile_field(percentile)] = 0.0
    return row


def layer_metric_tensors(key_by_head: torch.Tensor, eps: float = 1e-12) -> dict[str, torch.Tensor]:
    key = key_by_head.float()
    abs_key = key.abs()
    key_l2 = torch.linalg.vector_norm(key, ord=2, dim=-1)
    metrics = {
        "k_value": key,
        "abs_k_value": abs_key,
        "k_l2_norm": key_l2,
        "k_l1_norm": abs_key.sum(dim=-1),
        "k_linf_norm": abs_key.max(dim=-1).values,
        "k_mean_abs": abs_key.mean(dim=-1),
    }

    if key.shape[1] >= 2:
        prev_key = key[:, :-1, :]
        curr_key = key[:, 1:, :]
        delta = curr_key - prev_key
        abs_delta = delta.abs()
        delta_l2 = torch.linalg.vector_norm(delta, ord=2, dim=-1)
        prev_l2 = key_l2[:, :-1]
        metrics.update(
            {
                "delta_value": delta,
                "abs_delta_value": abs_delta,
                "delta_l2_norm": delta_l2,
                "delta_l1_norm": abs_delta.sum(dim=-1),
                "delta_linf_norm": abs_delta.max(dim=-1).values,
                "delta_mean_abs": abs_delta.mean(dim=-1),
                "relative_delta_l2": delta_l2 / prev_l2.clamp_min(eps),
                "cosine_prev": F.cosine_similarity(curr_key, prev_key, dim=-1, eps=eps),
            }
        )
    else:
        empty_vector = key.new_empty((key.shape[0], 0))
        empty_scalar = key.new_empty((key.shape[0], 0, key.shape[2]))
        metrics.update(
            {
                "delta_value": empty_scalar,
                "abs_delta_value": empty_scalar,
                "delta_l2_norm": empty_vector,
                "delta_l1_norm": empty_vector,
                "delta_linf_norm": empty_vector,
                "delta_mean_abs": empty_vector,
                "relative_delta_l2": empty_vector,
                "cosine_prev": empty_vector,
            }
        )

    return metrics


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_profile_timings(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["chunk", "start_token", "end_token_exclusive", "token_count", "seconds"]
    write_csv(path, rows, fields)


@torch.inference_mode()
def build_k_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    chunk_size: int,
    input_device: torch.device,
) -> tuple[Any, list[dict[str, Any]]]:
    total_tokens = int(input_ids.shape[1])
    past_key_values = None
    timing_rows: list[dict[str, Any]] = []
    total_chunks = math.ceil(total_tokens / chunk_size)

    for chunk_idx, start in enumerate(range(0, total_tokens, chunk_size), start=1):
        end = min(start + chunk_size, total_tokens)
        chunk = input_ids[:, start:end].to(input_device)
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(start, end, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values

        if input_device.type == "cuda":
            torch.cuda.synchronize(input_device)
        start_time = time.perf_counter()
        print(
            f"profile forward chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}",
            flush=True,
        )
        outputs = model_forward(model, kwargs)
        if input_device.type == "cuda":
            torch.cuda.synchronize(input_device)
        seconds = time.perf_counter() - start_time

        past_key_values = outputs.past_key_values
        if past_key_values is None:
            raise RuntimeError("Model did not return past_key_values. Check model/config use_cache support.")

        timing_rows.append(
            {
                "chunk": chunk_idx,
                "start_token": start,
                "end_token_exclusive": end,
                "token_count": end - start,
                "seconds": seconds,
            }
        )
        del outputs, chunk
        if input_device.type == "cuda":
            torch.cuda.empty_cache()

    return past_key_values, timing_rows


def choose_histogram_ranges(
    samples: SampleStore,
    metrics: list[str],
    bins: int,
    clip_percentile: float,
) -> dict[str, tuple[float, float]]:
    if bins <= 0:
        raise ValueError("--histogram_bins must be positive.")
    if clip_percentile <= 0.0 or clip_percentile > 100.0:
        raise ValueError("--histogram_clip_percentile must be in (0, 100].")

    ranges: dict[str, tuple[float, float]] = {}
    for metric in metrics:
        if metric in FIXED_HISTOGRAM_RANGES:
            ranges[metric] = FIXED_HISTOGRAM_RANGES[metric]
            continue

        sample = samples.get(metric)
        if sample.numel() == 0:
            ranges[metric] = (-1.0, 1.0) if metric in SIGNED_METRICS else (0.0, 1.0)
            continue

        sample = sample.float()
        quantile = torch.tensor(clip_percentile / 100.0, dtype=torch.float32)
        if metric in SIGNED_METRICS:
            max_abs = float(torch.quantile(sample.abs(), quantile))
            if max_abs <= 0.0 or not math.isfinite(max_abs):
                max_abs = float(sample.abs().max())
            if max_abs <= 0.0 or not math.isfinite(max_abs):
                max_abs = 1.0
            ranges[metric] = (-max_abs, max_abs)
        else:
            right = float(torch.quantile(sample, quantile))
            if right <= 0.0 or not math.isfinite(right):
                right = float(sample.max())
            if right <= 0.0 or not math.isfinite(right):
                right = 1.0
            ranges[metric] = (0.0, right)
    return ranges


def histogram_count_row(
    metric: str,
    scope: str,
    layer: int | None,
    head: int | None,
    bin_index: int,
    bin_left: float,
    bin_right: float,
    count: int,
    total_count: int,
) -> dict[str, Any]:
    return {
        "metric": metric,
        "scope": scope,
        "layer": layer,
        "head": head,
        "bin_index": bin_index,
        "bin_left": bin_left,
        "bin_right": bin_right,
        "count": count,
        "probability": float(count / total_count) if total_count else 0.0,
    }


def histogram_counts(values: torch.Tensor, bins: int, left: float, right: float) -> tuple[torch.Tensor, int, int, int]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    total = int(flat.numel())
    if total == 0:
        return torch.zeros(bins, dtype=torch.long), 0, 0, 0
    underflow = int((flat < left).sum().item())
    overflow = int((flat > right).sum().item())
    if right <= left:
        right = left + 1.0
    counts = torch.histc(flat, bins=bins, min=left, max=right).to(torch.long)
    return counts, underflow, overflow, total


def histogram_rows_from_counts(
    metric: str,
    scope: str,
    layer: int | None,
    head: int | None,
    counts: torch.Tensor,
    underflow: int,
    overflow: int,
    total_count: int,
    left: float,
    right: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bins = int(counts.numel())
    if underflow:
        rows.append(
            histogram_count_row(
                metric,
                scope,
                layer,
                head,
                -1,
                -math.inf,
                left,
                underflow,
                total_count,
            )
        )

    width = (right - left) / bins
    for idx, count in enumerate(counts.tolist()):
        bin_left = left + idx * width
        bin_right = left + (idx + 1) * width
        rows.append(
            histogram_count_row(
                metric,
                scope,
                layer,
                head,
                idx,
                bin_left,
                bin_right,
                int(count),
                total_count,
            )
        )

    if overflow:
        rows.append(
            histogram_count_row(
                metric,
                scope,
                layer,
                head,
                bins,
                right,
                math.inf,
                overflow,
                total_count,
            )
        )
    return rows


def add_histogram_counts(
    accumulator: dict[str, dict[str, Any]],
    metric: str,
    counts: torch.Tensor,
    underflow: int,
    overflow: int,
    total_count: int,
) -> None:
    if metric not in accumulator:
        accumulator[metric] = {
            "counts": torch.zeros_like(counts),
            "underflow": 0,
            "overflow": 0,
            "total_count": 0,
        }
    accumulator[metric]["counts"] += counts
    accumulator[metric]["underflow"] += underflow
    accumulator[metric]["overflow"] += overflow
    accumulator[metric]["total_count"] += total_count


def write_histogram_rows(
    path: Path,
    rows: Iterable[dict[str, Any]],
    append: bool = False,
) -> None:
    fields = [
        "metric",
        "scope",
        "layer",
        "head",
        "bin_index",
        "bin_left",
        "bin_right",
        "count",
        "probability",
    ]
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not append:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_histograms(
    key_tensors: list[torch.Tensor],
    expected_heads: int | None,
    output_dir: Path,
    histogram_metrics: list[str],
    histogram_ranges: dict[str, tuple[float, float]],
    bins: int,
    save_head_histograms: bool,
) -> dict[str, dict[str, Any]]:
    layer_hist_path = output_dir / "histogram_by_layer.csv"
    head_hist_path = output_dir / "histogram_by_head.csv"
    write_histogram_rows(layer_hist_path, [], append=False)
    if save_head_histograms:
        write_histogram_rows(head_hist_path, [], append=False)

    global_accumulator: dict[str, dict[str, Any]] = {}

    for layer_idx, key_tensor in enumerate(key_tensors):
        print(f"writing histograms for layer {layer_idx}", flush=True)
        key_by_head = key_tensor_to_head_token_dim(key_tensor, expected_heads)
        metrics = layer_metric_tensors(key_by_head)
        layer_rows: list[dict[str, Any]] = []
        head_rows: list[dict[str, Any]] = []

        for metric in histogram_metrics:
            values = metrics[metric]
            left, right = histogram_ranges[metric]
            layer_counts, layer_underflow, layer_overflow, layer_total = histogram_counts(
                values,
                bins,
                left,
                right,
            )
            layer_rows.extend(
                histogram_rows_from_counts(
                    metric,
                    "layer",
                    layer_idx,
                    None,
                    layer_counts,
                    layer_underflow,
                    layer_overflow,
                    layer_total,
                    left,
                    right,
                )
            )
            add_histogram_counts(
                global_accumulator,
                metric,
                layer_counts,
                layer_underflow,
                layer_overflow,
                layer_total,
            )

            if save_head_histograms:
                for head_idx in range(values.shape[0]):
                    head_counts, head_underflow, head_overflow, head_total = histogram_counts(
                        values[head_idx],
                        bins,
                        left,
                        right,
                    )
                    head_rows.extend(
                        histogram_rows_from_counts(
                            metric,
                            "head",
                            layer_idx,
                            head_idx,
                            head_counts,
                            head_underflow,
                            head_overflow,
                            head_total,
                            left,
                            right,
                        )
                    )

        write_histogram_rows(layer_hist_path, layer_rows, append=True)
        if save_head_histograms:
            write_histogram_rows(head_hist_path, head_rows, append=True)

        del key_by_head, metrics

    global_rows: list[dict[str, Any]] = []
    for metric, payload in global_accumulator.items():
        left, right = histogram_ranges[metric]
        global_rows.extend(
            histogram_rows_from_counts(
                metric,
                "global",
                None,
                None,
                payload["counts"],
                payload["underflow"],
                payload["overflow"],
                payload["total_count"],
                left,
                right,
            )
        )
    write_histogram_rows(output_dir / "histogram_global.csv", global_rows, append=False)
    return global_accumulator


def analyze_k_cache(
    past_key_values: Any,
    model: torch.nn.Module,
    output_dir: Path,
    percentiles: list[float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    key_tensors = extract_key_tensors(past_key_values)
    if not key_tensors:
        raise RuntimeError("No key tensors were extracted from past_key_values.")

    expected_heads = getattr(model.config, "num_key_value_heads", None)
    summary_fields = [
        "metric",
        "scope",
        "layer",
        "head",
        "description",
        "percentile_source",
        "count",
        "mean",
        "std",
        "variance",
        "min",
        "max",
        "rms",
        "mad",
        "cv",
        "skewness",
        "excess_kurtosis",
    ] + [percentile_field(p) for p in percentiles]

    head_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    global_stats: dict[str, RunningStats] = defaultdict(RunningStats)
    samples = SampleStore(args.global_sample_size, args.sample_seed)
    cache_shapes: list[dict[str, int]] = []

    tensor_payload: dict[str, Any] | None = None
    if args.save_k_tensors or args.save_delta_tensors:
        tensor_payload = {
            "metadata": {},
            "k_by_layer": [] if args.save_k_tensors else None,
            "delta_by_layer": [] if args.save_delta_tensors else None,
        }

    for layer_idx, key_tensor in enumerate(key_tensors):
        key_by_head = key_tensor_to_head_token_dim(key_tensor, expected_heads)
        layer_heads, layer_tokens, head_dim = key_by_head.shape
        cache_shapes.append(
            {
                "layer": layer_idx,
                "kv_heads": int(layer_heads),
                "tokens": int(layer_tokens),
                "head_dim": int(head_dim),
                "raw_shape": list(key_tensor.shape),
                "raw_dtype": str(key_tensor.dtype),
                "raw_device": str(key_tensor.device),
            }
        )
        print(
            f"analyzing layer {layer_idx}: kv_heads={layer_heads} tokens={layer_tokens} head_dim={head_dim}",
            flush=True,
        )

        metrics = layer_metric_tensors(key_by_head)
        for metric, values in metrics.items():
            layer_rows.append(
                make_summary_row(metric, "layer", layer_idx, None, values, percentiles, "exact")
            )
            global_stats[metric].update(values)
            samples.update(metric, values)
            for head_idx in range(values.shape[0]):
                head_rows.append(
                    make_summary_row(
                        metric,
                        "head",
                        layer_idx,
                        head_idx,
                        values[head_idx],
                        percentiles,
                        "exact",
                    )
                )

        if tensor_payload is not None:
            if args.save_k_tensors:
                tensor_payload["k_by_layer"].append(key_by_head.to(torch.float16))
            if args.save_delta_tensors:
                tensor_payload["delta_by_layer"].append(
                    (key_by_head[:, 1:, :] - key_by_head[:, :-1, :]).to(torch.float16)
                )

        del key_by_head, metrics

    global_rows = [
        make_global_summary_row(metric, stats, samples.get(metric), percentiles)
        for metric, stats in sorted(global_stats.items())
    ]

    write_csv(output_dir / "summary_by_head.csv", head_rows, summary_fields)
    write_csv(output_dir / "summary_by_layer.csv", layer_rows, summary_fields)
    write_csv(output_dir / "summary_global.csv", global_rows, summary_fields)

    histogram_metrics = list(METRIC_DESCRIPTIONS)
    histogram_ranges = choose_histogram_ranges(
        samples,
        histogram_metrics,
        args.histogram_bins,
        args.histogram_clip_percentile,
    )
    global_histograms = write_histograms(
        key_tensors,
        expected_heads,
        output_dir,
        histogram_metrics,
        histogram_ranges,
        args.histogram_bins,
        args.save_head_histograms,
    )

    plot_metrics = parse_metric_list(args.plot_metrics)
    plot_paths: dict[str, str] = {}
    if args.make_plots:
        plot_paths = make_plots(
            output_dir,
            plot_metrics,
            histogram_ranges,
            global_histograms,
            head_rows,
        )

    if tensor_payload is not None:
        tensor_payload["metadata"] = {
            "model_name_or_path": args.model_name_or_path,
            "text_path": args.text_path,
            "max_tokens": args.max_tokens,
            "cache_shapes": cache_shapes,
        }
        torch.save(tensor_payload, output_dir / "kcache_tensors.pt")

    return {
        "cache_shapes": cache_shapes,
        "histogram_ranges": {
            metric: {"left": bounds[0], "right": bounds[1]}
            for metric, bounds in histogram_ranges.items()
        },
        "plot_paths": plot_paths,
        "summary_paths": {
            "summary_by_head": str(output_dir / "summary_by_head.csv"),
            "summary_by_layer": str(output_dir / "summary_by_layer.csv"),
            "summary_global": str(output_dir / "summary_global.csv"),
            "histogram_global": str(output_dir / "histogram_global.csv"),
            "histogram_by_layer": str(output_dir / "histogram_by_layer.csv"),
            "histogram_by_head": str(output_dir / "histogram_by_head.csv")
            if args.save_head_histograms
            else None,
        },
    }


def make_plots(
    output_dir: Path,
    plot_metrics: list[str],
    histogram_ranges: dict[str, tuple[float, float]],
    global_histograms: dict[str, dict[str, Any]],
    head_rows: list[dict[str, Any]],
) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable, skipping plots: {exc}", flush=True)
        return {}

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, str] = {}

    for metric in plot_metrics:
        if metric not in global_histograms:
            continue
        payload = global_histograms[metric]
        counts = payload["counts"].float()
        total = float(payload["total_count"])
        if total <= 0.0:
            continue
        left, right = histogram_ranges[metric]
        bins = int(counts.numel())
        width = (right - left) / bins
        centers = torch.linspace(left + 0.5 * width, right - 0.5 * width, bins)
        probabilities = counts / total

        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        ax.bar(centers.tolist(), probabilities.tolist(), width=width, align="center")
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.set_ylabel("probability")
        ax.grid(axis="y", alpha=0.25)
        tail_text = (
            f"underflow={payload['underflow'] / total:.4%}, "
            f"overflow={payload['overflow'] / total:.4%}"
        )
        ax.text(0.99, 0.98, tail_text, transform=ax.transAxes, ha="right", va="top", fontsize=8)
        fig.tight_layout()
        path = plot_dir / f"global_{metric}_histogram.png"
        fig.savefig(path)
        plt.close(fig)
        plot_paths[f"global_{metric}_histogram"] = str(path)

    heatmap_specs = [
        ("abs_k_value", "mean", "mean_abs_k_value"),
        ("abs_delta_value", "mean", "mean_abs_delta_value"),
        ("k_l2_norm", "mean", "mean_k_l2_norm"),
        ("delta_l2_norm", "mean", "mean_delta_l2_norm"),
        ("relative_delta_l2", "mean", "mean_relative_delta_l2"),
        ("cosine_prev", "mean", "mean_cosine_prev"),
    ]
    by_metric: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    max_layer = -1
    max_head = -1
    for row in head_rows:
        layer = row["layer"]
        head = row["head"]
        if layer is None or head is None:
            continue
        max_layer = max(max_layer, int(layer))
        max_head = max(max_head, int(head))
        for metric, stat_name, _ in heatmap_specs:
            if row["metric"] == metric:
                by_metric[(metric, stat_name)].append(row)

    if max_layer >= 0 and max_head >= 0:
        for metric, stat_name, file_stem in heatmap_specs:
            rows = by_metric.get((metric, stat_name), [])
            if not rows:
                continue
            matrix = torch.full((max_layer + 1, max_head + 1), float("nan"))
            for row in rows:
                matrix[int(row["layer"]), int(row["head"])] = float(row[stat_name])

            fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
            image = ax.imshow(matrix.numpy(), aspect="auto", interpolation="nearest")
            ax.set_title(file_stem)
            ax.set_xlabel("KV head")
            ax.set_ylabel("Layer")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            path = plot_dir / f"{file_stem}_heatmap.png"
            fig.savefig(path)
            plt.close(fig)
            plot_paths[f"{file_stem}_heatmap"] = str(path)

    return plot_paths


def main() -> None:
    args = parse_args()
    percentiles = parse_percentiles(args.percentiles)
    parse_metric_list(args.plot_metrics)
    text_path = Path(args.text_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not text_path.exists():
        raise FileNotFoundError(f"text_path does not exist: {text_path}")
    if args.max_tokens <= 0:
        raise ValueError("--max_tokens must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive.")

    print(f"reading text: {text_path}", flush=True)
    text = read_text_prefix(text_path, args.max_chars)
    if not text.strip():
        raise ValueError(f"No usable text read from {text_path}")

    print(f"loading tokenizer: {args.model_name_or_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    token_ids = token_ids[: args.max_tokens]
    if len(token_ids) < 2:
        raise ValueError("Tokenization produced fewer than two tokens.")
    input_ids = torch.tensor(token_ids, dtype=torch.long).view(1, -1)
    print(f"using tokens: {input_ids.shape[1]} (max_tokens={args.max_tokens})", flush=True)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, requested_device)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation

    print(f"loading causal LM: {args.model_name_or_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if args.device_map.lower() == "none":
        model = model.to(requested_device)
    model.eval()
    model.config.use_cache = True

    input_device = pick_input_device(model, requested_device)
    past_key_values, timing_rows = build_k_cache(model, input_ids, args.chunk_size, input_device)
    write_profile_timings(output_dir / "profile_timings.csv", timing_rows)

    print("analyzing K cache values and deltas", flush=True)
    analysis_summary = analyze_k_cache(past_key_values, model, output_dir, percentiles, args)
    payload = {
        "args": vars(args),
        "resolved": {
            "tokens": int(input_ids.shape[1]),
            "text_path": str(text_path),
            "model_name_or_path": args.model_name_or_path,
            "percentiles": percentiles,
            "metric_descriptions": METRIC_DESCRIPTIONS,
        },
        "profile_timings_path": str(output_dir / "profile_timings.csv"),
        "analysis": analysis_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
