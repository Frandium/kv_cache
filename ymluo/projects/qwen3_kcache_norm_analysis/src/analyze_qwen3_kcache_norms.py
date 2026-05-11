from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = "/mnt/workspace/Qwen3-0.6B"
DEFAULT_TEXT_PATH = (
    "/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt"
)
DEFAULT_PERCENTILES = "1,5,10,20,30,50,70,80,90,95,99"
DEFAULT_ENERGY_THRESHOLDS = "50,75,90,95,98,100"
DEFAULT_TOP_FRACTION = 0.30


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
            if flat.numel() == 0:
                return
            count = int(flat.numel())
            total = float(flat.sum().item())
            total_sq = float(flat.square().sum().item())
            min_value = float(flat.min().item())
            max_value = float(flat.max().item())
        else:
            materialized = [float(value) for value in values]
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

    def row(self, prefix: str) -> dict[str, float | int]:
        if self.count == 0:
            return {
                f"{prefix}_count": 0,
                f"{prefix}_mean": 0.0,
                f"{prefix}_std": 0.0,
                f"{prefix}_min": 0.0,
                f"{prefix}_max": 0.0,
            }

        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        return {
            f"{prefix}_count": self.count,
            f"{prefix}_mean": mean,
            f"{prefix}_std": math.sqrt(variance),
            f"{prefix}_min": self.min_value,
            f"{prefix}_max": self.max_value,
        }


class AttentionPruneContext:
    def __init__(self) -> None:
        self.enabled = False
        self.threshold = 1.0
        self.start = 0
        self.source_attentions: tuple[torch.Tensor, ...] | list[torch.Tensor] | None = None

    def activate(
        self,
        threshold: float,
        start: int,
        source_attentions: tuple[torch.Tensor, ...] | list[torch.Tensor],
    ) -> None:
        self.enabled = True
        self.threshold = threshold
        self.start = start
        self.source_attentions = source_attentions

    def deactivate(self) -> None:
        self.enabled = False
        self.source_attentions = None


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


def parse_energy_thresholds(value: str) -> list[float]:
    thresholds: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        threshold = float(item)
        if threshold > 1.0:
            threshold = threshold / 100.0
        if threshold <= 0.0 or threshold > 1.0:
            raise ValueError(f"Energy threshold must be in (0, 100], got {item}.")
        thresholds.append(threshold)
    if not thresholds:
        raise ValueError("At least one energy threshold is required.")
    return sorted(set(thresholds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen3 on a short DCLM prefix, compute per-token loss/PPL, "
            "and summarize attention top-k energy per layer/head."
        )
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/kcache_norms")
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument(
        "--max_chars",
        type=int,
        default=4_000_000,
        help="Read at most this many characters from text_path. Use 0 to read the full file.",
    )
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device_map",
        default="auto",
        help='Use "auto" for accelerate placement, or "none" to move the model to --device.',
    )
    parser.add_argument(
        "--attn_implementation",
        default="eager",
        help='Attention backend. "eager" is recommended because output_attentions is required.',
    )
    parser.add_argument("--percentiles", default=DEFAULT_PERCENTILES)
    parser.add_argument("--histogram_bins", type=int, default=100)
    parser.add_argument(
        "--histogram_max",
        type=float,
        default=0.0,
        help="Use 0 to set the histogram range from 0 to the observed global max norm.",
    )
    parser.add_argument("--top_fraction", type=float, default=DEFAULT_TOP_FRACTION)
    parser.add_argument("--energy_thresholds", default=DEFAULT_ENERGY_THRESHOLDS)
    parser.add_argument(
        "--save_attention_token_rows",
        type=str2bool,
        default=True,
        help="Write one row per layer/head/query token with top-k counts and loss/PPL.",
    )
    parser.add_argument(
        "--compute_pruned_loss_ppl",
        type=str2bool,
        default=True,
        help=(
            "Re-run each chunk with attention positions pruned to the requested "
            "energy thresholds and write threshold-specific model loss/PPL."
        ),
    )
    parser.add_argument(
        "--save_pruned_token_rows",
        type=str2bool,
        default=True,
        help="Write one row per energy threshold and target token for pruned model loss/PPL.",
    )
    parser.add_argument("--save_norm_tensors", type=str2bool, default=False)
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


def model_forward(model: torch.nn.Module, kwargs: dict[str, Any]):
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "cache_position" in kwargs and "cache_position" in str(exc):
            kwargs = dict(kwargs)
            kwargs.pop("cache_position")
            return model(**kwargs)
        raise


def safe_exp(value: float) -> float:
    if value >= 80.0:
        return math.inf
    return math.exp(value)


def threshold_field(threshold: float) -> str:
    percent = threshold * 100.0
    if percent.is_integer():
        return str(int(percent))
    return str(percent).replace(".", "_")


def top_fraction_field(top_fraction: float) -> str:
    percent = top_fraction * 100.0
    if percent.is_integer():
        return f"top{int(percent)}"
    return "top" + str(percent).replace(".", "_")


def token_piece(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.convert_ids_to_tokens([token_id])[0]
    except Exception:
        return ""


def token_text(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id])
    except Exception:
        return ""


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def key_tensor_to_norm_matrix(
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

    return torch.linalg.vector_norm(key_by_head.float(), ord=2, dim=-1).cpu()


def percentile_field(percentile: float) -> str:
    if float(percentile).is_integer():
        return f"p{int(percentile)}"
    return "p" + str(percentile).replace(".", "_")


def summarize_values(values: torch.Tensor, percentiles: list[float]) -> dict[str, float | int]:
    flat = values.reshape(-1).float()
    if flat.numel() == 0:
        raise ValueError("Cannot summarize an empty tensor.")

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
        "cv": float(std / mean) if mean.item() != 0.0 else 0.0,
        "skewness": float(skewness),
        "excess_kurtosis": float(excess_kurtosis),
    }
    for percentile, quantile in zip(percentiles, quantiles):
        row[percentile_field(percentile)] = float(quantile)
    return row


def make_summary_row(
    scope: str,
    layer: int | None,
    head: int | None,
    values: torch.Tensor,
    percentiles: list[float],
) -> dict[str, Any]:
    row: dict[str, Any] = {"scope": scope, "layer": layer, "head": head}
    row.update(summarize_values(values, percentiles))
    return row


def histogram_rows(
    scope: str,
    layer: int | None,
    head: int | None,
    values: torch.Tensor,
    bins: int,
    hist_max: float,
) -> list[dict[str, Any]]:
    flat = values.reshape(-1).float()
    counts = torch.histc(flat, bins=bins, min=0.0, max=hist_max)
    width = hist_max / bins
    total = int(flat.numel())
    rows: list[dict[str, Any]] = []
    for idx, count in enumerate(counts.tolist()):
        left = idx * width
        right = (idx + 1) * width
        rows.append(
            {
                "scope": scope,
                "layer": layer,
                "head": head,
                "bin_index": idx,
                "bin_left": left,
                "bin_right": right,
                "count": int(count),
                "probability": float(count / total) if total else 0.0,
            }
        )

    overflow = int((flat > hist_max).sum())
    if overflow:
        rows.append(
            {
                "scope": scope,
                "layer": layer,
                "head": head,
                "bin_index": bins,
                "bin_left": hist_max,
                "bin_right": math.inf,
                "count": overflow,
                "probability": float(overflow / total) if total else 0.0,
            }
        )
    return rows


def valid_attention_mask(
    attention: torch.Tensor,
    start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attention.ndim != 3:
        raise ValueError(f"Expected attention shape [heads, query, key], got {tuple(attention.shape)}")

    _, query_count, key_count = attention.shape
    device = attention.device
    query_indices = torch.arange(start, start + query_count, device=device, dtype=torch.long)
    valid_counts = (query_indices + 1).clamp(max=key_count)
    key_indices = torch.arange(key_count, device=device, dtype=torch.long)
    mask = key_indices.view(1, 1, key_count) < valid_counts.view(1, query_count, 1)
    return mask.expand(attention.shape[0], query_count, key_count), valid_counts


def sorted_attention_by_energy(
    attention: torch.Tensor,
    start: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_mask, valid_counts = valid_attention_mask(attention, start)
    sort_scores = attention.float().masked_fill(~valid_mask, -1.0)
    _, sorted_indices = torch.sort(sort_scores, dim=-1, descending=True)
    sorted_valid_mask = torch.gather(valid_mask, dim=-1, index=sorted_indices)
    sorted_values = torch.gather(attention.float(), dim=-1, index=sorted_indices)
    sorted_values = sorted_values.masked_fill(~sorted_valid_mask, 0.0)
    denom = attention.float().masked_fill(~valid_mask, 0.0).sum(dim=-1).clamp_min(1e-12)
    cumulative_energy = sorted_values.cumsum(dim=-1) / denom.unsqueeze(-1)
    return sorted_indices, cumulative_energy, valid_mask, valid_counts


def threshold_counts_from_cumulative(
    cumulative_energy: torch.Tensor,
    valid_counts: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    heads, query_count, _ = cumulative_energy.shape
    valid_counts_by_head = valid_counts.view(1, query_count).expand(heads, query_count)
    if math.isclose(threshold, 1.0):
        return valid_counts_by_head
    counts = (cumulative_energy < threshold).sum(dim=-1).long() + 1
    return torch.minimum(counts, valid_counts_by_head)


def build_threshold_keep_mask(
    attention: torch.Tensor,
    start: int,
    threshold: float,
) -> torch.Tensor:
    sorted_indices, cumulative_energy, valid_mask, valid_counts = sorted_attention_by_energy(attention, start)
    if math.isclose(threshold, 1.0):
        return valid_mask

    counts = threshold_counts_from_cumulative(cumulative_energy, valid_counts, threshold)
    rank = torch.arange(attention.shape[-1], device=attention.device, dtype=torch.long)
    keep_sorted = rank.view(1, 1, -1) < counts.unsqueeze(-1)
    keep = torch.zeros_like(valid_mask)
    keep.scatter_(dim=-1, index=sorted_indices, src=keep_sorted)
    return keep & valid_mask


def make_additive_prune_mask(
    keep_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not dtype.is_floating_point:
        dtype = torch.float32
    additive_mask = torch.zeros(
        (1, keep_mask.shape[0], keep_mask.shape[1], keep_mask.shape[2]),
        device=device,
        dtype=dtype,
    )
    min_value = torch.finfo(dtype).min
    return additive_mask.masked_fill(~keep_mask.to(device=device).unsqueeze(0), min_value)


def get_attention_mask_arg_index(module: torch.nn.Module) -> int | None:
    try:
        parameters = list(inspect.signature(module.forward).parameters)
    except (TypeError, ValueError):
        return None
    try:
        return parameters.index("attention_mask")
    except ValueError:
        return None


def find_attention_modules(model: torch.nn.Module) -> list[tuple[int, str, torch.nn.Module]]:
    candidates: list[tuple[int, str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if not all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj")):
            continue

        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            parts = name.split(".")
            for idx, part in enumerate(parts[:-1]):
                if part == "layers" and parts[idx + 1].isdigit():
                    layer_idx = int(parts[idx + 1])
                    break
        if layer_idx is None:
            layer_idx = len(candidates)
        candidates.append((int(layer_idx), name, module))

    candidates.sort(key=lambda item: item[0])
    return candidates


def install_attention_prune_hooks(
    model: torch.nn.Module,
    prune_context: AttentionPruneContext,
) -> list[Any]:
    handles: list[Any] = []
    attention_modules = find_attention_modules(model)
    if not attention_modules:
        raise RuntimeError("Could not find Qwen/LLaMA-style attention modules to install pruning hooks.")

    for layer_idx, _, module in attention_modules:
        attention_mask_arg_index = get_attention_mask_arg_index(module)

        def hook(
            hooked_module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            *,
            layer_idx: int = layer_idx,
            attention_mask_arg_index: int | None = attention_mask_arg_index,
        ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
            if (
                not prune_context.enabled
                or prune_context.source_attentions is None
                or math.isclose(prune_context.threshold, 1.0)
            ):
                return None

            source_attention = prune_context.source_attentions[layer_idx]
            if source_attention.ndim != 4:
                raise ValueError(
                    f"Expected source attention shape [batch, heads, query, key], "
                    f"got {tuple(source_attention.shape)}"
                )
            keep_mask = build_threshold_keep_mask(
                source_attention[0].float(),
                prune_context.start,
                prune_context.threshold,
            )

            base_mask = kwargs.get("attention_mask")
            base_mask_from_args = False
            args_list = list(args)
            if base_mask is None and attention_mask_arg_index is not None and attention_mask_arg_index < len(args_list):
                base_mask = args_list[attention_mask_arg_index]
                base_mask_from_args = True

            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args_list:
                hidden_states = args_list[0]
            if hidden_states is None:
                raise RuntimeError("Cannot infer attention mask device because hidden_states was not passed.")

            mask_device = base_mask.device if isinstance(base_mask, torch.Tensor) else hidden_states.device
            mask_dtype = base_mask.dtype if isinstance(base_mask, torch.Tensor) else hidden_states.dtype
            additive_mask = make_additive_prune_mask(keep_mask, mask_device, mask_dtype)
            if isinstance(base_mask, torch.Tensor):
                base_mask = base_mask[:, :, : additive_mask.shape[-2], : additive_mask.shape[-1]]
                new_mask = base_mask + additive_mask
            else:
                new_mask = additive_mask

            if base_mask_from_args and attention_mask_arg_index is not None:
                args_list[attention_mask_arg_index] = new_mask
            else:
                kwargs = dict(kwargs)
                kwargs["attention_mask"] = new_mask

            return tuple(args_list), kwargs

        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))

    print(f"installed attention pruning hooks on {len(attention_modules)} modules", flush=True)
    return handles


def build_attention_token_fields(thresholds: list[float]) -> list[str]:
    fields = [
        "layer",
        "head",
        "query_token_index",
        "target_token_index",
        "loss",
        "ppl",
        "valid_attention_tokens",
        "top_fraction",
        "top_fraction_token_count",
        "top_fraction_energy",
    ]
    for threshold in thresholds:
        suffix = threshold_field(threshold)
        fields.extend(
            [
                f"topk_count_energy_{suffix}",
                f"topk_fraction_energy_{suffix}",
            ]
        )
    return fields


def build_pruned_summary_fields() -> list[str]:
    return [
        "energy_threshold",
        "loss_count",
        "loss_mean",
        "loss_std",
        "loss_min",
        "loss_max",
        "ppl_count",
        "ppl_mean",
        "ppl_std",
        "ppl_min",
        "ppl_max",
    ]


def compute_token_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    losses = F.cross_entropy(logits.float(), targets.to(logits.device), reduction="none").detach().cpu()
    ppls = torch.tensor([safe_exp(float(loss)) for loss in losses], dtype=torch.float32)
    return losses, ppls


def update_attention_metrics(
    attentions: tuple[torch.Tensor, ...] | list[torch.Tensor],
    start: int,
    valid_query_count: int,
    losses: torch.Tensor,
    ppls: torch.Tensor,
    total_tokens: int,
    top_fraction: float,
    energy_thresholds: list[float],
    attention_writer: csv.writer | None,
    top_fraction_energy_stats: dict[tuple[int, int], RunningStats],
    top_fraction_count_stats: dict[tuple[int, int], RunningStats],
    threshold_count_stats: dict[tuple[int, int, float], RunningStats],
    threshold_fraction_stats: dict[tuple[int, int, float], RunningStats],
) -> None:
    if valid_query_count <= 0:
        return

    query_indices = torch.arange(start, start + valid_query_count, dtype=torch.long)
    target_indices = query_indices + 1
    for layer_idx, attention in enumerate(attentions):
        if attention is None:
            raise RuntimeError(
                "Model returned empty attentions. Use --attn_implementation eager "
                "or another backend that supports output_attentions=True."
            )
        if attention.ndim != 4:
            raise ValueError(f"Expected attention shape [batch, heads, query, key], got {tuple(attention.shape)}")

        layer_attention = attention[0, :, :valid_query_count, :].float()
        num_heads = int(layer_attention.shape[0])
        key_count = int(layer_attention.shape[-1])
        expected_key_count = min(start + attention.shape[2], total_tokens)
        if key_count != expected_key_count:
            print(
                f"warning: layer {layer_idx} attention key count is {key_count}, expected {expected_key_count}",
                flush=True,
            )

        _, cumulative_energy, _, valid_counts = sorted_attention_by_energy(layer_attention, start)
        valid_counts_cpu = valid_counts.detach().cpu()
        top_counts = torch.ceil(valid_counts.float() * top_fraction).long().clamp(min=1)
        top_counts = torch.minimum(top_counts, torch.full_like(top_counts, key_count))
        top_gather = top_counts.view(1, valid_query_count, 1).expand(num_heads, valid_query_count, 1) - 1
        top_energy = cumulative_energy.gather(dim=-1, index=top_gather).squeeze(-1)

        threshold_counts: dict[float, torch.Tensor] = {}
        threshold_fractions: dict[float, torch.Tensor] = {}
        valid_counts_by_head = valid_counts.view(1, valid_query_count).expand(num_heads, valid_query_count)
        for threshold in energy_thresholds:
            counts = threshold_counts_from_cumulative(cumulative_energy, valid_counts, threshold)
            threshold_counts[threshold] = counts
            threshold_fractions[threshold] = counts.float() / valid_counts_by_head.float()

        for head_idx in range(num_heads):
            key = (layer_idx, head_idx)
            top_fraction_energy_stats.setdefault(key, RunningStats()).update(top_energy[head_idx])
            top_fraction_count_stats.setdefault(key, RunningStats()).update(top_counts.float())

            for threshold in energy_thresholds:
                threshold_key = (layer_idx, head_idx, threshold)
                threshold_count_stats.setdefault(threshold_key, RunningStats()).update(
                    threshold_counts[threshold][head_idx].float()
                )
                threshold_fraction_stats.setdefault(threshold_key, RunningStats()).update(
                    threshold_fractions[threshold][head_idx]
                )

            if attention_writer is not None:
                top_energy_cpu = top_energy[head_idx].detach().cpu().tolist()
                for query_offset in range(valid_query_count):
                    row: list[Any] = [
                        layer_idx,
                        head_idx,
                        int(query_indices[query_offset]),
                        int(target_indices[query_offset]),
                        float(losses[query_offset]),
                        float(ppls[query_offset]),
                        int(valid_counts_cpu[query_offset]),
                        top_fraction,
                        int(top_counts[query_offset].detach().cpu()),
                        float(top_energy_cpu[query_offset]),
                    ]
                    for threshold in energy_thresholds:
                        count_value = int(threshold_counts[threshold][head_idx, query_offset].detach().cpu())
                        fraction_value = float(
                            threshold_fractions[threshold][head_idx, query_offset].detach().cpu()
                        )
                        row.extend([count_value, fraction_value])
                    attention_writer.writerow(row)

        del layer_attention, cumulative_energy


def run_causal_attention_analysis(
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    chunk_size: int,
    input_device: torch.device,
    output_dir: Path,
    top_fraction: float,
    energy_thresholds: list[float],
    save_attention_token_rows: bool,
    compute_pruned_loss_ppl: bool,
    save_pruned_token_rows: bool,
) -> tuple[Any, dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("--chunk_size must be positive.")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError(f"--top_fraction must be in (0, 1], got {top_fraction}.")

    total_tokens = int(input_ids.shape[1])
    if total_tokens < 2:
        raise ValueError("At least two tokens are required to compute next-token loss.")

    token_loss_path = output_dir / "token_loss_ppl.csv"
    attention_token_path = output_dir / "attention_token_topk.csv"
    pruned_summary_path = output_dir / "attention_pruned_loss_ppl_by_threshold.csv"
    pruned_token_path = output_dir / "attention_pruned_token_loss_ppl.csv"
    top_fraction_energy_stats: dict[tuple[int, int], RunningStats] = {}
    top_fraction_count_stats: dict[tuple[int, int], RunningStats] = {}
    threshold_count_stats: dict[tuple[int, int, float], RunningStats] = {}
    threshold_fraction_stats: dict[tuple[int, int, float], RunningStats] = {}
    loss_stats = RunningStats()
    ppl_stats = RunningStats()
    past_key_values = None
    pruned_thresholds = [threshold for threshold in energy_thresholds if not math.isclose(threshold, 1.0)]
    pruned_past_key_values: dict[float, Any] = {threshold: None for threshold in pruned_thresholds}
    pruned_loss_stats: dict[float, RunningStats] = {
        threshold: RunningStats() for threshold in energy_thresholds
    }
    pruned_ppl_stats: dict[float, RunningStats] = {
        threshold: RunningStats() for threshold in energy_thresholds
    }
    prune_context = AttentionPruneContext()
    prune_handles: list[Any] = []
    if compute_pruned_loss_ppl and pruned_thresholds:
        prune_handles = install_attention_prune_hooks(model, prune_context)

    attention_handle = None
    attention_writer = None
    if save_attention_token_rows:
        attention_handle = attention_token_path.open("w", newline="", encoding="utf-8")
        attention_writer = csv.writer(attention_handle)
        attention_writer.writerow(build_attention_token_fields(energy_thresholds))

    pruned_token_handle = None
    pruned_token_writer = None
    if compute_pruned_loss_ppl and save_pruned_token_rows:
        pruned_token_handle = pruned_token_path.open("w", newline="", encoding="utf-8")
        pruned_token_writer = csv.writer(pruned_token_handle)
        pruned_token_writer.writerow(
            [
                "energy_threshold",
                "query_token_index",
                "target_token_index",
                "target_token_id",
                "target_token_piece",
                "target_text",
                "loss",
                "ppl",
            ]
        )

    print(f"writing per-token loss/PPL: {token_loss_path}", flush=True)
    if save_attention_token_rows:
        print(f"writing per-token attention top-k rows: {attention_token_path}", flush=True)
    if compute_pruned_loss_ppl:
        print(f"writing pruned attention loss/PPL summary: {pruned_summary_path}", flush=True)
        if save_pruned_token_rows:
            print(f"writing pruned attention token loss/PPL: {pruned_token_path}", flush=True)

    model.eval()
    try:
        with token_loss_path.open("w", newline="", encoding="utf-8") as token_loss_handle:
            token_loss_writer = csv.writer(token_loss_handle)
            token_loss_writer.writerow(
                [
                    "query_token_index",
                    "target_token_index",
                    "target_token_id",
                    "target_token_piece",
                    "target_text",
                    "loss",
                    "ppl",
                ]
            )

            with torch.inference_mode():
                total_chunks = math.ceil(total_tokens / chunk_size)
                for chunk_idx, start in enumerate(range(0, total_tokens, chunk_size), start=1):
                    end = min(start + chunk_size, total_tokens)
                    chunk = input_ids[:, start:end].to(input_device)
                    kwargs: dict[str, Any] = {
                        "input_ids": chunk,
                        "use_cache": True,
                        "return_dict": True,
                        "output_attentions": True,
                        "output_hidden_states": False,
                        "cache_position": torch.arange(start, end, device=input_device),
                    }
                    if past_key_values is not None:
                        kwargs["past_key_values"] = past_key_values

                    print(
                        f"forward chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}",
                        flush=True,
                    )
                    outputs = model_forward(model, kwargs)
                    past_key_values = outputs.past_key_values
                    if past_key_values is None:
                        raise RuntimeError(
                            "Model did not return past_key_values. Check model/config use_cache support."
                        )
                    if outputs.attentions is None:
                        raise RuntimeError(
                            "Model did not return attentions. Use --attn_implementation eager."
                        )

                    valid_query_count = max(0, min(end - start, total_tokens - 1 - start))
                    if valid_query_count > 0:
                        logits = outputs.logits[0, :valid_query_count, :].float()
                        targets = input_ids[0, start + 1 : start + valid_query_count + 1].to(logits.device)
                        losses, ppls = compute_token_losses(logits, targets)
                        loss_stats.update(losses)
                        ppl_stats.update(ppls)

                        for offset in range(valid_query_count):
                            target_idx = start + offset + 1
                            target_id = int(input_ids[0, target_idx])
                            token_loss_writer.writerow(
                                [
                                    start + offset,
                                    target_idx,
                                    target_id,
                                    token_piece(tokenizer, target_id),
                                    token_text(tokenizer, target_id),
                                    float(losses[offset]),
                                    float(ppls[offset]),
                                ]
                            )

                        print(
                            f"processing attentions for chunk {chunk_idx}/{total_chunks}: "
                            f"{len(outputs.attentions)} layers, {valid_query_count} query tokens",
                            flush=True,
                        )
                        update_attention_metrics(
                            outputs.attentions,
                            start,
                            valid_query_count,
                            losses,
                            ppls,
                            total_tokens,
                            top_fraction,
                            energy_thresholds,
                            attention_writer,
                            top_fraction_energy_stats,
                            top_fraction_count_stats,
                            threshold_count_stats,
                            threshold_fraction_stats,
                        )

                        if compute_pruned_loss_ppl:
                            for threshold in energy_thresholds:
                                if not math.isclose(threshold, 1.0):
                                    continue
                                pruned_loss_stats[threshold].update(losses)
                                pruned_ppl_stats[threshold].update(ppls)
                                if pruned_token_writer is not None:
                                    for offset in range(valid_query_count):
                                        target_idx = start + offset + 1
                                        target_id = int(input_ids[0, target_idx])
                                        pruned_token_writer.writerow(
                                            [
                                                threshold,
                                                start + offset,
                                                target_idx,
                                                target_id,
                                                token_piece(tokenizer, target_id),
                                                token_text(tokenizer, target_id),
                                                float(losses[offset]),
                                                float(ppls[offset]),
                                            ]
                                        )

                            for threshold in pruned_thresholds:
                                print(
                                    f"pruned forward chunk {chunk_idx}/{total_chunks}: "
                                    f"energy={threshold_field(threshold)} tokens {start}-{end - 1}",
                                    flush=True,
                                )
                                prune_context.activate(threshold, start, outputs.attentions)
                                pruned_chunk = input_ids[:, start:end].to(input_device)
                                pruned_kwargs: dict[str, Any] = {
                                    "input_ids": pruned_chunk,
                                    "use_cache": True,
                                    "return_dict": True,
                                    "output_attentions": False,
                                    "output_hidden_states": False,
                                    "cache_position": torch.arange(start, end, device=input_device),
                                }
                                if pruned_past_key_values[threshold] is not None:
                                    pruned_kwargs["past_key_values"] = pruned_past_key_values[threshold]

                                pruned_outputs = model_forward(model, pruned_kwargs)
                                prune_context.deactivate()
                                pruned_past_key_values[threshold] = pruned_outputs.past_key_values
                                if pruned_past_key_values[threshold] is None:
                                    raise RuntimeError(
                                        "Model did not return past_key_values during pruned forward."
                                    )

                                pruned_logits = pruned_outputs.logits[0, :valid_query_count, :]
                                pruned_targets = input_ids[0, start + 1 : start + valid_query_count + 1]
                                pruned_losses, pruned_ppls = compute_token_losses(
                                    pruned_logits,
                                    pruned_targets,
                                )
                                pruned_loss_stats[threshold].update(pruned_losses)
                                pruned_ppl_stats[threshold].update(pruned_ppls)
                                if pruned_token_writer is not None:
                                    for offset in range(valid_query_count):
                                        target_idx = start + offset + 1
                                        target_id = int(input_ids[0, target_idx])
                                        pruned_token_writer.writerow(
                                            [
                                                threshold,
                                                start + offset,
                                                target_idx,
                                                target_id,
                                                token_piece(tokenizer, target_id),
                                                token_text(tokenizer, target_id),
                                                float(pruned_losses[offset]),
                                                float(pruned_ppls[offset]),
                                            ]
                                        )
                                del pruned_outputs, pruned_chunk

                    del outputs, chunk
                    if input_device.type == "cuda":
                        torch.cuda.empty_cache()
                    print(f"completed chunk {chunk_idx}/{total_chunks}: {end}/{total_tokens} tokens", flush=True)
    finally:
        prune_context.deactivate()
        if attention_handle is not None:
            attention_handle.close()
        if pruned_token_handle is not None:
            pruned_token_handle.close()
        for handle in prune_handles:
            handle.remove()

    top_fraction_rows: list[dict[str, Any]] = []
    top_prefix = top_fraction_field(top_fraction)
    for layer_idx, head_idx in sorted(top_fraction_energy_stats):
        key = (layer_idx, head_idx)
        row: dict[str, Any] = {
            "layer": layer_idx,
            "head": head_idx,
            "top_fraction": top_fraction,
        }
        row.update(top_fraction_energy_stats[key].row(f"{top_prefix}_energy"))
        row.update(top_fraction_count_stats[key].row(f"{top_prefix}_token_count"))
        top_fraction_rows.append(row)

    threshold_rows: list[dict[str, Any]] = []
    for layer_idx, head_idx, threshold in sorted(threshold_count_stats):
        key = (layer_idx, head_idx, threshold)
        row = {
            "layer": layer_idx,
            "head": head_idx,
            "energy_threshold": threshold,
        }
        row.update(threshold_count_stats[key].row("topk_token_count"))
        row.update(threshold_fraction_stats[key].row("topk_token_fraction"))
        threshold_rows.append(row)

    top_fraction_fields = [
        "layer",
        "head",
        "top_fraction",
        f"{top_prefix}_energy_count",
        f"{top_prefix}_energy_mean",
        f"{top_prefix}_energy_std",
        f"{top_prefix}_energy_min",
        f"{top_prefix}_energy_max",
        f"{top_prefix}_token_count_count",
        f"{top_prefix}_token_count_mean",
        f"{top_prefix}_token_count_std",
        f"{top_prefix}_token_count_min",
        f"{top_prefix}_token_count_max",
    ]
    threshold_fields = [
        "layer",
        "head",
        "energy_threshold",
        "topk_token_count_count",
        "topk_token_count_mean",
        "topk_token_count_std",
        "topk_token_count_min",
        "topk_token_count_max",
        "topk_token_fraction_count",
        "topk_token_fraction_mean",
        "topk_token_fraction_std",
        "topk_token_fraction_min",
        "topk_token_fraction_max",
    ]
    write_csv(output_dir / "attention_top_fraction_energy_by_head.csv", top_fraction_rows, top_fraction_fields)
    write_csv(output_dir / "attention_energy_thresholds_by_head.csv", threshold_rows, threshold_fields)

    pruned_summary_rows: list[dict[str, Any]] = []
    if compute_pruned_loss_ppl:
        for threshold in energy_thresholds:
            row = {"energy_threshold": threshold}
            row.update(pruned_loss_stats[threshold].row("loss"))
            row.update(pruned_ppl_stats[threshold].row("ppl"))
            pruned_summary_rows.append(row)
        write_csv(pruned_summary_path, pruned_summary_rows, build_pruned_summary_fields())

    return past_key_values, {
        "token_loss_path": str(token_loss_path),
        "attention_token_path": str(attention_token_path) if save_attention_token_rows else None,
        "top_fraction_path": str(output_dir / "attention_top_fraction_energy_by_head.csv"),
        "threshold_path": str(output_dir / "attention_energy_thresholds_by_head.csv"),
        "pruned_loss_ppl_path": str(pruned_summary_path) if compute_pruned_loss_ppl else None,
        "pruned_token_loss_ppl_path": (
            str(pruned_token_path) if compute_pruned_loss_ppl and save_pruned_token_rows else None
        ),
        "loss": loss_stats.row("loss"),
        "ppl": ppl_stats.row("ppl"),
        "pruned_loss_ppl_by_threshold": pruned_summary_rows,
        "energy_thresholds": energy_thresholds,
        "top_fraction": top_fraction,
        "loss_bearing_tokens": total_tokens - 1,
    }


def write_kcache_norm_outputs(
    past_key_values: Any,
    model: torch.nn.Module,
    output_dir: Path,
    args: argparse.Namespace,
    input_token_count: int,
    percentiles: list[float],
) -> dict[str, Any]:
    key_tensors = extract_key_tensors(past_key_values)
    expected_heads = getattr(model.config, "num_key_value_heads", None)

    layer_norms: list[torch.Tensor] = []
    cache_shapes: list[dict[str, int]] = []
    for layer_idx, key_tensor in enumerate(key_tensors):
        norms = key_tensor_to_norm_matrix(key_tensor, expected_heads)
        layer_norms.append(norms)
        cache_shapes.append(
            {
                "layer": layer_idx,
                "kv_heads": int(norms.shape[0]),
                "tokens": int(norms.shape[1]),
            }
        )
        print(
            f"k-cache norms layer {layer_idx}: heads={norms.shape[0]} tokens={norms.shape[1]}",
            flush=True,
        )

    if not layer_norms:
        raise RuntimeError("No key tensors were extracted from past_key_values.")

    global_max = max(float(norms.max()) for norms in layer_norms)
    hist_max = args.histogram_max if args.histogram_max > 0.0 else global_max
    if hist_max <= 0.0:
        hist_max = 1.0

    head_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    head_hist_rows: list[dict[str, Any]] = []
    layer_hist_rows: list[dict[str, Any]] = []

    for layer_idx, norms in enumerate(layer_norms):
        layer_rows.append(make_summary_row("layer", layer_idx, None, norms, percentiles))
        layer_hist_rows.extend(
            histogram_rows("layer", layer_idx, None, norms, args.histogram_bins, hist_max)
        )
        for head_idx in range(norms.shape[0]):
            head_values = norms[head_idx]
            head_rows.append(make_summary_row("head", layer_idx, head_idx, head_values, percentiles))
            head_hist_rows.extend(
                histogram_rows(
                    "head",
                    layer_idx,
                    head_idx,
                    head_values,
                    args.histogram_bins,
                    hist_max,
                )
            )

    global_values = torch.cat([norms.reshape(-1) for norms in layer_norms])
    global_row = make_summary_row("global", None, None, global_values, percentiles)
    global_hist_rows = histogram_rows(
        "global",
        None,
        None,
        global_values,
        args.histogram_bins,
        hist_max,
    )

    stat_fields = [
        "scope",
        "layer",
        "head",
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
    hist_fields = [
        "scope",
        "layer",
        "head",
        "bin_index",
        "bin_left",
        "bin_right",
        "count",
        "probability",
    ]

    write_csv(output_dir / "summary_by_head.csv", head_rows, stat_fields)
    write_csv(output_dir / "summary_by_layer.csv", layer_rows, stat_fields)
    write_csv(output_dir / "histogram_by_head.csv", head_hist_rows, hist_fields)
    write_csv(output_dir / "histogram_by_layer.csv", layer_hist_rows, hist_fields)
    write_csv(output_dir / "histogram_global.csv", global_hist_rows, hist_fields)

    summary_payload = {
        "args": vars(args),
        "resolved": {
            "tokens": int(input_token_count),
            "text_path": str(args.text_path),
            "model_name_or_path": args.model_name_or_path,
            "percentiles": percentiles,
            "histogram_max": hist_max,
            "cache_shapes": cache_shapes,
        },
        "global": global_row,
        "by_layer": layer_rows,
        "by_head": head_rows,
    }

    if args.save_norm_tensors:
        torch.save(
            {
                "metadata": summary_payload["resolved"],
                "layer_norms": layer_norms,
            },
            output_dir / "layer_head_norms.pt",
        )

    return summary_payload


def main() -> None:
    args = parse_args()
    percentiles = parse_percentiles(args.percentiles)
    energy_thresholds = parse_energy_thresholds(args.energy_thresholds)
    text_path = Path(args.text_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not text_path.exists():
        raise FileNotFoundError(f"text_path does not exist: {text_path}")
    if args.max_tokens <= 0:
        raise ValueError("--max_tokens must be positive.")
    if args.histogram_bins <= 0:
        raise ValueError("--histogram_bins must be positive.")

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
    model.config.use_cache = True

    input_device = pick_input_device(model, requested_device)
    past_key_values, attention_summary = run_causal_attention_analysis(
        model,
        tokenizer,
        input_ids,
        args.chunk_size,
        input_device,
        output_dir,
        args.top_fraction,
        energy_thresholds,
        args.save_attention_token_rows,
        args.compute_pruned_loss_ppl,
        args.save_pruned_token_rows,
    )

    print("summarizing k-cache norm tensors", flush=True)
    kcache_summary = write_kcache_norm_outputs(
        past_key_values,
        model,
        output_dir,
        args,
        int(input_ids.shape[1]),
        percentiles,
    )
    kcache_summary["attention_analysis"] = attention_summary

    (output_dir / "summary.json").write_text(
        json.dumps(kcache_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
