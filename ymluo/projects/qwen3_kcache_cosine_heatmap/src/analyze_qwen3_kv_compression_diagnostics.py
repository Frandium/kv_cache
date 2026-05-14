from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from analyze_qwen3_kcache_cosine_heatmap import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PERCENTILES,
    DEFAULT_TEXT_PATH,
    AutoModelForCausalLM,
    AutoTokenizer,
    compute_cosine_matrix,
    key_tensor_to_head_token_dim,
    model_forward,
    named_torch_dtype,
    parse_index_spec,
    parse_percentiles,
    pick_input_device,
    read_text_prefix,
    resolve_dtype,
    resolve_similarity_device,
    save_layer_head_metric_heatmap,
    save_similarity_heatmap,
    stat_fieldnames,
    str2bool,
    summarize_similarity_matrix,
    write_csv,
    write_profile_timings,
    write_tokens_csv,
)


DEFAULT_ENERGY_THRESHOLDS = "50,75,90,95,99"
DEFAULT_VALIDATION_RANKS = "1,2,4,8,16,32,64,128"


def parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("At least one integer value is required.")
    if any(item <= 0 for item in items):
        raise ValueError(f"All integer values must be positive: {value!r}")
    return sorted(set(items))


def parse_energy_thresholds(value: str) -> list[float]:
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("At least one SVD energy threshold is required.")
    for threshold in thresholds:
        if threshold <= 0.0 or threshold > 100.0:
            raise ValueError(f"Energy threshold must be in (0, 100], got {threshold}.")
    return sorted(set(thresholds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Qwen3 K/V caches, compare raw vs mean-centered K cosine, "
            "write K/V SVD spectra, and optionally validate low-rank K/V "
            "approximations with sampled attention-weighted errors."
        )
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/kv_compression_diagnostics")
    parser.add_argument("--max_tokens", type=int, default=5000)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_max_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="auto")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--heads", default="all")
    parser.add_argument("--similarity_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--similarity_dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--summary_percentiles", default=DEFAULT_PERCENTILES)
    parser.add_argument("--summary_sample_size", type=int, default=1_000_000)
    parser.add_argument("--sample_seed", type=int, default=1234)
    parser.add_argument("--make_plots", type=str2bool, default=True)
    parser.add_argument("--plot_max_tokens", type=int, default=5000)
    parser.add_argument("--figure_size", type=float, default=7.5)
    parser.add_argument("--plot_dpi", type=int, default=180)
    parser.add_argument("--cmap", default="coolwarm")
    parser.add_argument("--vmin", type=float, default=-1.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--write_token_csv", type=str2bool, default=True)
    parser.add_argument("--analyze_v_cache", type=str2bool, default=True)
    parser.add_argument("--svd_energy_thresholds", default=DEFAULT_ENERGY_THRESHOLDS)
    parser.add_argument("--svd_plot_top_n", type=int, default=128)
    parser.add_argument("--attention_validation", type=str2bool, default=True)
    parser.add_argument("--validation_query_count", type=int, default=128)
    parser.add_argument("--validation_ranks", default=DEFAULT_VALIDATION_RANKS)
    parser.add_argument("--validation_variants", default="raw,centered")
    parser.add_argument("--validation_cache_types", default="k_only,kv")
    parser.add_argument(
        "--strict_query_capture",
        type=str2bool,
        default=False,
        help="Fail if RoPE-aligned query capture is unavailable. Otherwise skip attention validation.",
    )
    return parser.parse_args()


def extract_value_tensors(past_key_values: Any) -> list[torch.Tensor]:
    if hasattr(past_key_values, "value_cache"):
        return list(past_key_values.value_cache)

    if hasattr(past_key_values, "to_legacy_cache"):
        legacy_cache = past_key_values.to_legacy_cache()
        return [layer_cache[1] for layer_cache in legacy_cache]

    if isinstance(past_key_values, (list, tuple)):
        if past_key_values and isinstance(past_key_values[0], (list, tuple)):
            return [layer_cache[1] for layer_cache in past_key_values]

    if hasattr(past_key_values, "layers"):
        value_tensors: list[torch.Tensor] = []
        for layer_cache in past_key_values.layers:
            for attr_name in ("values", "value_cache", "value_states"):
                if hasattr(layer_cache, attr_name):
                    value_tensors.append(getattr(layer_cache, attr_name))
                    break
        if value_tensors:
            return value_tensors

    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def select_position_tensor(tensor: torch.Tensor, local_indices: torch.Tensor) -> torch.Tensor:
    max_index = int(local_indices.max().item()) if local_indices.numel() else -1
    if tensor.ndim == 2 and tensor.shape[0] > max_index:
        return tensor.index_select(0, local_indices.to(tensor.device))
    if tensor.ndim == 3 and tensor.shape[-2] > max_index:
        return tensor.index_select(-2, local_indices.to(tensor.device))
    if tensor.ndim == 4 and tensor.shape[-2] > max_index:
        return tensor.index_select(-2, local_indices.to(tensor.device))
    raise ValueError(f"Cannot select local positions from tensor shape {tuple(tensor.shape)}")


def normalize_rope_tensor(tensor: torch.Tensor, query_states: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim != 4:
        raise ValueError(f"Unsupported RoPE tensor shape: {tuple(tensor.shape)}")

    while tensor.ndim < query_states.ndim:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device=query_states.device, dtype=query_states.dtype)


def apply_rotary_to_query(
    query_states: torch.Tensor,
    position_embeddings: Any,
    local_indices: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(position_embeddings, (tuple, list)) or len(position_embeddings) != 2:
        raise ValueError("position_embeddings must be a (cos, sin) tuple to capture RoPE-aligned queries.")

    cos, sin = position_embeddings
    cos = select_position_tensor(cos, local_indices)
    sin = select_position_tensor(sin, local_indices)
    cos = normalize_rope_tensor(cos, query_states)
    sin = normalize_rope_tensor(sin, query_states)
    return (query_states * cos) + (rotate_half(query_states) * sin)


def infer_num_query_heads(module: torch.nn.Module, projected_width: int) -> int:
    for attr_name in ("num_heads", "num_attention_heads"):
        value = getattr(module, attr_name, None)
        if value is not None:
            return int(value)
    config = getattr(module, "config", None)
    if config is not None:
        value = getattr(config, "num_attention_heads", None)
        if value is not None:
            return int(value)
    head_dim = getattr(module, "head_dim", None)
    if head_dim is not None:
        return projected_width // int(head_dim)
    raise ValueError("Cannot infer number of query heads from attention module.")


class QuerySampleCapture:
    def __init__(
        self,
        attention_modules: list[tuple[int, str, torch.nn.Module]],
        selected_layers: list[int],
        sample_positions: list[int],
    ) -> None:
        self.layer_by_module = {id(module): layer for layer, _, module in attention_modules}
        self.selected_layers = set(selected_layers)
        self.sample_positions = sample_positions
        self.positions_by_layer: dict[int, list[int]] = defaultdict(list)
        self.queries_by_layer: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.handles: list[Any] = []
        self.current_start = 0
        self.current_end = 0
        self.disabled_reason: str | None = None
        self.query_head_counts: dict[int, int] = {}

    def set_chunk(self, start: int, end: int) -> None:
        self.current_start = start
        self.current_end = end

    def install(self, attention_modules: list[tuple[int, str, torch.nn.Module]]) -> None:
        for _, _, module in attention_modules:
            handle = module.register_forward_pre_hook(self._hook, with_kwargs=True)
            self.handles.append(handle)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _local_indices_in_chunk(self) -> tuple[list[int], torch.Tensor]:
        positions = [
            position
            for position in self.sample_positions
            if self.current_start <= position < self.current_end
        ]
        local = [position - self.current_start for position in positions]
        return positions, torch.tensor(local, dtype=torch.long)

    def _hook(self, module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if self.disabled_reason is not None:
            return
        layer_idx = self.layer_by_module.get(id(module))
        if layer_idx is None or layer_idx not in self.selected_layers:
            return

        global_positions, local_indices_cpu = self._local_indices_in_chunk()
        if not global_positions:
            return

        try:
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            if hidden_states is None:
                raise ValueError("attention hidden_states were not available in the forward pre-hook.")

            local_indices = local_indices_cpu.to(hidden_states.device)
            selected_hidden = hidden_states.index_select(1, local_indices)
            projected = module.q_proj(selected_hidden)
            num_query_heads = infer_num_query_heads(module, int(projected.shape[-1]))
            head_dim = int(projected.shape[-1]) // num_query_heads
            query_states = projected.view(
                projected.shape[0],
                projected.shape[1],
                num_query_heads,
                head_dim,
            ).transpose(1, 2)

            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(args) > 1:
                for item in args[1:]:
                    if isinstance(item, (tuple, list)) and len(item) == 2:
                        position_embeddings = item
                        break
            query_states = apply_rotary_to_query(query_states, position_embeddings, local_indices_cpu)
            query_states = query_states[0].transpose(0, 1).detach().float().cpu()

            self.query_head_counts[layer_idx] = int(query_states.shape[1])
            for idx, position in enumerate(global_positions):
                self.positions_by_layer[layer_idx].append(position)
                self.queries_by_layer[layer_idx].append(query_states[idx])
        except Exception as exc:
            self.disabled_reason = f"query capture failed at layer {layer_idx}: {exc}"

    def get_layer_samples(self, layer_idx: int) -> tuple[list[int], torch.Tensor | None]:
        positions = self.positions_by_layer.get(layer_idx, [])
        queries = self.queries_by_layer.get(layer_idx, [])
        if not positions or not queries:
            return [], None
        order = sorted(range(len(positions)), key=lambda idx: positions[idx])
        sorted_positions = [positions[idx] for idx in order]
        sorted_queries = torch.stack([queries[idx] for idx in order], dim=0)
        return sorted_positions, sorted_queries


def sample_query_positions(total_tokens: int, count: int, seed: int) -> list[int]:
    if count <= 0:
        return []
    candidates = torch.arange(1, max(total_tokens, 2), dtype=torch.long)
    if candidates.numel() <= count:
        return candidates.tolist()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(int(candidates.numel()), generator=generator)[:count]
    return sorted(candidates[indices].tolist())


@torch.inference_mode()
def build_kv_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    chunk_size: int,
    input_device: torch.device,
    query_capture: QuerySampleCapture | None,
) -> tuple[Any, list[dict[str, Any]]]:
    total_tokens = int(input_ids.shape[1])
    past_key_values = None
    timing_rows: list[dict[str, Any]] = []
    total_chunks = math.ceil(total_tokens / chunk_size)

    for chunk_idx, start in enumerate(range(0, total_tokens, chunk_size), start=1):
        end = min(start + chunk_size, total_tokens)
        if query_capture is not None:
            query_capture.set_chunk(start, end)
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

        print(f"profile chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
        started = time.perf_counter()
        outputs = model_forward(model, kwargs)
        seconds = time.perf_counter() - started
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


def centered_matrix(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = matrix.float().mean(dim=0, keepdim=True)
    return matrix.float() - mean, mean


def summarize_raw_and_centered_cosine(
    vectors: torch.Tensor,
    layer_idx: int,
    head_idx: int,
    args: argparse.Namespace,
    percentiles: list[float],
    generator: torch.Generator,
    plots_dir: Path,
    prefix: str,
    similarity_device: torch.device,
    similarity_dtype: torch.dtype,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for variant, source_vectors in (
        ("raw", vectors),
        ("centered", centered_matrix(vectors)[0]),
    ):
        started = time.perf_counter()
        matrix = compute_cosine_matrix(source_vectors, similarity_device, similarity_dtype)
        seconds = time.perf_counter() - started
        stats = summarize_similarity_matrix(matrix, percentiles, args.summary_sample_size, generator)
        row[f"{prefix}_{variant}_cosine_seconds"] = seconds
        row[f"{prefix}_{variant}_plot_path"] = None
        row[f"{prefix}_{variant}_plot_stride"] = None
        row[f"{prefix}_{variant}_plotted_tokens"] = None
        for key, value in stats.items():
            row[f"{prefix}_{variant}_{key}"] = value

        if args.make_plots:
            plot_dir = plots_dir / f"{prefix}_{variant}_cosine"
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / f"layer_{layer_idx:02d}_head_{head_idx:02d}_{prefix}_{variant}_cosine.png"
            plot_info = save_similarity_heatmap(matrix, plot_path, layer_idx, head_idx, args)
            row[f"{prefix}_{variant}_plot_path"] = plot_info["plot_path"]
            row[f"{prefix}_{variant}_plot_stride"] = plot_info["plot_stride"]
            row[f"{prefix}_{variant}_plotted_tokens"] = plot_info["plotted_tokens"]
        del matrix
    return row


def cosine_summary_fields(prefix: str, variant: str, percentiles: list[float]) -> list[str]:
    return [
        f"{prefix}_{variant}_cosine_seconds",
        f"{prefix}_{variant}_plot_path",
        f"{prefix}_{variant}_plot_stride",
        f"{prefix}_{variant}_plotted_tokens",
        f"{prefix}_{variant}_diag_mean",
        f"{prefix}_{variant}_diag_min",
        f"{prefix}_{variant}_diag_max",
    ] + [
        f"{prefix}_{variant}_{field}"
        for field in stat_fieldnames("all", percentiles) + stat_fieldnames("offdiag", percentiles)
    ]


def compression_summary_fields(percentiles: list[float]) -> list[str]:
    fields = ["layer", "head", "tokens", "head_dim", "query_samples_captured", "query_heads"]
    for cache_type in ("k", "v"):
        for variant in ("raw", "centered"):
            fields.extend(cosine_summary_fields(cache_type, variant, percentiles))
    return fields


def svd_decompose(matrix: torch.Tensor, centered: bool) -> dict[str, torch.Tensor]:
    source = matrix.float().cpu()
    if centered:
        residual, mean = centered_matrix(source)
        source = residual
    else:
        mean = torch.zeros((1, source.shape[1]), dtype=torch.float32)

    u, s, vh = torch.linalg.svd(source, full_matrices=False)
    energy = s.square()
    total_energy = energy.sum().clamp_min(1e-30)
    explained = energy / total_energy
    cumulative = explained.cumsum(dim=0)
    return {
        "mean": mean,
        "u": u,
        "s": s,
        "vh": vh,
        "explained_energy": explained,
        "cumulative_energy": cumulative,
        "total_energy": total_energy,
    }


def reconstruct_from_svd(decomp: dict[str, torch.Tensor], rank: int, centered: bool) -> torch.Tensor:
    s = decomp["s"]
    rank = min(rank, int(s.numel()))
    if rank <= 0:
        approx = torch.zeros((decomp["u"].shape[0], decomp["vh"].shape[1]), dtype=torch.float32)
    else:
        approx = (decomp["u"][:, :rank] * s[:rank].view(1, -1)) @ decomp["vh"][:rank, :]
    if centered:
        approx = approx + decomp["mean"]
    return approx.float()


def rank_for_energy(cumulative: torch.Tensor, threshold_percent: float) -> int:
    threshold = threshold_percent / 100.0
    hits = torch.nonzero(cumulative >= threshold, as_tuple=False)
    if hits.numel() == 0:
        return int(cumulative.numel())
    return int(hits[0, 0].item()) + 1


def svd_rows_for_head(
    cache_type: str,
    variant: str,
    layer_idx: int,
    head_idx: int,
    decomp: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    s = decomp["s"]
    explained = decomp["explained_energy"]
    cumulative = decomp["cumulative_energy"]
    for idx in range(int(s.numel())):
        rows.append(
            {
                "cache_type": cache_type,
                "variant": variant,
                "layer": layer_idx,
                "head": head_idx,
                "component": idx + 1,
                "singular_value": float(s[idx]),
                "explained_energy": float(explained[idx]),
                "cumulative_energy": float(cumulative[idx]),
            }
        )
    return rows


def svd_summary_row(
    cache_type: str,
    variant: str,
    layer_idx: int,
    head_idx: int,
    tokens: int,
    head_dim: int,
    decomp: dict[str, torch.Tensor],
    thresholds: list[float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cache_type": cache_type,
        "variant": variant,
        "layer": layer_idx,
        "head": head_idx,
        "tokens": tokens,
        "head_dim": head_dim,
        "rank_max": int(decomp["s"].numel()),
        "total_energy": float(decomp["total_energy"]),
        "top1_energy": float(decomp["cumulative_energy"][0]) if decomp["s"].numel() else 0.0,
        "top4_energy": float(decomp["cumulative_energy"][min(3, decomp["s"].numel() - 1)]),
        "top8_energy": float(decomp["cumulative_energy"][min(7, decomp["s"].numel() - 1)]),
        "top16_energy": float(decomp["cumulative_energy"][min(15, decomp["s"].numel() - 1)]),
        "top32_energy": float(decomp["cumulative_energy"][min(31, decomp["s"].numel() - 1)]),
        "top64_energy": float(decomp["cumulative_energy"][min(63, decomp["s"].numel() - 1)]),
    }
    for threshold in thresholds:
        field = str(int(threshold)) if float(threshold).is_integer() else str(threshold).replace(".", "_")
        row[f"rank_for_{field}_energy"] = rank_for_energy(decomp["cumulative_energy"], threshold)
    return row


def save_svd_plot(
    decomp: dict[str, torch.Tensor],
    path: Path,
    title: str,
    top_n: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = decomp["s"].float()
    cumulative = decomp["cumulative_energy"].float()
    count = int(s.numel()) if top_n <= 0 else min(top_n, int(s.numel()))
    x = torch.arange(1, count + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=180)
    ax1.plot(x.tolist(), s[:count].tolist(), marker=".", linewidth=1.2)
    ax1.set_title(title)
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Singular value")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x.tolist(), cumulative[:count].tolist(), color="#B45309", linewidth=1.2)
    ax2.set_ylabel("Cumulative energy")
    ax2.set_ylim(0.0, 1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


class ScalarAccumulator:
    def __init__(self) -> None:
        self.values: list[float] = []

    def update(self, value: float) -> None:
        if math.isfinite(value):
            self.values.append(float(value))

    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    def max(self) -> float:
        return max(self.values) if self.values else 0.0


def grouped_query_heads(kv_head: int, query_heads: int, kv_heads: int) -> list[int]:
    if query_heads < kv_heads or query_heads % kv_heads != 0:
        return [kv_head] if kv_head < query_heads else []
    group_size = query_heads // kv_heads
    start = kv_head * group_size
    return list(range(start, start + group_size))


def validate_attention_approximation(
    positions: list[int],
    query_samples: torch.Tensor,
    kv_head: int,
    kv_heads: int,
    k: torch.Tensor,
    v: torch.Tensor,
    k_hat: torch.Tensor,
    v_hat: torch.Tensor,
    cache_type: str,
    eps: float = 1e-12,
) -> dict[str, Any]:
    query_count, query_heads, head_dim = query_samples.shape
    selected_query_heads = grouped_query_heads(kv_head, query_heads, kv_heads)
    if not selected_query_heads:
        return {
            "query_count": 0,
            "query_head_count": 0,
            "q_dot_abs_error_mean": 0.0,
            "q_dot_abs_error_max": 0.0,
            "attention_weighted_q_dot_abs_error_mean": 0.0,
            "attention_kl_mean": 0.0,
            "attention_kl_max": 0.0,
            "top1_match_fraction": 0.0,
            "output_l2_error_mean": 0.0,
            "output_relative_l2_error_mean": 0.0,
            "output_relative_l2_error_max": 0.0,
        }

    scale = 1.0 / math.sqrt(float(head_dim))
    q_dot_error = ScalarAccumulator()
    weighted_q_dot_error = ScalarAccumulator()
    kl = ScalarAccumulator()
    output_l2 = ScalarAccumulator()
    output_relative_l2 = ScalarAccumulator()
    top1_matches = 0
    comparisons = 0

    k = k.float()
    v = v.float()
    k_hat = k_hat.float()
    v_hat = v_hat.float()

    for sample_idx, position in enumerate(positions):
        valid_count = min(int(position) + 1, int(k.shape[0]))
        if valid_count <= 0:
            continue
        k_valid = k[:valid_count]
        v_valid = v[:valid_count]
        k_hat_valid = k_hat[:valid_count]
        v_hat_valid = v_hat[:valid_count]
        for query_head in selected_query_heads:
            q = query_samples[sample_idx, query_head].float()
            original_q_dot = torch.mv(k_valid, q)
            approx_q_dot = torch.mv(k_hat_valid, q)
            original_logits = original_q_dot * scale
            approx_logits = approx_q_dot * scale
            original_attention = F.softmax(original_logits, dim=-1)
            approx_attention = F.softmax(approx_logits, dim=-1)

            abs_q_dot = (approx_q_dot - original_q_dot).abs()
            q_dot_error.update(float(abs_q_dot.mean()))
            weighted_q_dot_error.update(float((original_attention * abs_q_dot).sum()))
            kl_value = (
                original_attention
                * ((original_attention + eps).log() - (approx_attention + eps).log())
            ).sum()
            kl.update(float(kl_value))
            if int(original_attention.argmax()) == int(approx_attention.argmax()):
                top1_matches += 1
            comparisons += 1

            if cache_type == "k_only":
                original_output = original_attention @ v_valid
                approx_output = approx_attention @ v_valid
            else:
                original_output = original_attention @ v_valid
                approx_output = approx_attention @ v_hat_valid
            l2 = torch.linalg.vector_norm(approx_output - original_output, ord=2)
            rel = l2 / torch.linalg.vector_norm(original_output, ord=2).clamp_min(eps)
            output_l2.update(float(l2))
            output_relative_l2.update(float(rel))

    return {
        "query_count": query_count,
        "query_head_count": len(selected_query_heads),
        "q_dot_abs_error_mean": q_dot_error.mean(),
        "q_dot_abs_error_max": q_dot_error.max(),
        "attention_weighted_q_dot_abs_error_mean": weighted_q_dot_error.mean(),
        "attention_kl_mean": kl.mean(),
        "attention_kl_max": kl.max(),
        "top1_match_fraction": top1_matches / comparisons if comparisons else 0.0,
        "output_l2_error_mean": output_l2.mean(),
        "output_relative_l2_error_mean": output_relative_l2.mean(),
        "output_relative_l2_error_max": output_relative_l2.max(),
    }


def validation_rows_for_head(
    layer_idx: int,
    head_idx: int,
    kv_heads: int,
    positions: list[int],
    query_samples: torch.Tensor | None,
    k: torch.Tensor,
    v: torch.Tensor,
    decomps: dict[tuple[str, str], dict[str, torch.Tensor]],
    ranks: list[int],
    variants: list[str],
    cache_types: list[str],
) -> list[dict[str, Any]]:
    if query_samples is None or not positions:
        return []

    rows: list[dict[str, Any]] = []
    for variant in variants:
        centered = variant == "centered"
        k_decomp = decomps[("k", variant)]
        v_decomp = decomps[("v", variant)]
        for rank in ranks:
            rank = min(rank, int(k_decomp["s"].numel()), int(v_decomp["s"].numel()))
            k_hat = reconstruct_from_svd(k_decomp, rank, centered)
            v_hat = reconstruct_from_svd(v_decomp, rank, centered)
            for cache_type in cache_types:
                stats = validate_attention_approximation(
                    positions,
                    query_samples,
                    head_idx,
                    kv_heads,
                    k,
                    v,
                    k_hat,
                    v_hat,
                    cache_type,
                )
                row: dict[str, Any] = {
                    "layer": layer_idx,
                    "head": head_idx,
                    "variant": variant,
                    "cache_type": cache_type,
                    "rank": rank,
                }
                row.update(stats)
                row["loss_delta"] = ""
                row["ppl_ratio"] = ""
                row["loss_ppl_note"] = (
                    "not computed; exact loss/PPL requires running the model with compressed KV injected "
                    "inside every attention layer during prefill/decode"
                )
                rows.append(row)
            del k_hat, v_hat
    return rows


def write_rows_append(path: Path, rows: list[dict[str, Any]], fields: list[str], append: bool) -> None:
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not append:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    percentiles = parse_percentiles(args.summary_percentiles)
    energy_thresholds = parse_energy_thresholds(args.svd_energy_thresholds)
    validation_ranks = parse_int_list(args.validation_ranks)
    validation_variants = [item.strip() for item in args.validation_variants.split(",") if item.strip()]
    validation_cache_types = [item.strip() for item in args.validation_cache_types.split(",") if item.strip()]
    for variant in validation_variants:
        if variant not in {"raw", "centered"}:
            raise ValueError(f"Unsupported validation variant: {variant}")
    for cache_type in validation_cache_types:
        if cache_type not in {"k_only", "kv"}:
            raise ValueError(f"Unsupported validation cache type: {cache_type}")

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
    if args.require_max_tokens and len(token_ids) < args.max_tokens:
        raise ValueError(
            f"Tokenization produced {len(token_ids)} tokens, fewer than --max_tokens {args.max_tokens}."
        )
    token_ids = token_ids[: args.max_tokens]
    if len(token_ids) < 2:
        raise ValueError("Tokenization produced fewer than two tokens.")
    input_ids = torch.tensor(token_ids, dtype=torch.long).view(1, -1)
    print(f"using tokens: {input_ids.shape[1]} (max_tokens={args.max_tokens})", flush=True)
    if args.write_token_csv:
        write_tokens_csv(output_dir / "tokens.csv", tokenizer, token_ids)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_dtype(args.dtype, requested_device)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": model_dtype,
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

    attention_modules = find_attention_modules(model)
    num_layers_for_selection = len(attention_modules) if attention_modules else int(getattr(model.config, "num_hidden_layers", 0))
    if num_layers_for_selection <= 0:
        raise RuntimeError("Could not infer the model layer count.")
    layer_indices = parse_index_spec(args.layers, num_layers_for_selection, "layers")

    query_capture: QuerySampleCapture | None = None
    query_capture_status: dict[str, Any] = {"enabled": False, "reason": None}
    if args.attention_validation and args.validation_query_count > 0:
        if not attention_modules:
            query_capture_status["reason"] = "no attention modules with q_proj/k_proj/v_proj/o_proj were found"
        else:
            sample_positions = sample_query_positions(
                int(input_ids.shape[1]),
                args.validation_query_count,
                args.sample_seed,
            )
            query_capture = QuerySampleCapture(attention_modules, layer_indices, sample_positions)
            query_capture.install(attention_modules)
            query_capture_status = {
                "enabled": True,
                "sample_positions": sample_positions,
                "reason": None,
            }

    input_device = pick_input_device(model, requested_device)
    try:
        past_key_values, timing_rows = build_kv_cache(
            model,
            input_ids,
            args.chunk_size,
            input_device,
            query_capture,
        )
    finally:
        if query_capture is not None:
            query_capture.close()

    if query_capture is not None and query_capture.disabled_reason is not None:
        query_capture_status["enabled"] = False
        query_capture_status["reason"] = query_capture.disabled_reason
        if args.strict_query_capture:
            raise RuntimeError(query_capture.disabled_reason)
        print(f"attention validation disabled: {query_capture.disabled_reason}", flush=True)

    write_profile_timings(output_dir / "profile_timings.csv", timing_rows)

    key_tensors = extract_key_tensors(past_key_values)
    value_tensors = extract_value_tensors(past_key_values)
    if len(key_tensors) != len(value_tensors):
        raise RuntimeError(f"K/V layer count mismatch: {len(key_tensors)} keys vs {len(value_tensors)} values")

    layer_indices = parse_index_spec(args.layers, len(key_tensors), "layers")
    expected_heads = getattr(model.config, "num_key_value_heads", None)
    similarity_device = resolve_similarity_device(args.similarity_device)
    similarity_dtype = named_torch_dtype(args.similarity_dtype)
    if similarity_device.type == "cpu" and similarity_dtype != torch.float32:
        print("CPU similarity matmul uses float32 regardless of --similarity_dtype.", flush=True)
        similarity_dtype = torch.float32

    plots_dir = output_dir / "plots"
    svd_plot_dir = plots_dir / "svd"
    if args.make_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        svd_plot_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    svd_summary_rows: list[dict[str, Any]] = []
    singular_value_path = output_dir / "singular_values.csv"
    validation_path = output_dir / "attention_validation_by_head_rank.csv"

    singular_fields = [
        "cache_type",
        "variant",
        "layer",
        "head",
        "component",
        "singular_value",
        "explained_energy",
        "cumulative_energy",
    ]
    validation_fields = [
        "layer",
        "head",
        "variant",
        "cache_type",
        "rank",
        "query_count",
        "query_head_count",
        "q_dot_abs_error_mean",
        "q_dot_abs_error_max",
        "attention_weighted_q_dot_abs_error_mean",
        "attention_kl_mean",
        "attention_kl_max",
        "top1_match_fraction",
        "output_l2_error_mean",
        "output_relative_l2_error_mean",
        "output_relative_l2_error_max",
        "loss_delta",
        "ppl_ratio",
        "loss_ppl_note",
    ]
    write_rows_append(singular_value_path, [], singular_fields, append=False)
    write_rows_append(validation_path, [], validation_fields, append=False)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.sample_seed)
    cache_shapes: list[dict[str, Any]] = []
    wrote_singular_rows = True
    wrote_validation_rows = True

    for layer_idx in layer_indices:
        key_by_head = key_tensor_to_head_token_dim(key_tensors[layer_idx], expected_heads)
        value_by_head = key_tensor_to_head_token_dim(value_tensors[layer_idx], expected_heads)
        kv_heads, tokens, head_dim = key_by_head.shape
        if value_by_head.shape != key_by_head.shape:
            raise RuntimeError(
                f"K/V shape mismatch at layer {layer_idx}: {tuple(key_by_head.shape)} vs {tuple(value_by_head.shape)}"
            )
        head_indices = parse_index_spec(args.heads, int(kv_heads), "heads")
        positions, query_samples = ([], None)
        if query_capture is not None and query_capture.disabled_reason is None:
            positions, query_samples = query_capture.get_layer_samples(layer_idx)

        cache_shapes.append(
            {
                "layer": layer_idx,
                "kv_heads": int(kv_heads),
                "tokens": int(tokens),
                "head_dim": int(head_dim),
                "key_raw_shape": list(key_tensors[layer_idx].shape),
                "value_raw_shape": list(value_tensors[layer_idx].shape),
                "query_samples": len(positions),
                "query_heads": int(query_samples.shape[1]) if query_samples is not None else 0,
            }
        )
        print(
            f"layer {layer_idx}: kv_heads={kv_heads} tokens={tokens} head_dim={head_dim} "
            f"query_samples={len(positions)}",
            flush=True,
        )

        for head_idx in head_indices:
            print(f"diagnostics for layer {layer_idx}, head {head_idx}", flush=True)
            row: dict[str, Any] = {
                "layer": layer_idx,
                "head": head_idx,
                "tokens": int(tokens),
                "head_dim": int(head_dim),
                "query_samples_captured": len(positions),
                "query_heads": int(query_samples.shape[1]) if query_samples is not None else 0,
            }

            k = key_by_head[head_idx].float()
            v = value_by_head[head_idx].float()
            row.update(
                summarize_raw_and_centered_cosine(
                    k,
                    layer_idx,
                    head_idx,
                    args,
                    percentiles,
                    generator,
                    plots_dir,
                    "k",
                    similarity_device,
                    similarity_dtype,
                )
            )
            if args.analyze_v_cache:
                row.update(
                    summarize_raw_and_centered_cosine(
                        v,
                        layer_idx,
                        head_idx,
                        args,
                        percentiles,
                        generator,
                        plots_dir,
                        "v",
                        similarity_device,
                        similarity_dtype,
                    )
                )

            decomps: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
            for cache_type, matrix in (("k", k), ("v", v)):
                if cache_type == "v" and not args.analyze_v_cache:
                    continue
                for variant in ("raw", "centered"):
                    centered = variant == "centered"
                    decomp = svd_decompose(matrix, centered=centered)
                    decomps[(cache_type, variant)] = decomp
                    svd_summary_rows.append(
                        svd_summary_row(
                            cache_type,
                            variant,
                            layer_idx,
                            head_idx,
                            int(tokens),
                            int(head_dim),
                            decomp,
                            energy_thresholds,
                        )
                    )
                    singular_rows = svd_rows_for_head(cache_type, variant, layer_idx, head_idx, decomp)
                    write_rows_append(singular_value_path, singular_rows, singular_fields, append=wrote_singular_rows)
                    wrote_singular_rows = True
                    if args.make_plots:
                        path = svd_plot_dir / (
                            f"{cache_type}_{variant}_layer_{layer_idx:02d}_head_{head_idx:02d}_singular_values.png"
                        )
                        save_svd_plot(
                            decomp,
                            path,
                            f"{cache_type.upper()} {variant} SVD L{layer_idx} H{head_idx}",
                            args.svd_plot_top_n,
                        )

            if args.attention_validation and args.analyze_v_cache and query_samples is not None:
                validation_rows = validation_rows_for_head(
                    layer_idx,
                    head_idx,
                    int(kv_heads),
                    positions,
                    query_samples,
                    k,
                    v,
                    decomps,
                    validation_ranks,
                    validation_variants,
                    validation_cache_types,
                )
                write_rows_append(validation_path, validation_rows, validation_fields, append=wrote_validation_rows)
                wrote_validation_rows = True

            summary_rows.append(row)
            del k, v, decomps
            if similarity_device.type == "cuda":
                torch.cuda.empty_cache()

        del key_by_head, value_by_head

    summary_fields = compression_summary_fields(percentiles)
    write_csv(output_dir / "compression_summary_by_head.csv", summary_rows, summary_fields)

    svd_threshold_fields = []
    for threshold in energy_thresholds:
        field = str(int(threshold)) if float(threshold).is_integer() else str(threshold).replace(".", "_")
        svd_threshold_fields.append(f"rank_for_{field}_energy")
    svd_summary_fields = [
        "cache_type",
        "variant",
        "layer",
        "head",
        "tokens",
        "head_dim",
        "rank_max",
        "total_energy",
        "top1_energy",
        "top4_energy",
        "top8_energy",
        "top16_energy",
        "top32_energy",
        "top64_energy",
    ] + svd_threshold_fields
    write_csv(output_dir / "svd_summary_by_head.csv", svd_summary_rows, svd_summary_fields)

    aggregate_plot_paths: dict[str, str | None] = {}
    if args.make_plots:
        aggregate_plot_paths["k_raw_offdiag_mean_heatmap"] = save_layer_head_metric_heatmap(
            summary_rows,
            plots_dir,
            "k_raw_offdiag_mean",
            "K raw mean off-diagonal cosine",
            args.cmap,
        )
        aggregate_plot_paths["k_centered_offdiag_mean_heatmap"] = save_layer_head_metric_heatmap(
            summary_rows,
            plots_dir,
            "k_centered_offdiag_mean",
            "K centered mean off-diagonal cosine",
            args.cmap,
        )
        if args.analyze_v_cache:
            aggregate_plot_paths["v_raw_offdiag_mean_heatmap"] = save_layer_head_metric_heatmap(
                summary_rows,
                plots_dir,
                "v_raw_offdiag_mean",
                "V raw mean off-diagonal cosine",
                args.cmap,
            )
            aggregate_plot_paths["v_centered_offdiag_mean_heatmap"] = save_layer_head_metric_heatmap(
                summary_rows,
                plots_dir,
                "v_centered_offdiag_mean",
                "V centered mean off-diagonal cosine",
                args.cmap,
            )

    payload = {
        "args": vars(args),
        "resolved": {
            "tokens": int(input_ids.shape[1]),
            "text_path": str(text_path),
            "model_name_or_path": args.model_name_or_path,
            "layers": layer_indices,
            "summary_percentiles": percentiles,
            "svd_energy_thresholds": energy_thresholds,
            "validation_ranks": validation_ranks,
            "similarity_device": str(similarity_device),
            "similarity_dtype": str(similarity_dtype),
            "cache_shapes": cache_shapes,
            "query_capture": query_capture_status,
        },
        "paths": {
            "tokens": str(output_dir / "tokens.csv") if args.write_token_csv else None,
            "profile_timings": str(output_dir / "profile_timings.csv"),
            "compression_summary_by_head": str(output_dir / "compression_summary_by_head.csv"),
            "singular_values": str(singular_value_path),
            "svd_summary_by_head": str(output_dir / "svd_summary_by_head.csv"),
            "attention_validation_by_head_rank": str(validation_path) if args.attention_validation else None,
            "plots_dir": str(plots_dir) if args.make_plots else None,
            "aggregate_plots": aggregate_plot_paths,
        },
        "notes": {
            "loss_ppl": (
                "Exact loss/PPL change is not computed here because it requires injecting compressed K/V "
                "inside every attention layer during the model forward. This script writes sampled "
                "attention-weighted logit, KL, and output-vector errors as a safer pre-screening step."
            )
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
