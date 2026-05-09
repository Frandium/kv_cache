from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_PATH = "/mnt/workspace/Qwen3-0.6B"
DEFAULT_TEXT_PATH = (
    "/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt"
)
DEFAULT_PERCENTILES = "1,5,10,20,30,50,70,80,90,95,99"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a long Qwen3 K-cache and summarize per-layer/head key-vector norms."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/kcache_norms")
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--chunk_size", type=int, default=1024)
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
        default="auto",
        help='Transformer attention backend. Use "auto" to keep the model default.',
    )
    parser.add_argument("--percentiles", default=DEFAULT_PERCENTILES)
    parser.add_argument("--histogram_bins", type=int, default=100)
    parser.add_argument(
        "--histogram_max",
        type=float,
        default=0.0,
        help="Use 0 to set the histogram range from 0 to the observed global max norm.",
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


def build_kv_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    chunk_size: int,
    input_device: torch.device,
) -> Any:
    if chunk_size <= 0:
        raise ValueError("--chunk_size must be positive.")

    past_key_values = None
    total_tokens = input_ids.shape[1]
    model.eval()

    with torch.inference_mode():
        for start in range(0, total_tokens, chunk_size):
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

            outputs = model_forward(model, kwargs)
            past_key_values = outputs.past_key_values
            if past_key_values is None:
                raise RuntimeError("Model did not return past_key_values. Check model/config use_cache support.")
            del outputs
            print(f"cached tokens: {end}/{total_tokens}", flush=True)

    return past_key_values


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    percentiles = parse_percentiles(args.percentiles)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    token_ids = token_ids[: args.max_tokens]
    if not token_ids:
        raise ValueError("Tokenization produced no tokens.")
    input_ids = torch.tensor(token_ids, dtype=torch.long).view(1, -1)
    print(f"using tokens: {input_ids.shape[1]}", flush=True)

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

    print(f"loading model: {args.model_name_or_path}", flush=True)
    model = AutoModel.from_pretrained(args.model_name_or_path, **load_kwargs)
    if args.device_map.lower() == "none":
        model = model.to(requested_device)
    model.config.use_cache = True

    input_device = pick_input_device(model, requested_device)
    past_key_values = build_kv_cache(model, input_ids, args.chunk_size, input_device)
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
            f"layer {layer_idx}: heads={norms.shape[0]} tokens={norms.shape[1]}",
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
            "tokens": int(input_ids.shape[1]),
            "text_path": str(text_path),
            "model_name_or_path": args.model_name_or_path,
            "percentiles": percentiles,
            "histogram_max": hist_max,
            "cache_shapes": cache_shapes,
        },
        "global": global_row,
        "by_layer": layer_rows,
        "by_head": head_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.save_norm_tensors:
        torch.save(
            {
                "metadata": summary_payload["resolved"],
                "layer_norms": layer_norms,
            },
            output_dir / "layer_head_norms.pt",
        )

    print(f"wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
