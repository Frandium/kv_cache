from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)\.pt$")


@dataclass(frozen=True)
class RunSpec:
    size: str
    method: str
    run_name: str
    directory: Path


def default_run_specs() -> list[RunSpec]:
    l_root = Path(os.environ.get("L_OUTPUT_ROOT", "/mnt/workspace/fmoe_cuda_2b_8e_outputs"))
    m_root = Path(os.environ.get("M_OUTPUT_ROOT", "/mnt/workspace/fmoe_cuda_m_8e_outputs"))
    return [
        RunSpec("L", "baseline", "baseline_lb", l_root / "baseline_lb"),
        RunSpec("L", "proposed", "proposed_lb", l_root / "proposed_lb"),
        RunSpec("M", "baseline", "baseline", m_root / "baseline"),
        RunSpec("M", "proposed", "proposed", m_root / "proposed"),
    ]


def checkpoint_steps(directory: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in directory.glob("checkpoint-*.pt"):
        match = CHECKPOINT_RE.search(path.name)
        if match:
            found[int(match.group(1))] = path.resolve()
    return found


def selected_runs() -> list[dict[str, Any]]:
    specs = default_run_specs()
    selected: list[dict[str, Any]] = []
    for size in ("L", "M"):
        pair = [spec for spec in specs if spec.size == size]
        if len(pair) != 2:
            raise RuntimeError(f"expected two runs for size {size}")
        indexed = {spec.method: checkpoint_steps(spec.directory) for spec in pair}
        common = sorted(set(indexed["baseline"]) & set(indexed["proposed"]))
        if not common:
            raise FileNotFoundError(
                f"no common checkpoint for {size}: "
                f"{pair[0].directory} and {pair[1].directory}"
            )
        step = common[-1]
        for spec in pair:
            selected.append(
                {
                    "size": spec.size,
                    "method": spec.method,
                    "run_name": spec.run_name,
                    "run_dir": str(spec.directory.resolve()),
                    "step": step,
                    "checkpoint": str(indexed[spec.method][step]),
                }
            )
    return selected


def forward_flops_per_token(config: dict[str, Any], train_args: dict[str, Any]) -> dict[str, float]:
    hidden = int(config["hidden_size"])
    layers = int(config["num_hidden_layers"])
    q_dim = int(config["num_attention_heads"]) * int(config["head_dim"])
    kv_dim = int(config["num_key_value_heads"]) * int(config["head_dim"])
    sequence = int(train_args["sequence_length"])
    common = int(config["common_intermediate_size"])
    tail = int(config["tail_intermediate_size"])
    experts = int(config["num_tail_experts"])
    vocab = int(config["vocab_size"])
    attention_projection = 2.0 * hidden * (2 * q_dim + 2 * kv_dim)
    attention_core = 2.0 * q_dim * (sequence + 1)
    active_ffn = 6.0 * hidden * (common + tail)
    router = 2.0 * hidden * experts
    layer_total = attention_projection + attention_core + active_ffn + router
    lm_head = 2.0 * hidden * vocab
    return {
        "attention_projection_flops_per_token": layers * attention_projection,
        "attention_core_flops_per_token": layers * attention_core,
        "active_ffn_flops_per_token": layers * active_ffn,
        "router_flops_per_token": layers * router,
        "lm_head_flops_per_token": lm_head,
        "forward_flops_per_token": layers * layer_total + lm_head,
    }


def training_flops(metadata: dict[str, Any], step: int) -> dict[str, float | int]:
    config = metadata["model_config"]
    train_args = metadata["train_args"]
    flops = forward_flops_per_token(config, train_args)
    tokens_per_step = (
        int(train_args["batch_size"])
        * int(metadata["world_size"])
        * int(train_args["gradient_accumulation"])
        * int(train_args["sequence_length"])
    )
    tokens = int(step) * tokens_per_step
    total = 3.0 * flops["forward_flops_per_token"] * tokens
    return {
        "tokens_per_step": tokens_per_step,
        "training_tokens": tokens,
        "backward_multiplier": 3.0,
        **flops,
        "training_flops": total,
        "log10_training_flops": __import__("math").log10(total),
    }


def parse_flop_targets(value: str | None = None) -> list[float]:
    raw = value or os.environ.get(
        "SCALING_FLOPS_TARGETS", "1.25e19,2.5e19,5e19,1e20,2e20,4e20"
    )
    targets = sorted({float(item.strip()) for item in raw.split(",") if item.strip()})
    if not targets or any(target <= 0 for target in targets):
        raise ValueError("SCALING_FLOPS_TARGETS must contain positive numbers")
    return targets


def selected_scaling_runs(
    latest_runs: list[dict[str, Any]], targets: list[float]
) -> list[dict[str, Any]]:
    specs = default_run_specs()
    latest_by_key = {(row["size"], row["method"]): row for row in latest_runs}
    selected: list[dict[str, Any]] = []
    for size in ("M", "L"):
        pair = [spec for spec in specs if spec.size == size]
        indexed = {spec.method: checkpoint_steps(spec.directory) for spec in pair}
        common_steps = sorted(set(indexed["baseline"]) & set(indexed["proposed"]))
        baseline_metadata = latest_by_key[(size, "baseline")]["metadata"]
        step_flops = {
            step: float(training_flops(baseline_metadata, step)["training_flops"])
            for step in common_steps
        }
        maximum = step_flops[common_steps[-1]]
        chosen: dict[int, list[float]] = {}
        for target in targets:
            if target > maximum:
                continue
            step = min(common_steps, key=lambda candidate: (abs(step_flops[candidate] - target), candidate))
            chosen.setdefault(step, []).append(target)
        chosen.setdefault(common_steps[-1], []).append(maximum)
        for step in sorted(chosen):
            for spec in pair:
                selected.append(
                    {
                        "size": size,
                        "method": spec.method,
                        "run_name": spec.run_name,
                        "run_dir": str(spec.directory.resolve()),
                        "step": step,
                        "checkpoint": str(indexed[spec.method][step]),
                        "target_flops": min(chosen[step], key=lambda target: abs(target - step_flops[step])),
                        "is_latest": int(step == common_steps[-1]),
                        "metadata": latest_by_key[(size, spec.method)]["metadata"],
                    }
                )
    return selected


def job_fingerprint(checkpoint: str | Path, protocol: str, config: str) -> str:
    path = Path(checkpoint)
    stat = path.stat()
    payload = f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}\0{protocol}\0{config}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not value:
        raise ValueError("manifest must be a non-empty JSON list")
    return value


def find_run(manifest: Iterable[dict[str, Any]], size: str, method: str) -> dict[str, Any]:
    matches = [row for row in manifest if row["size"] == size and row["method"] == method]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {size}/{method}, got {len(matches)}")
    return matches[0]


def checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    result = {
        "step": int(payload["step"]),
        "world_size": int(payload.get("world_size", 1)),
        "model_config": dict(payload["model_config"]),
        "train_args": dict(payload.get("train_args", {})),
    }
    del payload
    return result


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError(f"cannot infer CSV fields for empty output: {output}")
    fields = fieldnames or list(rows[0])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
