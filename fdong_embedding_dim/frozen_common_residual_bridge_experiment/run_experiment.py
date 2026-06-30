#!/usr/bin/env python3
"""Two-stage frozen-common residual-bridge experiment.

Stage 1 trains a tied-embedding, one-layer attention model on four equal-weight
shared-K cycles.  Once the top hidden-side singular direction of the tied
embedding is stable, stage 2 clones the exact checkpoint into continuation and
adapter controls.  The constrained adapter has forward map P_R A P_C, so it can
read the frozen common direction but can only write to its orthogonal complement.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-frozen-common-bridge")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


VARIANTS = (
    "baseline_continue",
    "frozen_no_bridge",
    "unconstrained_bridge",
    "common_to_residual",
    "residual_to_residual",
)


@dataclass(frozen=True)
class Config:
    dims: Tuple[int, ...] = (2, 3)
    seeds: Tuple[int, ...] = (0, 1, 2)
    base_lr: float = 0.03
    bridge_lr: float = 0.05
    max_pretrain_steps: int = 300
    post_steps: int = 1500
    min_pretrain_steps: int = 15
    check_interval: int = 3
    stable_checks: int = 3
    stability_rule: str = "angle_or_sigma"
    angle_tol_deg: float = 0.5
    sigma_rel_tol: float = 0.002
    stable_window: int = 10
    residual_alpha: float = 1.0
    init_scale: float = 0.35
    matrix_init_scale: float = 0.08
    weight_decay: float = 0.0
    device: str = "cpu"
    output_dir: str = "results"


def parse_int_tuple(text: str) -> Tuple[int, ...]:
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def build_data() -> Dict[str, object]:
    token_names = ["K"]
    token_ids: Dict[str, int] = {"K": 0}
    for group in "ABCD":
        for index in range(3):
            name = f"{group}{index}"
            token_ids[name] = len(token_names)
            token_names.append(name)

    c1: List[int] = []
    c2: List[int] = []
    targets: List[int] = []
    groups: List[str] = []
    families: List[str] = []
    names: List[str] = []
    for group in "ABCD":
        g0 = token_ids[f"{group}0"]
        g1 = token_ids[f"{group}1"]
        g2 = token_ids[f"{group}2"]
        k = token_ids["K"]
        patterns = (
            (g0, g1, k, "to_K", f"{group}0,{group}1->K"),
            (g1, k, g2, "after_K", f"{group}1,K->{group}2"),
            (k, g2, g0, "K_then_group", f"K,{group}2->{group}0"),
            (g2, g0, g1, "internal", f"{group}2,{group}0->{group}1"),
        )
        for left, right, target, family, name in patterns:
            c1.append(left)
            c2.append(right)
            targets.append(target)
            groups.append(group)
            families.append(family)
            names.append(name)

    return {
        "token_names": token_names,
        "c1": torch.tensor(c1, dtype=torch.long),
        "c2": torch.tensor(c2, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "groups": groups,
        "families": families,
        "pattern_names": names,
    }


class TiedAttentionLM(torch.nn.Module):
    def __init__(self, vocab_size: int, dim: int, cfg: Config, seed: int):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed + 1009 * dim)
        embedding = torch.randn(vocab_size, dim, generator=generator)
        embedding = F.normalize(embedding, dim=-1) * cfg.init_scale
        self.E = torch.nn.Parameter(embedding)
        eye = torch.eye(dim)
        self.Wq = torch.nn.Parameter(
            eye + cfg.matrix_init_scale * torch.randn(dim, dim, generator=generator)
        )
        self.Wk = torch.nn.Parameter(
            eye + cfg.matrix_init_scale * torch.randn(dim, dim, generator=generator)
        )
        self.Wv = torch.nn.Parameter(
            eye + cfg.matrix_init_scale * torch.randn(dim, dim, generator=generator)
        )
        self.Wo = torch.nn.Parameter(
            eye + cfg.matrix_init_scale * torch.randn(dim, dim, generator=generator)
        )
        self.dim = dim
        self.residual_alpha = cfg.residual_alpha

    def base_hidden(
        self, c1: torch.Tensor, c2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h1 = self.E[c1]
        h2 = self.E[c2]
        q = h2 @ self.Wq.T
        k1 = h1 @ self.Wk.T
        k2 = h2 @ self.Wk.T
        v1 = h1 @ self.Wv.T
        v2 = h2 @ self.Wv.T
        scores = torch.stack(((q * k1).sum(-1), (q * k2).sum(-1)), dim=-1)
        attention = F.softmax(scores / math.sqrt(self.dim), dim=-1)
        pooled_v = attention[:, :1] * v1 + attention[:, 1:] * v2
        pooled_embedding = attention[:, :1] * h1 + attention[:, 1:] * h2
        attention_out = pooled_v @ self.Wo.T
        hidden = attention_out + self.residual_alpha * h2
        return hidden, {
            "attention": attention,
            "pooled_v": pooled_v,
            "pooled_embedding": pooled_embedding,
        }

    def forward(
        self, c1: torch.Tensor, c2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hidden, cache = self.base_hidden(c1, c2)
        logits = hidden @ self.E.T
        cache["base_hidden"] = hidden
        cache["final_hidden"] = hidden
        return logits, cache


class AdaptedModel(torch.nn.Module):
    def __init__(
        self,
        base: TiedAttentionLM,
        common_input_direction: torch.Tensor,
        common_output_direction: torch.Tensor,
        variant: str,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.base = base
        self.variant = variant
        input_direction = F.normalize(common_input_direction.detach().clone(), dim=0)
        output_direction = F.normalize(common_output_direction.detach().clone(), dim=0)
        p_common_input = torch.outer(input_direction, input_direction)
        p_common_output = torch.outer(output_direction, output_direction)
        identity = torch.eye(input_direction.numel(), dtype=input_direction.dtype)
        self.register_buffer("common_input_direction", input_direction)
        self.register_buffer("common_output_direction", output_direction)
        self.register_buffer("p_common_input", p_common_input)
        self.register_buffer("p_residual_input", identity - p_common_input)
        self.register_buffer("p_common_output", p_common_output)
        self.register_buffer("p_residual_output", identity - p_common_output)
        if variant in {
            "unconstrained_bridge",
            "common_to_residual",
            "residual_to_residual",
        }:
            self.A = torch.nn.Parameter(
                torch.zeros(input_direction.numel(), input_direction.numel())
            )
        else:
            self.register_parameter("A", None)

    def adapter_output(self, pooled_embedding: torch.Tensor) -> torch.Tensor:
        if self.A is None:
            return torch.zeros_like(pooled_embedding)
        if self.variant == "unconstrained_bridge":
            return pooled_embedding @ self.A.T
        if self.variant == "common_to_residual":
            common_input = pooled_embedding @ self.p_common_input
            return (common_input @ self.A.T) @ self.p_residual_output
        if self.variant == "residual_to_residual":
            residual_input = pooled_embedding @ self.p_residual_input
            return (residual_input @ self.A.T) @ self.p_residual_output
        raise RuntimeError(f"adapter_output called for {self.variant}")

    def forward(
        self, c1: torch.Tensor, c2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        base_hidden, cache = self.base.base_hidden(c1, c2)
        adapter_output = self.adapter_output(cache["pooled_embedding"])
        final_hidden = base_hidden + adapter_output
        logits = final_hidden @ self.base.E.T
        cache.update(
            {
                "base_hidden": base_hidden,
                "adapter_output": adapter_output,
                "final_hidden": final_hidden,
            }
        )
        return logits, cache


def embedding_spectrum(model: TiedAttentionLM) -> Dict[str, object]:
    with torch.no_grad():
        _, singular_values, vh = torch.linalg.svd(model.E.float(), full_matrices=False)
        direction = F.normalize(vh[0], dim=0)
        squared = singular_values.square()
        probs = squared / squared.sum().clamp_min(1e-20)
        effective_rank = torch.exp(-(probs * probs.clamp_min(1e-20).log()).sum())
        top1_energy = probs[0]
    return {
        "direction": direction,
        "sigma1": float(singular_values[0]),
        "sigma2": float(singular_values[1]) if singular_values.numel() > 1 else 0.0,
        "effective_rank": float(effective_rank),
        "top1_energy": float(top1_energy),
    }


def value_map_spectrum(model: TiedAttentionLM) -> Dict[str, object]:
    """SVD of the column-form effective attention value map Wo @ Wv."""
    with torch.no_grad():
        matrix = model.Wo.float() @ model.Wv.float()
        u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        squared = singular_values.square()
        probs = squared / squared.sum().clamp_min(1e-20)
        effective_rank = torch.exp(-(probs * probs.clamp_min(1e-20).log()).sum())
    return {
        "input_direction": F.normalize(vh[0], dim=0),
        "output_direction": F.normalize(u[:, 0], dim=0),
        "sigma1": float(singular_values[0]),
        "sigma2": float(singular_values[1]) if singular_values.numel() > 1 else 0.0,
        "effective_rank": float(effective_rank),
        "top1_energy": float(probs[0]),
    }


def angle_deg(left: torch.Tensor, right: torch.Tensor) -> float:
    cosine = torch.dot(F.normalize(left, dim=0), F.normalize(right, dim=0)).abs()
    cosine = cosine.clamp(0.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)))


def tensor_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def state_relative_drift(model: torch.nn.Module, reference: Dict[str, torch.Tensor]) -> float:
    numerator = 0.0
    denominator = 0.0
    for name, value in model.state_dict().items():
        if name not in reference:
            continue
        current = value.detach().float()
        initial = reference[name].detach().float()
        numerator += float((current - initial).square().sum())
        denominator += float(initial.square().sum())
    return math.sqrt(numerator / max(denominator, 1e-30))


def group_mean(values: torch.Tensor, labels: Sequence[str], label: str) -> float:
    indices = [i for i, item in enumerate(labels) if item == label]
    return float(values[indices].mean()) if indices else float("nan")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: Dict[str, object],
    frozen_input_direction: Optional[torch.Tensor] = None,
    frozen_output_direction: Optional[torch.Tensor] = None,
    frozen_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    logits, cache = model(data["c1"], data["c2"])
    targets = data["targets"]
    losses = F.cross_entropy(logits, targets, reduction="none")
    correct = logits.argmax(dim=-1).eq(targets).float()
    to_k_mask = targets.eq(0)
    non_k_mask = ~to_k_mask

    base = model.base if isinstance(model, AdaptedModel) else model
    spectrum = embedding_spectrum(base)
    value_spectrum = value_map_spectrum(base)
    k_embedding = F.normalize(base.E[0].detach(), dim=0)
    embedding_top_direction = spectrum["direction"]
    metrics: Dict[str, float] = {
        "loss": float(losses.mean()),
        "to_K_loss": float(losses[to_k_mask].mean()),
        "non_K_loss": float(losses[non_k_mask].mean()),
        "full_accuracy": float(correct.mean()),
        "to_K_accuracy": float(correct[to_k_mask].mean()),
        "non_K_accuracy": float(correct[non_k_mask].mean()),
        "embedding_sigma1": float(spectrum["sigma1"]),
        "embedding_sigma2": float(spectrum["sigma2"]),
        "embedding_top1_energy": float(spectrum["top1_energy"]),
        "embedding_effective_rank": float(spectrum["effective_rank"]),
        "embedding_top_direction_K_alignment": float(
            torch.dot(embedding_top_direction, k_embedding).abs()
        ),
        "value_sigma1": float(value_spectrum["sigma1"]),
        "value_sigma2": float(value_spectrum["sigma2"]),
        "value_top1_energy": float(value_spectrum["top1_energy"]),
        "value_effective_rank": float(value_spectrum["effective_rank"]),
        "value_output_direction_K_alignment": float(
            torch.dot(value_spectrum["output_direction"], k_embedding).abs()
        ),
    }
    for group in "ABCD":
        metrics[f"accuracy_{group}"] = group_mean(correct, data["groups"], group)
    for family in ("to_K", "after_K", "K_then_group", "internal"):
        metrics[f"accuracy_{family}"] = group_mean(correct, data["families"], family)

    if frozen_input_direction is None or frozen_output_direction is None:
        metrics["frozen_input_direction_angle_deg"] = float("nan")
        metrics["frozen_output_direction_angle_deg"] = float("nan")
    else:
        metrics["frozen_input_direction_angle_deg"] = angle_deg(
            value_spectrum["input_direction"], frozen_input_direction
        )
        metrics["frozen_output_direction_angle_deg"] = angle_deg(
            value_spectrum["output_direction"], frozen_output_direction
        )

    adapter_output = cache.get("adapter_output")
    adapter_input = cache["pooled_embedding"]
    if isinstance(model, AdaptedModel):
        adapter_norm_sq = adapter_output.square().sum()
        output_common = (adapter_output @ model.p_common_output).square().sum()
        input_norm_sq = adapter_input.square().sum()
        input_common = (adapter_input @ model.p_common_input).square().sum()
        metrics["adapter_norm"] = float(model.A.detach().norm()) if model.A is not None else 0.0
        metrics["adapter_output_common_energy_fraction"] = float(
            output_common / adapter_norm_sq.clamp_min(1e-30)
        )
        metrics["adapter_input_common_energy_fraction"] = float(
            input_common / input_norm_sq.clamp_min(1e-30)
        )
        metrics["base_relative_drift"] = (
            state_relative_drift(model.base, frozen_state) if frozen_state is not None else float("nan")
        )
    else:
        metrics.update(
            {
                "adapter_norm": 0.0,
                "adapter_output_common_energy_fraction": 0.0,
                "adapter_input_common_energy_fraction": float("nan"),
                "base_relative_drift": (
                    state_relative_drift(model, frozen_state)
                    if frozen_state is not None
                    else float("nan")
                ),
            }
        )
    return metrics


def gradient_block_metrics(model: AdaptedModel) -> Dict[str, float]:
    if model.A is None or model.A.grad is None:
        return {
            "grad_common_to_common_fraction": float("nan"),
            "grad_common_to_residual_fraction": float("nan"),
            "grad_residual_to_residual_fraction": float("nan"),
        }
    grad = model.A.grad.detach()
    pc_in = model.p_common_input
    pr_in = model.p_residual_input
    pc_out = model.p_common_output
    pr_out = model.p_residual_output
    total = grad.square().sum().clamp_min(1e-30)
    # For column-form A x, rows are output and columns are input.
    cc = pc_out @ grad @ pc_in
    rc = pr_out @ grad @ pc_in
    rr = pr_out @ grad @ pr_in
    return {
        "grad_common_to_common_fraction": float(cc.square().sum() / total),
        "grad_common_to_residual_fraction": float(rc.square().sum() / total),
        "grad_residual_to_residual_fraction": float(rr.square().sum() / total),
    }


def history_row(
    dim: int,
    seed: int,
    variant: str,
    phase: str,
    global_step: int,
    local_step: int,
    metrics: Dict[str, float],
    check_angle: float = float("nan"),
    check_sigma_rel_change: float = float("nan"),
    stable_check_count: int = 0,
    switch_reason: str = "",
    grad_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "dim": dim,
        "seed": seed,
        "variant": variant,
        "phase": phase,
        "global_step": global_step,
        "local_step": local_step,
        "check_angle_deg": check_angle,
        "check_sigma_rel_change": check_sigma_rel_change,
        "stable_check_count": stable_check_count,
        "switch_reason": switch_reason,
    }
    row.update(metrics)
    row.update(
        grad_metrics
        or {
            "grad_common_to_common_fraction": float("nan"),
            "grad_common_to_residual_fraction": float("nan"),
            "grad_residual_to_residual_fraction": float("nan"),
        }
    )
    return row


def first_stable_step(
    rows: Sequence[Dict[str, object]], metric: str, threshold: float, window: int
) -> Optional[int]:
    ordered = sorted(rows, key=lambda row: int(row["local_step"]))
    for start in range(0, len(ordered) - window + 1):
        segment = ordered[start : start + window]
        if all(float(row[metric]) >= threshold for row in segment):
            return int(segment[0]["local_step"])
    return None


def train_one_seed_dim(
    cfg: Config, data: Dict[str, object], dim: int, seed: int
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    device = torch.device(cfg.device)
    device_data = dict(data)
    device_data["c1"] = data["c1"].to(device)
    device_data["c2"] = data["c2"].to(device)
    device_data["targets"] = data["targets"].to(device)

    torch.manual_seed(seed)
    base = TiedAttentionLM(len(data["token_names"]), dim, cfg, seed).to(device)
    optimizer = torch.optim.Adam(
        base.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay
    )
    history: List[Dict[str, object]] = []
    previous_input_direction: Optional[torch.Tensor] = None
    previous_output_direction: Optional[torch.Tensor] = None
    previous_sigma: Optional[float] = None
    stable_count = 0
    switch_step = cfg.max_pretrain_steps
    switch_reason = "forced_max_steps"

    initial_metrics = evaluate(base, device_data)
    history.append(
        history_row(dim, seed, "pretrain", "pretrain", 0, 0, initial_metrics)
    )

    for step in range(1, cfg.max_pretrain_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = base(device_data["c1"], device_data["c2"])
        loss = F.cross_entropy(logits, device_data["targets"])
        loss.backward()
        optimizer.step()

        check_angle = float("nan")
        sigma_change = float("nan")
        if step % cfg.check_interval == 0:
            spectrum = value_map_spectrum(base)
            current_input_direction = spectrum["input_direction"]
            current_output_direction = spectrum["output_direction"]
            current_sigma = float(spectrum["sigma1"])
            if (
                previous_input_direction is not None
                and previous_output_direction is not None
                and previous_sigma is not None
            ):
                input_angle = angle_deg(current_input_direction, previous_input_direction)
                output_angle = angle_deg(current_output_direction, previous_output_direction)
                check_angle = max(input_angle, output_angle)
                sigma_change = abs(current_sigma - previous_sigma) / max(abs(previous_sigma), 1e-12)
            metrics = evaluate(base, device_data)
            angle_is_stable = check_angle <= cfg.angle_tol_deg
            sigma_is_stable = sigma_change <= cfg.sigma_rel_tol
            if cfg.stability_rule == "angle_or_sigma":
                geometry_stable = angle_is_stable or sigma_is_stable
            else:
                geometry_stable = angle_is_stable and sigma_is_stable
            is_stable = (
                step >= cfg.min_pretrain_steps
                and metrics["to_K_accuracy"] >= 1.0
                and math.isfinite(check_angle)
                and math.isfinite(sigma_change)
                and geometry_stable
            )
            stable_count = stable_count + 1 if is_stable else 0
            history.append(
                history_row(
                    dim,
                    seed,
                    "pretrain",
                    "pretrain",
                    step,
                    step,
                    metrics,
                    check_angle,
                    sigma_change,
                    stable_count,
                )
            )
            previous_input_direction = current_input_direction.detach().clone()
            previous_output_direction = current_output_direction.detach().clone()
            previous_sigma = current_sigma
            if stable_count >= cfg.stable_checks:
                switch_step = step
                switch_reason = "detected_stable_common"
                break

    switch_spectrum = value_map_spectrum(base)
    frozen_input_direction = switch_spectrum["input_direction"].detach().clone()
    frozen_output_direction = switch_spectrum["output_direction"].detach().clone()
    frozen_state = tensor_state(base)
    switch_optimizer_state = copy.deepcopy(optimizer.state_dict())
    switch_metrics = evaluate(
        base,
        device_data,
        frozen_input_direction,
        frozen_output_direction,
        frozen_state,
    )
    history.append(
        history_row(
            dim,
            seed,
            "switch",
            "switch",
            switch_step,
            0,
            switch_metrics,
            switch_reason=switch_reason,
        )
    )

    summaries: List[Dict[str, object]] = []
    for variant in VARIANTS:
        branch_base = copy.deepcopy(base)
        if variant == "baseline_continue":
            branch_model: torch.nn.Module = branch_base
            branch_optimizer: Optional[torch.optim.Optimizer] = torch.optim.Adam(
                branch_base.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay
            )
            branch_optimizer.load_state_dict(switch_optimizer_state)
        else:
            for parameter in branch_base.parameters():
                parameter.requires_grad_(False)
            branch_model = AdaptedModel(
                branch_base,
                frozen_input_direction,
                frozen_output_direction,
                variant,
            ).to(device)
            branch_optimizer = (
                torch.optim.Adam([branch_model.A], lr=cfg.bridge_lr)
                if branch_model.A is not None
                else None
            )

        variant_rows: List[Dict[str, object]] = []
        metrics = evaluate(
            branch_model,
            device_data,
            frozen_input_direction,
            frozen_output_direction,
            frozen_state,
        )
        row = history_row(
            dim, seed, variant, "post", switch_step, 0, metrics, switch_reason=switch_reason
        )
        history.append(row)
        variant_rows.append(row)

        for local_step in range(1, cfg.post_steps + 1):
            grad_metrics: Optional[Dict[str, float]] = None
            if branch_optimizer is not None:
                branch_optimizer.zero_grad(set_to_none=True)
                logits, _ = branch_model(device_data["c1"], device_data["c2"])
                loss = F.cross_entropy(logits, device_data["targets"])
                loss.backward()
                if isinstance(branch_model, AdaptedModel):
                    grad_metrics = gradient_block_metrics(branch_model)
                branch_optimizer.step()
            metrics = evaluate(
                branch_model,
                device_data,
                frozen_input_direction,
                frozen_output_direction,
                frozen_state,
            )
            row = history_row(
                dim,
                seed,
                variant,
                "post",
                switch_step + local_step,
                local_step,
                metrics,
                switch_reason=switch_reason,
                grad_metrics=grad_metrics,
            )
            history.append(row)
            variant_rows.append(row)

        final = variant_rows[-1]
        initial = variant_rows[0]
        summaries.append(
            {
                "dim": dim,
                "seed": seed,
                "variant": variant,
                "switch_step": switch_step,
                "switch_reason": switch_reason,
                "switch_full_accuracy": switch_metrics["full_accuracy"],
                "switch_to_K_accuracy": switch_metrics["to_K_accuracy"],
                "switch_non_K_accuracy": switch_metrics["non_K_accuracy"],
                "switch_value_output_direction_K_alignment": switch_metrics[
                    "value_output_direction_K_alignment"
                ],
                "initial_loss": initial["loss"],
                "final_loss": final["loss"],
                "initial_non_K_loss": initial["non_K_loss"],
                "final_non_K_loss": final["non_K_loss"],
                "final_full_accuracy": final["full_accuracy"],
                "final_to_K_accuracy": final["to_K_accuracy"],
                "final_non_K_accuracy": final["non_K_accuracy"],
                "first_stable_all_post_step": first_stable_step(
                    variant_rows, "full_accuracy", 1.0, cfg.stable_window
                ),
                "first_stable_non_K_post_step": first_stable_step(
                    variant_rows, "non_K_accuracy", 1.0, cfg.stable_window
                ),
                "final_adapter_norm": final["adapter_norm"],
                "final_adapter_output_common_energy_fraction": final[
                    "adapter_output_common_energy_fraction"
                ],
                "final_adapter_input_common_energy_fraction": final[
                    "adapter_input_common_energy_fraction"
                ],
                "final_base_relative_drift": final["base_relative_drift"],
                "final_frozen_input_direction_angle_deg": final[
                    "frozen_input_direction_angle_deg"
                ],
                "final_frozen_output_direction_angle_deg": final[
                    "frozen_output_direction_angle_deg"
                ],
            }
        )
    return history, summaries


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_curve(
    rows: Sequence[Dict[str, object]], dim: int, variant: str, metric: str
) -> Tuple[np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if row["phase"] == "post" and row["dim"] == dim and row["variant"] == variant
    ]
    by_step: Dict[int, List[float]] = {}
    for row in selected:
        by_step.setdefault(int(row["local_step"]), []).append(float(row[metric]))
    steps = np.array(sorted(by_step), dtype=np.int64)
    values = np.array([np.mean(by_step[step]) for step in steps], dtype=np.float64)
    return steps, values


def make_plots(history: Sequence[Dict[str, object]], output_dir: Path, cfg: Config) -> None:
    colors = {
        "baseline_continue": "black",
        "frozen_no_bridge": "gray",
        "unconstrained_bridge": "tab:blue",
        "common_to_residual": "tab:green",
        "residual_to_residual": "tab:red",
    }

    fig, axes = plt.subplots(len(cfg.dims), 2, figsize=(11, 4 * len(cfg.dims)), squeeze=False)
    for row_index, dim in enumerate(cfg.dims):
        for variant in VARIANTS:
            steps, losses = mean_curve(history, dim, variant, "non_K_loss")
            _, accuracies = mean_curve(history, dim, variant, "non_K_accuracy")
            axes[row_index, 0].plot(steps, losses, label=variant, color=colors[variant])
            axes[row_index, 1].plot(steps, accuracies, label=variant, color=colors[variant])
        axes[row_index, 0].set_title(f"dim={dim}: post-switch non-K loss")
        axes[row_index, 1].set_title(f"dim={dim}: post-switch non-K accuracy")
        axes[row_index, 0].set_xlabel("post-switch step")
        axes[row_index, 1].set_xlabel("post-switch step")
        axes[row_index, 0].set_yscale("log")
        axes[row_index, 1].set_ylim(-0.02, 1.02)
        axes[row_index, 0].grid(alpha=0.25)
        axes[row_index, 1].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(cfg.dims), 2, figsize=(11, 4 * len(cfg.dims)), squeeze=False)
    for row_index, dim in enumerate(cfg.dims):
        for seed in cfg.seeds:
            selected = sorted(
                [
                    row
                    for row in history
                    if row["phase"] == "pretrain"
                    and row["dim"] == dim
                    and row["seed"] == seed
                    and math.isfinite(float(row["check_angle_deg"]))
                ],
                key=lambda row: int(row["local_step"]),
            )
            if not selected:
                continue
            steps = [int(row["local_step"]) for row in selected]
            angles = [float(row["check_angle_deg"]) for row in selected]
            sigma_changes = [float(row["check_sigma_rel_change"]) for row in selected]
            axes[row_index, 0].plot(steps, angles, label=f"seed={seed}")
            axes[row_index, 1].plot(steps, sigma_changes, label=f"seed={seed}")
        axes[row_index, 0].axhline(cfg.angle_tol_deg, color="black", linestyle="--", alpha=0.6)
        axes[row_index, 1].axhline(cfg.sigma_rel_tol, color="black", linestyle="--", alpha=0.6)
        axes[row_index, 0].set_title(f"dim={dim}: top-direction angle change")
        axes[row_index, 1].set_title(f"dim={dim}: sigma1 relative change")
        axes[row_index, 0].set_ylabel("degrees")
        axes[row_index, 1].set_yscale("log")
        axes[row_index, 0].grid(alpha=0.25)
        axes[row_index, 1].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "switch_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(cfg.dims), 2, figsize=(11, 4 * len(cfg.dims)), squeeze=False)
    for row_index, dim in enumerate(cfg.dims):
        for variant in ("unconstrained_bridge", "common_to_residual", "residual_to_residual"):
            steps, leakage = mean_curve(
                history, dim, variant, "adapter_output_common_energy_fraction"
            )
            _, drift = mean_curve(history, dim, variant, "base_relative_drift")
            axes[row_index, 0].plot(steps, leakage, label=variant, color=colors[variant])
            axes[row_index, 1].plot(steps, drift, label=variant, color=colors[variant])
        axes[row_index, 0].set_title(f"dim={dim}: adapter output common energy")
        axes[row_index, 1].set_title(f"dim={dim}: frozen-base relative drift")
        axes[row_index, 0].set_yscale("symlog", linthresh=1e-14)
        axes[row_index, 1].set_yscale("symlog", linthresh=1e-14)
        axes[row_index, 0].grid(alpha=0.25)
        axes[row_index, 1].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "geometry_curves.png", dpi=180)
    plt.close(fig)


def aggregate_summary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for dim in sorted({int(row["dim"]) for row in rows}):
        for variant in VARIANTS:
            selected = [row for row in rows if row["dim"] == dim and row["variant"] == variant]
            if not selected:
                continue
            aggregate: Dict[str, object] = {"dim": dim, "variant": variant, "runs": len(selected)}
            for metric in (
                "switch_step",
                "switch_full_accuracy",
                "switch_non_K_accuracy",
                "switch_value_output_direction_K_alignment",
                "initial_loss",
                "final_loss",
                "initial_non_K_loss",
                "final_non_K_loss",
                "final_full_accuracy",
                "final_non_K_accuracy",
                "final_adapter_output_common_energy_fraction",
                "final_base_relative_drift",
            ):
                values = [float(row[metric]) for row in selected]
                aggregate[f"mean_{metric}"] = float(np.mean(values))
            stable = [
                float(row["first_stable_all_post_step"])
                for row in selected
                if row["first_stable_all_post_step"] not in (None, "")
            ]
            aggregate["stable_all_successes"] = len(stable)
            aggregate["median_first_stable_all_post_step"] = (
                float(np.median(stable)) if stable else None
            )
            output.append(aggregate)
    return output


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=parse_int_tuple, default=(2, 3))
    parser.add_argument("--seeds", type=parse_int_tuple, default=(0, 1, 2))
    parser.add_argument("--base-lr", type=float, default=0.03)
    parser.add_argument("--bridge-lr", type=float, default=0.05)
    parser.add_argument("--max-pretrain-steps", type=int, default=300)
    parser.add_argument("--post-steps", type=int, default=1500)
    parser.add_argument("--min-pretrain-steps", type=int, default=15)
    parser.add_argument("--check-interval", type=int, default=3)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument(
        "--stability-rule",
        choices=("angle_or_sigma", "angle_and_sigma"),
        default="angle_or_sigma",
    )
    parser.add_argument("--angle-tol-deg", type=float, default=0.5)
    parser.add_argument("--sigma-rel-tol", type=float, default=0.002)
    parser.add_argument("--stable-window", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    return Config(
        dims=args.dims,
        seeds=args.seeds,
        base_lr=args.base_lr,
        bridge_lr=args.bridge_lr,
        max_pretrain_steps=args.max_pretrain_steps,
        post_steps=args.post_steps,
        min_pretrain_steps=args.min_pretrain_steps,
        check_interval=args.check_interval,
        stable_checks=args.stable_checks,
        stability_rule=args.stability_rule,
        angle_tol_deg=args.angle_tol_deg,
        sigma_rel_tol=args.sigma_rel_tol,
        stable_window=args.stable_window,
        device=args.device,
        output_dir=args.output_dir,
    )


def main() -> None:
    cfg = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = build_data()
    all_history: List[Dict[str, object]] = []
    all_summary: List[Dict[str, object]] = []
    for dim in cfg.dims:
        for seed in cfg.seeds:
            history, summary = train_one_seed_dim(cfg, data, dim, seed)
            all_history.extend(history)
            all_summary.extend(summary)
            switch = summary[0]
            print(
                f"dim={dim} seed={seed} switch={switch['switch_step']} "
                f"reason={switch['switch_reason']} "
                f"switch_acc={float(switch['switch_full_accuracy']):.3f}"
            )

    write_csv(output_dir / "history.csv", all_history)
    write_csv(output_dir / "summary.csv", all_summary)
    aggregate = aggregate_summary(all_summary)
    write_csv(output_dir / "aggregate_summary.csv", aggregate)
    with (output_dir / "config.json").open("w") as handle:
        json.dump(asdict(cfg), handle, indent=2)
    with (output_dir / "data_contract.json").open("w") as handle:
        json.dump(
            {
                "token_names": data["token_names"],
                "pattern_names": data["pattern_names"],
                "groups": data["groups"],
                "families": data["families"],
            },
            handle,
            indent=2,
        )
    make_plots(all_history, output_dir, cfg)
    print(f"wrote results to {output_dir}")


if __name__ == "__main__":
    main()
