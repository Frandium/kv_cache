#!/usr/bin/env python3
"""Factorial test of nested token sharing versus frequency-weighted learning."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nested-frequency-ceiling")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


OBJECTIVES = ("zipf_raw", "uniform_raw", "zipf_reweight")
SHARING_MODES = ("shared", "split")
PATTERN_NAMES = (
    "the_sun",
    "the_moon",
    "a_moon_cake",
    "a_banana_cake",
    "a_fruit_cake",
)


@dataclass(frozen=True)
class Config:
    dims: Tuple[int, ...] = (8, 16)
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    steps: int = 500
    eval_interval: int = 10
    stable_evals: int = 5
    bayes_gap_threshold: float = 0.03
    cake_loss_threshold: float = 0.03
    lr: float = 0.03
    init_scale: float = 0.35
    matrix_init_scale: float = 0.08
    residual_alpha: float = 1.0
    device: str = "cpu"
    output_dir: str = "results"


def parse_int_tuple(text: str) -> Tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def semantic_token_vectors(dim: int, cfg: Config, seed: int) -> Dict[str, torch.Tensor]:
    names = ("pad", "the", "a", "sun", "moon", "banana", "fruit", "cake")
    generator = torch.Generator().manual_seed(7919 + seed * 101 + dim * 1009)
    matrix = torch.randn(len(names), dim, generator=generator)
    matrix = F.normalize(matrix, dim=-1) * cfg.init_scale
    return {name: matrix[index].clone() for index, name in enumerate(names)}


def build_data(sharing: str, dim: int, cfg: Config, seed: int) -> Dict[str, object]:
    if sharing not in SHARING_MODES:
        raise ValueError(sharing)
    semantic = semantic_token_vectors(dim, cfg, seed)
    if sharing == "shared":
        token_names = ["pad", "the", "a", "sun", "moon", "banana", "fruit", "cake"]
        init_rows = [semantic[name] for name in token_names]
        sequences = (
            ("the", "sun", "pad"),
            ("the", "moon", "pad"),
            ("a", "moon", "cake"),
            ("a", "banana", "cake"),
            ("a", "fruit", "cake"),
        )
        moon_high_name = "moon"
        moon_tail_name = "moon"
    else:
        token_names = [
            "pad",
            "the",
            "a",
            "sun",
            "moon_H",
            "moon_T",
            "banana",
            "fruit",
            "cake",
        ]
        init_rows = []
        for name in token_names:
            semantic_name = "moon" if name in {"moon_H", "moon_T"} else name
            init_rows.append(semantic[semantic_name])
        sequences = (
            ("the", "sun", "pad"),
            ("the", "moon_H", "pad"),
            ("a", "moon_T", "cake"),
            ("a", "banana", "cake"),
            ("a", "fruit", "cake"),
        )
        moon_high_name = "moon_H"
        moon_tail_name = "moon_T"

    token_to_id = {name: index for index, name in enumerate(token_names)}
    token_ids = torch.tensor(
        [[token_to_id[token] for token in sequence] for sequence in sequences],
        dtype=torch.long,
    )
    pad_id = token_to_id["pad"]
    input_ids = token_ids[:, :-1]
    targets = token_ids[:, 1:]
    valid_target_mask = targets.ne(pad_id)

    counts = {
        "zipf_raw": torch.tensor([6, 6, 1, 1, 1], dtype=torch.float32),
        "uniform_raw": torch.tensor([3, 3, 3, 3, 3], dtype=torch.float32),
        "zipf_reweight": torch.tensor([6, 6, 1, 1, 1], dtype=torch.float32),
    }

    return {
        "token_names": token_names,
        "token_to_id": token_to_id,
        "init_embedding": torch.stack(init_rows),
        "input_ids": input_ids,
        "targets": targets,
        "valid_target_mask": valid_target_mask,
        "pad_id": pad_id,
        "counts": counts,
        "sequences": sequences,
        "moon_high_name": moon_high_name,
        "moon_tail_name": moon_tail_name,
    }


def objective_weights(data: Dict[str, object], objective: str) -> Dict[str, torch.Tensor]:
    counts = data["counts"][objective]
    probabilities = counts / counts.sum()
    if objective == "zipf_reweight":
        coefficients = torch.full_like(probabilities, 0.2) / probabilities
        effective = torch.full_like(probabilities, 0.2)
    else:
        coefficients = torch.ones_like(probabilities)
        effective = probabilities * coefficients
        effective = effective / effective.sum()
    return {
        "counts": counts,
        "probabilities": probabilities,
        "coefficients": coefficients,
        "effective": effective,
    }


class TinyAttentionLM(torch.nn.Module):
    def __init__(self, init_embedding: torch.Tensor, cfg: Config, seed: int):
        super().__init__()
        vocab_size, dim = init_embedding.shape
        self.E = torch.nn.Parameter(init_embedding.clone())
        generator = torch.Generator().manual_seed(1543 + seed * 103 + dim * 2017)
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
        self.vocab_size = vocab_size

    def forward(
        self, input_ids: torch.Tensor, pad_id: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.E[input_ids]
        q = x @ self.Wq.T
        k = x @ self.Wk.T
        v = x @ self.Wv.T
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dim)
        length = input_ids.shape[1]
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=input_ids.device), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        key_padding = input_ids.eq(pad_id).unsqueeze(1)
        scores = scores.masked_fill(key_padding, float("-inf"))
        attention = F.softmax(scores, dim=-1)
        pooled_v = attention @ v
        attention_out = pooled_v @ self.Wo.T
        hidden = attention_out + self.residual_alpha * x
        logits = hidden @ self.E.T
        logits = logits.clone()
        logits[..., pad_id] = -1e9
        return logits, {"hidden": hidden, "attention": attention, "input_embedding": x}


def sequence_losses(
    logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    vocab_size = logits.shape[-1]
    token_losses = F.cross_entropy(
        logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    token_losses = token_losses * valid_mask.float()
    seq_losses = token_losses.sum(-1) / valid_mask.sum(-1).clamp_min(1)
    return seq_losses, token_losses


def entropy_effective_rank(values: torch.Tensor) -> float:
    values = values.float().clamp_min(0)
    probs = values / values.sum().clamp_min(1e-30)
    return float(torch.exp(-(probs * probs.clamp_min(1e-30).log()).sum()))


def matrix_spectrum(matrix: torch.Tensor, center_rows: bool = False) -> Dict[str, float]:
    matrix = matrix.detach().float()
    if center_rows:
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(matrix)
    energy = singular_values.square()
    return {
        "top1_energy": float(energy[0] / energy.sum().clamp_min(1e-30)),
        "effective_rank": entropy_effective_rank(energy),
        "sigma1": float(singular_values[0]),
        "sigma2": float(singular_values[1]) if singular_values.numel() > 1 else 0.0,
    }


def covariance_spectrum(
    rows: torch.Tensor, weights: Optional[torch.Tensor] = None, center: bool = False
) -> Dict[str, float]:
    rows = rows.detach().float()
    if weights is None:
        weights = torch.full((rows.shape[0],), 1.0 / rows.shape[0], device=rows.device)
    else:
        weights = weights.float()
        weights = weights / weights.sum().clamp_min(1e-30)
    if center:
        rows = rows - (weights[:, None] * rows).sum(0, keepdim=True)
    covariance = (rows * weights.sqrt()[:, None]).T @ (rows * weights.sqrt()[:, None])
    eigenvalues = torch.linalg.eigvalsh(covariance).flip(0).clamp_min(0)
    return {
        "top1_energy": float(eigenvalues[0] / eigenvalues.sum().clamp_min(1e-30)),
        "effective_rank": entropy_effective_rank(eigenvalues),
        "lambda1": float(eigenvalues[0]),
        "lambda2": float(eigenvalues[1]) if eigenvalues.numel() > 1 else 0.0,
    }


def canonical_embedding(model: TinyAttentionLM, data: Dict[str, object]) -> torch.Tensor:
    ids = data["token_to_id"]
    rows = [model.E[ids[name]] for name in ("the", "a", "sun")]
    if data["moon_high_name"] == data["moon_tail_name"]:
        moon = model.E[ids[data["moon_high_name"]]]
    else:
        moon = 0.5 * (
            model.E[ids[data["moon_high_name"]]]
            + model.E[ids[data["moon_tail_name"]]]
        )
    rows.append(moon)
    rows.extend(model.E[ids[name]] for name in ("banana", "fruit", "cake"))
    return torch.stack(rows)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left[None], right[None]).squeeze())


def first_stable_step(
    rows: Sequence[Dict[str, object]], metrics: Sequence[str], thresholds: Sequence[float], count: int
) -> Optional[int]:
    ordered = sorted(rows, key=lambda row: int(row["step"]))
    for start in range(len(ordered) - count + 1):
        segment = ordered[start : start + count]
        if all(
            all(float(row[metric]) <= threshold for metric, threshold in zip(metrics, thresholds))
            for row in segment
        ):
            return int(segment[0]["step"])
    return None


@torch.no_grad()
def evaluate(
    model: TinyAttentionLM,
    data: Dict[str, object],
    weights: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    logits, cache = model(data["input_ids"], int(data["pad_id"]))
    seq_losses, token_losses = sequence_losses(
        logits, data["targets"], data["valid_target_mask"]
    )
    effective = weights["effective"].to(logits.device)
    objective_loss = (effective * seq_losses).sum()
    macro_loss = seq_losses.mean()

    the_ce = token_losses[:2, 0].mean()
    a_ce = token_losses[2:, 0].mean()
    cake_losses = token_losses[2:, 1]
    cake_loss = cake_losses.mean()
    cake_accuracy = float(
        logits[2:, 1].argmax(-1).eq(data["targets"][2:, 1]).float().mean()
    )

    hidden = cache["hidden"]
    # High contextual states: h(the), h(the sun), h(the moon_H/shared).
    high_states = torch.stack((hidden[0, 0], hidden[0, 1], hidden[1, 1]))
    _, _, high_vh = torch.linalg.svd(high_states.float(), full_matrices=False)
    common_direction = F.normalize(high_vh[0], dim=0)
    p_common = torch.outer(common_direction, common_direction)
    p_residual = torch.eye(model.dim, device=hidden.device) - p_common
    tail_noun_states = hidden[2:, 1]
    tail_common_energy = (tail_noun_states @ p_common).square().sum()
    tail_total_energy = tail_noun_states.square().sum().clamp_min(1e-30)
    tail_common_fraction = tail_common_energy / tail_total_energy
    tail_contrast = tail_noun_states - tail_noun_states.mean(0, keepdim=True)
    contrast_common = (tail_contrast @ p_common).square().sum()
    contrast_total = tail_contrast.square().sum().clamp_min(1e-30)

    common_logits = (tail_noun_states @ p_common) @ model.E.T
    residual_logits = (tail_noun_states @ p_residual) @ model.E.T
    common_logits[:, int(data["pad_id"])] = -1e9
    residual_logits[:, int(data["pad_id"])] = -1e9
    cake_targets = data["targets"][2:, 1]
    common_only_cake_loss = F.cross_entropy(common_logits, cake_targets)
    residual_only_cake_loss = F.cross_entropy(residual_logits, cake_targets)

    # Seven macro semantic contexts; duplicates of h(the) and h(a) are removed.
    macro_states = torch.stack(
        (
            hidden[0, 0],
            hidden[2, 0],
            hidden[0, 1],
            hidden[1, 1],
            hidden[2, 1],
            hidden[3, 1],
            hidden[4, 1],
        )
    )
    macro_spec = covariance_spectrum(macro_states, center=True)

    # Objective-weighted valid prediction contexts. Each sequence's weight is
    # divided over its valid positions, matching the training objective.
    weighted_states: List[torch.Tensor] = []
    state_weights: List[torch.Tensor] = []
    valid_mask = data["valid_target_mask"]
    for pattern_index in range(5):
        positions = torch.nonzero(valid_mask[pattern_index], as_tuple=False).flatten()
        per_position = effective[pattern_index] / max(int(positions.numel()), 1)
        for position in positions.tolist():
            weighted_states.append(hidden[pattern_index, position])
            state_weights.append(per_position)
    weighted_spec = covariance_spectrum(
        torch.stack(weighted_states), torch.stack(state_weights), center=True
    )

    embedding_raw = matrix_spectrum(canonical_embedding(model, data), center_rows=False)
    embedding_centered = matrix_spectrum(canonical_embedding(model, data), center_rows=True)
    bqk_spec = matrix_spectrum(model.Wq.T @ model.Wk)
    bvo_spec = matrix_spectrum(model.Wo @ model.Wv)

    ids = data["token_to_id"]
    moon_high = model.E[ids[data["moon_high_name"]]]
    moon_tail = model.E[ids[data["moon_tail_name"]]]
    cake = model.E[ids["cake"]]
    static_cosines = {
        "cos_cake_moon_tail": cosine(cake, moon_tail),
        "cos_cake_moon_high": cosine(cake, moon_high),
        "cos_cake_banana": cosine(cake, model.E[ids["banana"]]),
        "cos_cake_fruit": cosine(cake, model.E[ids["fruit"]]),
        "cos_cake_sun": cosine(cake, model.E[ids["sun"]]),
        "cos_cake_the": cosine(cake, model.E[ids["the"]]),
        "cos_cake_a": cosine(cake, model.E[ids["a"]]),
        "cos_moon_high_the": cosine(moon_high, model.E[ids["the"]]),
        "cos_moon_tail_a": cosine(moon_tail, model.E[ids["a"]]),
        "cos_context_highmoon_tailmoon": cosine(hidden[1, 1], hidden[2, 1]),
        "cos_context_cake_moon_banana": cosine(hidden[2, 1], hidden[3, 1]),
        "cos_context_cake_moon_fruit": cosine(hidden[2, 1], hidden[4, 1]),
    }

    result: Dict[str, float] = {
        "objective_loss": float(objective_loss),
        "macro_loss": float(macro_loss),
        "the_bayes_gap": float(the_ce - math.log(2.0)),
        "a_bayes_gap": float(a_ce - math.log(3.0)),
        "cake_loss": float(cake_loss),
        "cake_accuracy": cake_accuracy,
        "tail_common_energy_fraction": float(tail_common_fraction),
        "tail_contrast_common_energy_fraction": float(contrast_common / contrast_total),
        "common_only_cake_loss": float(common_only_cake_loss),
        "residual_only_cake_loss": float(residual_only_cake_loss),
        "rep_macro_top1_energy": macro_spec["top1_energy"],
        "rep_macro_effective_rank": macro_spec["effective_rank"],
        "rep_weighted_top1_energy": weighted_spec["top1_energy"],
        "rep_weighted_effective_rank": weighted_spec["effective_rank"],
        "embedding_raw_top1_energy": embedding_raw["top1_energy"],
        "embedding_raw_effective_rank": embedding_raw["effective_rank"],
        "embedding_centered_top1_energy": embedding_centered["top1_energy"],
        "embedding_centered_effective_rank": embedding_centered["effective_rank"],
        "Bqk_top1_energy": bqk_spec["top1_energy"],
        "Bqk_effective_rank": bqk_spec["effective_rank"],
        "Bvo_top1_energy": bvo_spec["top1_energy"],
        "Bvo_effective_rank": bvo_spec["effective_rank"],
    }
    for index, name in enumerate(PATTERN_NAMES):
        result[f"loss_{name}"] = float(seq_losses[index])
    result.update(static_cosines)
    return result


def flat_gradients(
    loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter], retain_graph: bool
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=False
    )
    return torch.cat([gradient.detach().reshape(-1) for gradient in gradients])


def gradient_diagnostics(
    model: TinyAttentionLM,
    data: Dict[str, object],
    weights: Dict[str, torch.Tensor],
    metadata: Dict[str, object],
) -> List[Dict[str, object]]:
    logits, _ = model(data["input_ids"], int(data["pad_id"]))
    seq_losses, _ = sequence_losses(logits, data["targets"], data["valid_target_mask"])
    parameters = tuple(model.parameters())
    gradients = [
        flat_gradients(seq_losses[index], parameters, retain_graph=index < 4)
        for index in range(5)
    ]
    high_gradient = 0.5 * (gradients[0] + gradients[1])
    high_norm = high_gradient.norm()
    common_direction = high_gradient / high_norm.clamp_min(1e-30)
    effective = weights["effective"].to(high_gradient.device)
    rows: List[Dict[str, object]] = []
    for index, (name, gradient) in enumerate(zip(PATTERN_NAMES, gradients)):
        projection_signed = torch.dot(gradient, common_direction)
        projection_abs = projection_signed.abs()
        residual = gradient - projection_signed * common_direction
        row = dict(metadata)
        row.update(
            {
                "pattern": name,
                "is_tail": int(index >= 2),
                "raw_grad_norm": float(gradient.norm()),
                "raw_common_abs": float(projection_abs),
                "raw_residual_norm": float(residual.norm()),
                "effective_weight": float(effective[index]),
                "weighted_grad_norm": float(effective[index] * gradient.norm()),
                "weighted_common_abs": float(effective[index] * projection_abs),
                "weighted_residual_norm": float(effective[index] * residual.norm()),
                "cosine_with_high_gradient": float(
                    torch.dot(gradient, common_direction) / gradient.norm().clamp_min(1e-30)
                ),
                "high_macro_grad_norm": float(high_norm),
            }
        )
        rows.append(row)
    return rows


def train_condition(
    cfg: Config,
    dim: int,
    seed: int,
    sharing: str,
    objective: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    device = torch.device(cfg.device)
    data = build_data(sharing, dim, cfg, seed)
    for key in ("init_embedding", "input_ids", "targets", "valid_target_mask"):
        data[key] = data[key].to(device)
    weights = objective_weights(data, objective)
    for key in weights:
        weights[key] = weights[key].to(device)

    torch.manual_seed(seed)
    model = TinyAttentionLM(data["init_embedding"], cfg, seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: List[Dict[str, object]] = []
    gradient_history: List[Dict[str, object]] = []

    eval_steps = set(range(0, cfg.steps + 1, cfg.eval_interval))
    eval_steps.add(cfg.steps)
    for step in range(cfg.steps + 1):
        if step in eval_steps:
            metrics = evaluate(model, data, weights)
            row: Dict[str, object] = {
                "dim": dim,
                "seed": seed,
                "sharing": sharing,
                "objective": objective,
                "step": step,
            }
            row.update(metrics)
            history.append(row)
            gradient_history.extend(
                gradient_diagnostics(
                    model,
                    data,
                    weights,
                    {
                        "dim": dim,
                        "seed": seed,
                        "sharing": sharing,
                        "objective": objective,
                        "step": step,
                    },
                )
            )
        if step == cfg.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(data["input_ids"], int(data["pad_id"]))
        seq_losses, _ = sequence_losses(logits, data["targets"], data["valid_target_mask"])
        loss = (weights["effective"] * seq_losses).sum()
        loss.backward()
        optimizer.step()

    stable_the = first_stable_step(
        history, ("the_bayes_gap",), (cfg.bayes_gap_threshold,), cfg.stable_evals
    )
    stable_a = first_stable_step(
        history, ("a_bayes_gap",), (cfg.bayes_gap_threshold,), cfg.stable_evals
    )
    stable_cake = first_stable_step(
        history, ("cake_loss",), (cfg.cake_loss_threshold,), cfg.stable_evals
    )
    stable_all = first_stable_step(
        history,
        ("the_bayes_gap", "a_bayes_gap", "cake_loss"),
        (cfg.bayes_gap_threshold, cfg.bayes_gap_threshold, cfg.cake_loss_threshold),
        cfg.stable_evals,
    )
    final = history[-1]
    counts = weights["counts"].detach().cpu()
    summary: Dict[str, object] = {
        "dim": dim,
        "seed": seed,
        "sharing": sharing,
        "objective": objective,
        "first_stable_the_step": stable_the,
        "first_stable_a_step": stable_a,
        "first_stable_cake_step": stable_cake,
        "first_stable_all_step": stable_all,
        "cake_pattern_exposure_at_stable": (
            None if stable_cake is None else float(counts[2:].mean() * stable_cake)
        ),
    }
    for key, value in final.items():
        if key not in {"dim", "seed", "sharing", "objective", "step"}:
            summary[f"final_{key}"] = value
    if stable_all is not None:
        convergence_row = next(row for row in history if int(row["step"]) == stable_all)
        for key, value in convergence_row.items():
            if key not in {"dim", "seed", "sharing", "objective", "step"}:
                summary[f"converged_{key}"] = value
    return history, gradient_history, summary


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    numeric_metrics = (
        "first_stable_the_step",
        "first_stable_a_step",
        "first_stable_cake_step",
        "first_stable_all_step",
        "cake_pattern_exposure_at_stable",
        "final_the_bayes_gap",
        "final_a_bayes_gap",
        "final_cake_loss",
        "final_Bqk_top1_energy",
        "final_Bvo_top1_energy",
        "final_embedding_centered_top1_energy",
        "final_rep_macro_top1_energy",
        "final_rep_weighted_top1_energy",
        "final_tail_common_energy_fraction",
        "final_tail_contrast_common_energy_fraction",
        "final_common_only_cake_loss",
        "final_residual_only_cake_loss",
        "final_cos_cake_moon_tail",
        "final_cos_cake_banana",
        "final_cos_cake_fruit",
        "final_cos_cake_sun",
        "final_cos_context_highmoon_tailmoon",
        "converged_Bqk_top1_energy",
        "converged_Bvo_top1_energy",
        "converged_embedding_centered_top1_energy",
        "converged_rep_macro_top1_energy",
        "converged_rep_weighted_top1_energy",
        "converged_tail_common_energy_fraction",
        "converged_tail_contrast_common_energy_fraction",
        "converged_common_only_cake_loss",
        "converged_residual_only_cake_loss",
    )
    for dim in sorted({int(row["dim"]) for row in rows}):
        for sharing in SHARING_MODES:
            for objective in OBJECTIVES:
                selected = [
                    row
                    for row in rows
                    if int(row["dim"]) == dim
                    and row["sharing"] == sharing
                    and row["objective"] == objective
                ]
                if not selected:
                    continue
                aggregate: Dict[str, object] = {
                    "dim": dim,
                    "sharing": sharing,
                    "objective": objective,
                    "runs": len(selected),
                }
                for metric in numeric_metrics:
                    values = [
                        float(row[metric])
                        for row in selected
                        if row.get(metric) not in (None, "")
                    ]
                    aggregate[f"mean_{metric}"] = float(np.mean(values)) if values else None
                    aggregate[f"median_{metric}"] = float(np.median(values)) if values else None
                    if metric.startswith("first_stable"):
                        aggregate[f"successes_{metric}"] = len(values)
                output.append(aggregate)
    return output


def mean_curve(
    rows: Sequence[Dict[str, object]], dim: int, sharing: str, objective: str, metric: str
) -> Tuple[np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if int(row["dim"]) == dim
        and row["sharing"] == sharing
        and row["objective"] == objective
    ]
    by_step: Dict[int, List[float]] = {}
    for row in selected:
        by_step.setdefault(int(row["step"]), []).append(float(row[metric]))
    steps = np.array(sorted(by_step), dtype=np.int64)
    values = np.array([np.mean(by_step[step]) for step in steps], dtype=np.float64)
    return steps, values


def make_plots(
    history: Sequence[Dict[str, object]],
    gradient_history: Sequence[Dict[str, object]],
    summary: Sequence[Dict[str, object]],
    output_dir: Path,
    primary_dim: int,
) -> None:
    colors = {"zipf_raw": "tab:red", "uniform_raw": "tab:blue", "zipf_reweight": "tab:green"}
    linestyles = {"shared": "-", "split": "--"}
    labels = {
        (sharing, objective): f"{objective}, {sharing}"
        for sharing in SHARING_MODES
        for objective in OBJECTIVES
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for sharing in SHARING_MODES:
        for objective in OBJECTIVES:
            for axis, metric, title in zip(
                axes,
                ("the_bayes_gap", "a_bayes_gap", "cake_loss"),
                ("the prefix: Bayes gap", "a prefix: Bayes gap", "noun -> cake loss"),
            ):
                steps, values = mean_curve(history, primary_dim, sharing, objective, metric)
                axis.plot(
                    steps,
                    np.maximum(values, 1e-8),
                    color=colors[objective],
                    linestyle=linestyles[sharing],
                    label=labels[(sharing, objective)],
                )
                axis.set_title(title)
                axis.set_yscale("log")
                axis.set_xlabel("optimization step")
                axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"dim={primary_dim}: convergence to the attainable NTP target")
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for sharing in SHARING_MODES:
        for objective in OBJECTIVES:
            for axis, metric, title in zip(
                axes,
                ("Bqk_top1_energy", "Bvo_top1_energy", "rep_macro_top1_energy"),
                ("Bqk top-1 energy", "Bvo top-1 energy", "macro representation top-1 energy"),
            ):
                steps, values = mean_curve(history, primary_dim, sharing, objective, metric)
                axis.plot(
                    steps,
                    values,
                    color=colors[objective],
                    linestyle=linestyles[sharing],
                    label=labels[(sharing, objective)],
                )
                axis.set_title(title)
                axis.set_ylim(0, 1.02)
                axis.set_xlabel("optimization step")
                axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"dim={primary_dim}: parameter and macro representation concentration")
    fig.tight_layout()
    fig.savefig(output_dir / "spectral_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for sharing in SHARING_MODES:
        for objective in OBJECTIVES:
            selected = [
                row
                for row in gradient_history
                if int(row["dim"]) == primary_dim
                and row["sharing"] == sharing
                and row["objective"] == objective
                and int(row["is_tail"]) == 1
            ]
            by_step_raw: Dict[int, List[float]] = {}
            by_step_weighted: Dict[int, List[float]] = {}
            for row in selected:
                step = int(row["step"])
                by_step_raw.setdefault(step, []).append(float(row["raw_grad_norm"]))
                by_step_weighted.setdefault(step, []).append(float(row["weighted_grad_norm"]))
            steps = sorted(by_step_raw)
            axes[0].plot(
                steps,
                [np.mean(by_step_raw[step]) for step in steps],
                color=colors[objective],
                linestyle=linestyles[sharing],
                label=labels[(sharing, objective)],
            )
            axes[1].plot(
                steps,
                [np.mean(by_step_weighted[step]) for step in steps],
                color=colors[objective],
                linestyle=linestyles[sharing],
                label=labels[(sharing, objective)],
            )
    axes[0].set_title("raw per-tail-pattern gradient norm")
    axes[1].set_title("frequency-weighted tail gradient contribution")
    for axis in axes:
        axis.set_yscale("log")
        axis.set_xlabel("optimization step")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"dim={primary_dim}: raw gradient versus optimizer-visible contribution")
    fig.tight_layout()
    fig.savefig(output_dir / "gradient_contributions.png", dpi=180)
    plt.close(fig)

    groups = [
        (sharing, objective)
        for sharing in SHARING_MODES
        for objective in OBJECTIVES
    ]
    grouped_summary = [
        [
            row
            for row in summary
            if int(row["dim"]) == primary_dim
            and row["sharing"] == sharing
            and row["objective"] == objective
        ]
        for sharing, objective in groups
    ]
    x_labels = [f"{objective}\n{sharing}" for sharing, objective in groups]
    x = np.arange(len(groups))
    mean_metric = lambda rows, key: float(np.mean([float(row[key]) for row in rows]))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(
        x,
        [mean_metric(rows, "converged_tail_common_energy_fraction") for rows in grouped_summary],
    )
    axes[0].set_title("tail hidden energy in high-pattern top direction")
    axes[0].set_ylim(0.0, 1.05)
    width = 0.4
    axes[1].bar(
        x - width / 2,
        [mean_metric(rows, "converged_common_only_cake_loss") for rows in grouped_summary],
        width,
        label="common only",
    )
    axes[1].bar(
        x + width / 2,
        [mean_metric(rows, "converged_residual_only_cake_loss") for rows in grouped_summary],
        width,
        label="residual only",
    )
    axes[1].set_title("causal cake loss after subspace ablation")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[2].bar(
        x - width / 2,
        [mean_metric(rows, "converged_cos_cake_moon_tail") for rows in grouped_summary],
        width,
        label="cake vs moon",
    )
    axes[2].bar(
        x + width / 2,
        [
            0.5
            * (
                mean_metric(rows, "converged_cos_cake_banana")
                + mean_metric(rows, "converged_cos_cake_fruit")
            )
            for rows in grouped_summary
        ],
        width,
        label="cake vs banana/fruit",
    )
    axes[2].set_title("static embedding cosine")
    axes[2].legend()
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(f"dim={primary_dim}: representation geometry at first stable convergence (seed mean)")
    fig.tight_layout()
    fig.savefig(output_dir / "representation_geometry.png", dpi=180)
    plt.close(fig)


def validate_contract(cfg: Config) -> None:
    for dim in cfg.dims:
        shared = build_data("shared", dim, cfg, cfg.seeds[0])
        split = build_data("split", dim, cfg, cfg.seeds[0])
        shared_moon = shared["init_embedding"][shared["token_to_id"]["moon"]]
        split_high = split["init_embedding"][split["token_to_id"]["moon_H"]]
        split_tail = split["init_embedding"][split["token_to_id"]["moon_T"]]
        if not torch.equal(shared_moon, split_high) or not torch.equal(split_high, split_tail):
            raise AssertionError("moon aliases do not have identical initialization")
        uniform = objective_weights(shared, "uniform_raw")["effective"]
        reweighted = objective_weights(shared, "zipf_reweight")["effective"]
        if not torch.allclose(uniform, reweighted, atol=1e-8, rtol=0):
            raise AssertionError("zipf_reweight effective objective is not uniform")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=parse_int_tuple, default=(8, 16))
    parser.add_argument("--seeds", type=parse_int_tuple, default=(0, 1, 2, 3, 4))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--stable-evals", type=int, default=5)
    parser.add_argument("--bayes-gap-threshold", type=float, default=0.03)
    parser.add_argument("--cake-loss-threshold", type=float, default=0.03)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    return Config(
        dims=args.dims,
        seeds=args.seeds,
        steps=args.steps,
        eval_interval=args.eval_interval,
        stable_evals=args.stable_evals,
        bayes_gap_threshold=args.bayes_gap_threshold,
        cake_loss_threshold=args.cake_loss_threshold,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
    )


def main() -> None:
    cfg = parse_args()
    validate_contract(cfg)
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_history: List[Dict[str, object]] = []
    all_gradient_history: List[Dict[str, object]] = []
    all_summary: List[Dict[str, object]] = []
    for dim in cfg.dims:
        for seed in cfg.seeds:
            for sharing in SHARING_MODES:
                for objective in OBJECTIVES:
                    history, gradient_history, summary = train_condition(
                        cfg, dim, seed, sharing, objective
                    )
                    all_history.extend(history)
                    all_gradient_history.extend(gradient_history)
                    all_summary.append(summary)
                    print(
                        f"dim={dim} seed={seed} {sharing} {objective} "
                        f"stable_all={summary['first_stable_all_step']} "
                        f"cake={float(summary['final_cake_loss']):.4g}"
                    )

    aggregate = aggregate_summary(all_summary)
    write_csv(output_dir / "history.csv", all_history)
    write_csv(output_dir / "gradient_history.csv", all_gradient_history)
    write_csv(output_dir / "summary.csv", all_summary)
    write_csv(output_dir / "aggregate_summary.csv", aggregate)
    with (output_dir / "config.json").open("w") as handle:
        json.dump(asdict(cfg), handle, indent=2)
    make_plots(
        all_history,
        all_gradient_history,
        all_summary,
        output_dir,
        primary_dim=cfg.dims[0],
    )
    print(f"wrote results to {output_dir}")


if __name__ == "__main__":
    main()
