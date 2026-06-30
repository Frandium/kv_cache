#!/usr/bin/env python3
"""Forced Orthogonal Ceiling Experiment.

Phase 1: Pretrain on high-frequency patterns only, extract common direction c.
Phase 2: Compare three ceiling conditions for tail learning:
  - Natural (lower bound): Zipf, no intervention
  - Reweight (baseline): frequency-balanced forward/backward
  - Forced Orthogonal (ceiling): tail embeddings projected to orthogonal complement of c

Key question: Can any scheme that prevents tail tokens from using the common direction
match the learning efficiency of simply reweighting frequencies?
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-forced-orthogonal-ceiling")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ── Constants ────────────────────────────────────────────────────────────────

PATTERN_NAMES = (
    "the_sun", "the_moon", "a_moon_cake", "a_banana_cake", "a_fruit_cake",
)
PHASE2_CONDITIONS = ("natural", "reweight", "forced_orthogonal")


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
    # Phase 1 max steps; early stop on convergence
    pretrain_max_steps: int = 500
    # Phase 2 training steps (from pretrained checkpoint)
    phase2_steps: int = 500


# ── Helpers ──────────────────────────────────────────────────────────────────

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


# ── Data builders ─────────────────────────────────────────────────────────────

def build_shared_data(dim: int, cfg: Config, seed: int) -> Dict[str, object]:
    """Build shared-mode data (moon is one token for both contexts)."""
    semantic = semantic_token_vectors(dim, cfg, seed)
    token_names = ["pad", "the", "a", "sun", "moon", "banana", "fruit", "cake"]
    init_rows = [semantic[name] for name in token_names]
    sequences = (
        ("the", "sun", "pad"),
        ("the", "moon", "pad"),
        ("a", "moon", "cake"),
        ("a", "banana", "cake"),
        ("a", "fruit", "cake"),
    )
    token_to_id = {name: index for index, name in enumerate(token_names)}
    token_ids = torch.tensor(
        [[token_to_id[token] for token in seq] for seq in sequences],
        dtype=torch.long,
    )
    pad_id = token_to_id["pad"]
    input_ids = token_ids[:, :-1]
    targets = token_ids[:, 1:]
    valid_target_mask = targets.ne(pad_id)
    moon_high_name = "moon"
    moon_tail_name = "moon"
    return {
        "token_names": token_names,
        "token_to_id": token_to_id,
        "init_embedding": torch.stack(init_rows),
        "input_ids": input_ids,
        "targets": targets,
        "valid_target_mask": valid_target_mask,
        "pad_id": pad_id,
        "sequences": sequences,
        "moon_high_name": moon_high_name,
        "moon_tail_name": moon_tail_name,
    }


def build_split_data(dim: int, cfg: Config, seed: int) -> Dict[str, object]:
    """Build split-mode data (moon_H for high, moon_T for tail)."""
    semantic = semantic_token_vectors(dim, cfg, seed)
    token_names = [
        "pad", "the", "a", "sun", "moon_H", "moon_T", "banana", "fruit", "cake",
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
    token_to_id = {name: index for index, name in enumerate(token_names)}
    token_ids = torch.tensor(
        [[token_to_id[token] for token in seq] for seq in sequences],
        dtype=torch.long,
    )
    pad_id = token_to_id["pad"]
    input_ids = token_ids[:, :-1]
    targets = token_ids[:, 1:]
    valid_target_mask = targets.ne(pad_id)
    moon_high_name = "moon_H"
    moon_tail_name = "moon_T"
    return {
        "token_names": token_names,
        "token_to_id": token_to_id,
        "init_embedding": torch.stack(init_rows),
        "input_ids": input_ids,
        "targets": targets,
        "valid_target_mask": valid_target_mask,
        "pad_id": pad_id,
        "sequences": sequences,
        "moon_high_name": moon_high_name,
        "moon_tail_name": moon_tail_name,
    }


# ── Objective weights ────────────────────────────────────────────────────────

def objective_weights_zipf() -> Dict[str, torch.Tensor]:
    counts = torch.tensor([6.0, 6.0, 1.0, 1.0, 1.0])
    probabilities = counts / counts.sum()
    return {
        "counts": counts,
        "probabilities": probabilities,
        "coefficients": torch.ones_like(probabilities),
        "effective": probabilities,
    }


def objective_weights_uniform() -> Dict[str, torch.Tensor]:
    counts = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0])
    probabilities = counts / counts.sum()
    return {
        "counts": counts,
        "probabilities": probabilities,
        "coefficients": torch.ones_like(probabilities),
        "effective": torch.full_like(probabilities, 0.2),
    }


def objective_weights_reweight() -> Dict[str, torch.Tensor]:
    counts = torch.tensor([6.0, 6.0, 1.0, 1.0, 1.0])
    probabilities = counts / counts.sum()
    coefficients = torch.full_like(probabilities, 0.2) / probabilities
    return {
        "counts": counts,
        "probabilities": probabilities,
        "coefficients": coefficients,
        "effective": torch.full_like(probabilities, 0.2),
    }


def objective_weights_high_only() -> Dict[str, torch.Tensor]:
    """For Phase 1: only high patterns."""
    counts = torch.tensor([3.0, 3.0, 0.0, 0.0, 0.0])
    total = counts.sum()
    effective = counts / total  # (0.5, 0.5, 0, 0, 0)
    return {
        "counts": counts,
        "probabilities": effective,
        "coefficients": torch.ones(5),
        "effective": effective,
    }


# ── Model ────────────────────────────────────────────────────────────────────

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


# ── Loss helpers ─────────────────────────────────────────────────────────────

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


# ── Spectral helpers ─────────────────────────────────────────────────────────

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


# ── Common direction extraction ──────────────────────────────────────────────

@torch.no_grad()
def extract_common_direction(model: TinyAttentionLM, data: Dict[str, object]) -> torch.Tensor:
    """Extract common direction c from high-pattern contextual hidden states.

    Uses top left singular vector of [h(the), h(the sun), h(the moon)].
    """
    logits, cache = model(data["input_ids"], int(data["pad_id"]))
    hidden = cache["hidden"]
    # h(the) at position 0 of pattern 0, h(the sun) at position 1 of pattern 0,
    # h(the moon) at position 1 of pattern 1
    high_states = torch.stack((hidden[0, 0], hidden[0, 1], hidden[1, 1]))
    _, _, vh = torch.linalg.svd(high_states.float(), full_matrices=False)
    common_direction = F.normalize(vh[0], dim=0)
    return common_direction


# ── Diagnostics ──────────────────────────────────────────────────────────────

def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left[None], right[None]).squeeze())


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


def first_stable_step(
    rows: Sequence[Dict[str, object]], metrics: Sequence[str],
    thresholds: Sequence[float], count: int,
) -> Optional[int]:
    ordered = sorted(rows, key=lambda row: int(row["step"]))
    for start in range(len(ordered) - count + 1):
        segment = ordered[start : start + count]
        if all(
            all(float(row[metric]) <= threshold
                for metric, threshold in zip(metrics, thresholds))
            for row in segment
        ):
            return int(segment[0]["step"])
    return None


@torch.no_grad()
def evaluate_full(
    model: TinyAttentionLM,
    data: Dict[str, object],
    weights: Dict[str, torch.Tensor],
    common_direction: Optional[torch.Tensor] = None,
    extra_metrics: bool = False,
) -> Dict[str, float]:
    """Full evaluation with all spectral, gradient, and geometric metrics.

    If common_direction is provided, also reports tail projection onto that
    specific direction (for Phase 2 consistency).
    """
    logits, cache = model(data["input_ids"], int(data["pad_id"]))
    seq_losses_vals, token_losses = sequence_losses(
        logits, data["targets"], data["valid_target_mask"]
    )
    effective = weights["effective"].to(logits.device)
    objective_loss = (effective * seq_losses_vals).sum()
    macro_loss = seq_losses_vals.mean()

    the_ce = token_losses[:2, 0].mean()
    a_ce = token_losses[2:, 0].mean()
    cake_losses = token_losses[2:, 1]
    cake_loss = cake_losses.mean()
    cake_accuracy = float(
        logits[2:, 1].argmax(-1).eq(data["targets"][2:, 1]).float().mean()
    )

    hidden = cache["hidden"]
    # High contextual states
    high_states = torch.stack((hidden[0, 0], hidden[0, 1], hidden[1, 1]))
    _, _, high_vh = torch.linalg.svd(high_states.float(), full_matrices=False)
    top_common_direction = F.normalize(high_vh[0], dim=0)

    p_common = torch.outer(top_common_direction, top_common_direction)
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

    # Macro representation spectrum
    macro_states = torch.stack((
        hidden[0, 0], hidden[2, 0], hidden[0, 1], hidden[1, 1],
        hidden[2, 1], hidden[3, 1], hidden[4, 1],
    ))
    macro_spec = covariance_spectrum(macro_states, center=True)

    # Weighted representation spectrum
    weighted_states: List[torch.Tensor] = []
    state_weights_list: List[torch.Tensor] = []
    valid_mask = data["valid_target_mask"]
    for pattern_index in range(5):
        positions = torch.nonzero(valid_mask[pattern_index], as_tuple=False).flatten()
        per_position = effective[pattern_index] / max(int(positions.numel()), 1)
        for position in positions.tolist():
            weighted_states.append(hidden[pattern_index, position])
            state_weights_list.append(per_position)
    weighted_spec = covariance_spectrum(
        torch.stack(weighted_states), torch.stack(state_weights_list), center=True
    )

    # Parameter spectra
    embedding_raw = matrix_spectrum(canonical_embedding(model, data), center_rows=False)
    embedding_centered = matrix_spectrum(canonical_embedding(model, data), center_rows=True)
    bqk_spec = matrix_spectrum(model.Wq.T @ model.Wk)
    bvo_spec = matrix_spectrum(model.Wo @ model.Wv)

    # Cosine similarities
    ids = data["token_to_id"]
    moon_high = model.E[ids[data["moon_high_name"]]]
    moon_tail = model.E[ids[data["moon_tail_name"]]]
    cake_emb = model.E[ids["cake"]]
    static_cosines = {
        "cos_cake_moon_tail": cosine(cake_emb, moon_tail),
        "cos_cake_moon_high": cosine(cake_emb, moon_high),
        "cos_cake_banana": cosine(cake_emb, model.E[ids["banana"]]),
        "cos_cake_fruit": cosine(cake_emb, model.E[ids["fruit"]]),
        "cos_cake_sun": cosine(cake_emb, model.E[ids["sun"]]),
        "cos_moon_high_the": cosine(moon_high, model.E[ids["the"]]),
        "cos_moon_tail_a": cosine(moon_tail, model.E[ids["a"]]),
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

    # Tail projection onto Phase 1 common direction (for Phase 2 diagnostics)
    if common_direction is not None:
        tail_norm = tail_noun_states.norm(dim=-1).clamp_min(1e-30)
        tail_proj = (tail_noun_states @ common_direction).abs() / tail_norm
        result["tail_proj_onto_phase1_c"] = float(tail_proj.mean())
        # Also measure embedding projection
        moon_tail_proj = (moon_tail @ common_direction).abs() / moon_tail.norm().clamp_min(1e-30)
        cake_proj = (cake_emb @ common_direction).abs() / cake_emb.norm().clamp_min(1e-30)
        result["moon_tail_emb_proj_onto_c"] = float(moon_tail_proj)
        result["cake_emb_proj_onto_c"] = float(cake_proj)

    for index, name in enumerate(PATTERN_NAMES):
        result[f"loss_{name}"] = float(seq_losses_vals[index])
    result.update(static_cosines)
    return result


# ── Gradient diagnostics ─────────────────────────────────────────────────────

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
    seq_losses_vals, _ = sequence_losses(logits, data["targets"], data["valid_target_mask"])
    parameters = tuple(model.parameters())
    gradients = [
        flat_gradients(seq_losses_vals[index], parameters, retain_graph=index < 4)
        for index in range(5)
    ]
    high_gradient = 0.5 * (gradients[0] + gradients[1])
    high_norm = high_gradient.norm()
    common_dir = high_gradient / high_norm.clamp_min(1e-30)
    effective = weights["effective"].to(high_gradient.device)
    rows: List[Dict[str, object]] = []
    for index, (name, gradient) in enumerate(zip(PATTERN_NAMES, gradients)):
        projection_signed = torch.dot(gradient, common_dir)
        projection_abs = projection_signed.abs()
        residual = gradient - projection_signed * common_dir
        row = dict(metadata)
        row.update({
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
                torch.dot(gradient, common_dir) / gradient.norm().clamp_min(1e-30)
            ),
            "high_macro_grad_norm": float(high_norm),
        })
        rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Pretrain on high-frequency patterns only
# ═══════════════════════════════════════════════════════════════════════════════

def pretrain_high_only(
    cfg: Config, dim: int, seed: int,
) -> Tuple[TinyAttentionLM, torch.Tensor, Dict[str, object],
           List[Dict[str, object]], int]:
    """Pretrain on high-frequency patterns (the→sun, the→moon) in shared mode.

    Returns: (model, common_direction_c, data, history, convergence_step)
    """
    device = torch.device(cfg.device)
    data = build_shared_data(dim, cfg, seed)
    for key in ("init_embedding", "input_ids", "targets", "valid_target_mask"):
        data[key] = data[key].to(device)

    weights = objective_weights_high_only()
    for key in weights:
        weights[key] = weights[key].to(device)

    torch.manual_seed(seed)
    model = TinyAttentionLM(data["init_embedding"], cfg, seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: List[Dict[str, object]] = []

    eval_steps = set(range(0, cfg.pretrain_max_steps + 1, cfg.eval_interval))
    eval_steps.add(cfg.pretrain_max_steps)

    # Convergence tracking
    the_gap_history: List[float] = []

    for step in range(cfg.pretrain_max_steps + 1):
        if step in eval_steps:
            logits, cache = model(data["input_ids"], int(data["pad_id"]))
            seq_losses_vals, token_losses = sequence_losses(
                logits, data["targets"], data["valid_target_mask"]
            )
            the_ce = token_losses[:2, 0].mean()
            the_gap = float(the_ce - math.log(2.0))
            the_gap_history.append(the_gap)
            row: Dict[str, object] = {
                "phase": "pretrain", "dim": dim, "seed": seed, "step": step,
                "the_bayes_gap": the_gap, "the_ce": float(the_ce),
                "objective_loss": float((weights["effective"] * seq_losses_vals).sum()),
                "macro_loss": float(seq_losses_vals.mean()),
            }
            for idx, name in enumerate(PATTERN_NAMES[:2]):
                row[f"loss_{name}"] = float(seq_losses_vals[idx])
            history.append(row)

        if step == cfg.pretrain_max_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(data["input_ids"], int(data["pad_id"]))
        seq_losses_vals, _ = sequence_losses(
            logits, data["targets"], data["valid_target_mask"]
        )
        # Only loss from high patterns
        loss = (weights["effective"][:2] * seq_losses_vals[:2]).sum()
        loss.backward()
        optimizer.step()

    # Determine convergence: the_bayes_gap <= threshold for stable_evals consecutive evals
    convergence_step = None
    for i in range(len(the_gap_history) - cfg.stable_evals + 1):
        if all(g <= cfg.bayes_gap_threshold for g in the_gap_history[i:i + cfg.stable_evals]):
            convergence_step = i * cfg.eval_interval
            break

    # Extract common direction
    common_direction = extract_common_direction(model, data)
    return model, common_direction, data, history, convergence_step


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Three ceiling comparisons
# ═══════════════════════════════════════════════════════════════════════════════

def phase2_natural(
    cfg: Config, pretrained_model: TinyAttentionLM, shared_data: Dict[str, object],
    dim: int, seed: int, condition_key: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    """Phase 2 Natural: shared mode, Zipf frequencies, no intervention."""
    device = torch.device(cfg.device)
    data = shared_data  # reuse the Phase 1 shared data
    weights = objective_weights_zipf()
    for key in weights:
        weights[key] = weights[key].to(device)

    torch.manual_seed(seed + 10000)
    model = copy.deepcopy(pretrained_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: List[Dict[str, object]] = []
    gradient_history: List[Dict[str, object]] = []

    eval_steps = set(range(0, cfg.phase2_steps + 1, cfg.eval_interval))
    eval_steps.add(cfg.phase2_steps)
    for step in range(cfg.phase2_steps + 1):
        if step in eval_steps:
            metrics = evaluate_full(model, data, weights)
            row: Dict[str, object] = {
                "phase": "phase2", "condition": condition_key,
                "dim": dim, "seed": seed, "step": step,
            }
            row.update(metrics)
            history.append(row)
            gradient_history.extend(
                gradient_diagnostics(model, data, weights, {
                    "phase": "phase2", "condition": condition_key,
                    "dim": dim, "seed": seed, "step": step,
                })
            )
        if step == cfg.phase2_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(data["input_ids"], int(data["pad_id"]))
        seq_losses_vals, _ = sequence_losses(logits, data["targets"], data["valid_target_mask"])
        loss = (weights["effective"] * seq_losses_vals).sum()
        loss.backward()
        optimizer.step()

    stable_all = first_stable_step(
        history,
        ("the_bayes_gap", "a_bayes_gap", "cake_loss"),
        (cfg.bayes_gap_threshold, cfg.bayes_gap_threshold, cfg.cake_loss_threshold),
        cfg.stable_evals,
    )
    final = history[-1]
    summary: Dict[str, object] = {
        "phase": "phase2", "condition": condition_key,
        "dim": dim, "seed": seed,
        "first_stable_all_step": stable_all,
    }
    for key, value in final.items():
        if key not in {"phase", "condition", "dim", "seed", "step"}:
            summary[f"final_{key}"] = value
    if stable_all is not None:
        conv_row = next(row for row in history if int(row["step"]) == stable_all)
        for key, value in conv_row.items():
            if key not in {"phase", "condition", "dim", "seed", "step"}:
                summary[f"converged_{key}"] = value
    return history, gradient_history, summary


def phase2_reweight(
    cfg: Config, pretrained_model: TinyAttentionLM, shared_data: Dict[str, object],
    dim: int, seed: int, condition_key: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    """Phase 2 Reweight: shared mode, frequency-balanced objective."""
    device = torch.device(cfg.device)
    data = shared_data
    weights = objective_weights_reweight()
    for key in weights:
        weights[key] = weights[key].to(device)

    torch.manual_seed(seed + 10000)
    model = copy.deepcopy(pretrained_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: List[Dict[str, object]] = []
    gradient_history: List[Dict[str, object]] = []

    eval_steps = set(range(0, cfg.phase2_steps + 1, cfg.eval_interval))
    eval_steps.add(cfg.phase2_steps)
    for step in range(cfg.phase2_steps + 1):
        if step in eval_steps:
            metrics = evaluate_full(model, data, weights)
            row: Dict[str, object] = {
                "phase": "phase2", "condition": condition_key,
                "dim": dim, "seed": seed, "step": step,
            }
            row.update(metrics)
            history.append(row)
            gradient_history.extend(
                gradient_diagnostics(model, data, weights, {
                    "phase": "phase2", "condition": condition_key,
                    "dim": dim, "seed": seed, "step": step,
                })
            )
        if step == cfg.phase2_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(data["input_ids"], int(data["pad_id"]))
        seq_losses_vals, _ = sequence_losses(logits, data["targets"], data["valid_target_mask"])
        loss = (weights["effective"] * seq_losses_vals).sum()
        loss.backward()
        optimizer.step()

    stable_all = first_stable_step(
        history,
        ("the_bayes_gap", "a_bayes_gap", "cake_loss"),
        (cfg.bayes_gap_threshold, cfg.bayes_gap_threshold, cfg.cake_loss_threshold),
        cfg.stable_evals,
    )
    final = history[-1]
    summary: Dict[str, object] = {
        "phase": "phase2", "condition": condition_key,
        "dim": dim, "seed": seed,
        "first_stable_all_step": stable_all,
    }
    for key, value in final.items():
        if key not in {"phase", "condition", "dim", "seed", "step"}:
            summary[f"final_{key}"] = value
    if stable_all is not None:
        conv_row = next(row for row in history if int(row["step"]) == stable_all)
        for key, value in conv_row.items():
            if key not in {"phase", "condition", "dim", "seed", "step"}:
                summary[f"converged_{key}"] = value
    return history, gradient_history, summary


def phase2_forced_orthogonal(
    cfg: Config, pretrained_model: TinyAttentionLM, dim: int, seed: int,
    common_direction: torch.Tensor, condition_key: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object],
           Dict[str, object]]:
    """Phase 2 Forced Orthogonal: split mode, reweight objective,
    tail embeddings projected to orthogonal complement of c after each optimizer step.

    Uses split mode to avoid penalizing high-frequency patterns:
    - moon_H (high pattern) is NOT projected
    - moon_T, cake, banana, fruit (tail) ARE projected after each step

    This gives Forced Orthogonal an oracle advantage (split) compared to
    Natural and Reweight (shared). If it still loses, evidence is extremely strong.
    """
    device = torch.device(cfg.device)
    # Build split-mode data; start from Phase 1 pretrained model
    split_data = build_split_data(dim, cfg, seed)
    for key in ("init_embedding", "input_ids", "targets", "valid_target_mask"):
        split_data[key] = split_data[key].to(device)

    weights = objective_weights_reweight()
    for key in weights:
        weights[key] = weights[key].to(device)

    torch.manual_seed(seed + 10000)

    # Convert pretrained (shared) model to split mode:
    # - moon_H gets pretrained moon embedding
    # - moon_T gets pretrained moon embedding
    # - cake/banana/fruit/others keep pretrained embeddings
    shared_ids = pretrained_model.E.shape[0]
    # shared vocab: pad, the, a, sun, moon, banana, fruit, cake  (8 tokens)
    # split vocab:  pad, the, a, sun, moon_H, moon_T, banana, fruit, cake (9 tokens)
    split_vocab_size = len(split_data["token_names"])
    split_init = torch.zeros(split_vocab_size, dim, device=device)
    # Map shared indices to split indices (moon_H=4, moon_T=5)
    shared_to_split = [0, 1, 2, 3, 4, 6, 7, 8]  # pad,the,a,sun,moon(H),banana,fruit,cake
    for s_idx, t_idx in enumerate(shared_to_split):
        split_init[t_idx] = pretrained_model.E.data[s_idx].clone()
    # moon_T (index 5) gets the same as moon_H (index 4) 
    split_init[5] = pretrained_model.E.data[4].clone()

    model = TinyAttentionLM(split_init, cfg, seed).to(device)
    # Copy pretrained weight matrices
    model.Wq.data = pretrained_model.Wq.data.clone()
    model.Wk.data = pretrained_model.Wk.data.clone()
    model.Wv.data = pretrained_model.Wv.data.clone()
    model.Wo.data = pretrained_model.Wo.data.clone()

    c = common_direction.to(device)
    # Token indices to project: moon_T=5, cake=8, banana=6, fruit=7
    tail_token_ids = [5, 8, 6, 7]

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: List[Dict[str, object]] = []
    gradient_history: List[Dict[str, object]] = []

    eval_steps = set(range(0, cfg.phase2_steps + 1, cfg.eval_interval))
    eval_steps.add(cfg.phase2_steps)

    for step in range(cfg.phase2_steps + 1):
        if step in eval_steps:
            metrics = evaluate_full(model, split_data, weights, common_direction=c)
            row: Dict[str, object] = {
                "phase": "phase2", "condition": condition_key,
                "dim": dim, "seed": seed, "step": step,
            }
            row.update(metrics)
            history.append(row)
            gradient_history.extend(
                gradient_diagnostics(model, split_data, weights, {
                    "phase": "phase2", "condition": condition_key,
                    "dim": dim, "seed": seed, "step": step,
                })
            )
        if step == cfg.phase2_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(split_data["input_ids"], int(split_data["pad_id"]))
        seq_losses_vals, _ = sequence_losses(
            logits, split_data["targets"], split_data["valid_target_mask"]
        )
        loss = (weights["effective"] * seq_losses_vals).sum()
        loss.backward()
        optimizer.step()

        # ── FORCED ORTHOGONAL PROJECTION ──
        # After optimizer step, remove common-direction component from tail embeddings.
        # This happens BEFORE the next forward pass, so the model cannot use c
        # to represent tail tokens. It must route through orthogonal directions.
        with torch.no_grad():
            for tok_id in tail_token_ids:
                emb = model.E.data[tok_id]
                # Remove common direction projection
                model.E.data[tok_id] = emb - torch.dot(emb, c) * c

    stable_all = first_stable_step(
        history,
        ("the_bayes_gap", "a_bayes_gap", "cake_loss"),
        (cfg.bayes_gap_threshold, cfg.bayes_gap_threshold, cfg.cake_loss_threshold),
        cfg.stable_evals,
    )
    final = history[-1]
    summary: Dict[str, object] = {
        "phase": "phase2", "condition": condition_key,
        "dim": dim, "seed": seed,
        "first_stable_all_step": stable_all,
    }
    for key, value in final.items():
        if key not in {"phase", "condition", "dim", "seed", "step"}:
            summary[f"final_{key}"] = value
    if stable_all is not None:
        conv_row = next(row for row in history if int(row["step"]) == stable_all)
        for key, value in conv_row.items():
            if key not in {"phase", "condition", "dim", "seed", "step"}:
                summary[f"converged_{key}"] = value

    # Also record split_data reference for later use
    return history, gradient_history, summary, split_data


# ═══════════════════════════════════════════════════════════════════════════════
# Run full experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    cfg: Config, dim: int, seed: int, output_dir: Path,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    """Run Phase 1 + Phase 2 for one (dim, seed) combination."""
    device = torch.device(cfg.device)

    # ── Phase 1: Pretrain on high patterns ──
    print(f"  Phase 1: pretraining dim={dim} seed={seed} ...")
    pretrained_model, common_direction, shared_data, pretrain_history, conv_step = (
        pretrain_high_only(cfg, dim, seed)
    )
    if conv_step is not None:
        print(f"    Phase 1 converged at step {conv_step}")
    else:
        print(f"    Phase 1 did not converge within {cfg.pretrain_max_steps} steps")

    # ── Phase 2: Three conditions ──
    all_history: List[Dict[str, object]] = list(pretrain_history)
    all_gradient: List[Dict[str, object]] = []
    all_summary: List[Dict[str, object]] = []

    # 1. Natural (lower bound)
    print(f"    Phase 2 natural ...")
    hist, grad_hist, summary = phase2_natural(
        cfg, pretrained_model, shared_data, dim, seed, "natural"
    )
    all_history.extend(hist)
    all_gradient.extend(grad_hist)
    all_summary.append(summary)

    # 2. Reweight (baseline)
    print(f"    Phase 2 reweight ...")
    hist, grad_hist, summary = phase2_reweight(
        cfg, pretrained_model, shared_data, dim, seed, "reweight"
    )
    all_history.extend(hist)
    all_gradient.extend(grad_hist)
    all_summary.append(summary)

    # 3. Forced Orthogonal (ceiling)
    print(f"    Phase 2 forced_orthogonal ...")
    hist, grad_hist, summary, split_data = phase2_forced_orthogonal(
        cfg, pretrained_model, dim, seed, common_direction, "forced_orthogonal"
    )
    all_history.extend(hist)
    all_gradient.extend(grad_hist)
    all_summary.append(summary)

    return all_history, all_gradient, all_summary


# ═══════════════════════════════════════════════════════════════════════════════
# CSV & Visualization
# ═══════════════════════════════════════════════════════════════════════════════

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


def mean_curve(
    rows: Sequence[Dict[str, object]], dim: int, condition: str, metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    selected = [
        row for row in rows
        if int(row["dim"]) == dim
        and row.get("condition") == condition
        and row.get("phase") == "phase2"
    ]
    by_step: Dict[int, List[float]] = {}
    for row in selected:
        by_step.setdefault(int(row["step"]), []).append(float(row[metric]))
    steps = np.array(sorted(by_step), dtype=np.int64)
    values = np.array([np.mean(by_step[step]) for step in steps], dtype=np.float64)
    return steps, values


def make_plots(
    all_history: Sequence[Dict[str, object]],
    all_gradient: Sequence[Dict[str, object]],
    all_summary: Sequence[Dict[str, object]],
    output_dir: Path,
    primary_dim: int,
) -> None:
    conditions = PHASE2_CONDITIONS
    colors = {"natural": "tab:red", "reweight": "tab:blue", "forced_orthogonal": "tab:orange"}
    labels = {
        "natural": "Natural (Zipf, lower bound)",
        "reweight": "Reweight (baseline)",
        "forced_orthogonal": "Forced Orthogonal (ceiling)",
    }

    # ── Learning curves ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for cond in conditions:
        for axis, metric, title in zip(
            axes,
            ("the_bayes_gap", "a_bayes_gap", "cake_loss"),
            ("the prefix: Bayes gap", "a prefix: Bayes gap", "noun \u2192 cake loss"),
        ):
            steps, values = mean_curve(all_history, primary_dim, cond, metric)
            if len(steps) > 0:
                axis.plot(
                    steps, np.maximum(values, 1e-8),
                    color=colors[cond], label=labels[cond],
                )
            axis.set_title(title)
            axis.set_yscale("log")
            axis.set_xlabel("optimization step")
            axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"dim={primary_dim}: Phase 2 learning curves (from pretrained checkpoint)")
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=180)
    plt.close(fig)

    # ── Spectral curves ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for cond in conditions:
        for axis, metric, title in zip(
            axes,
            ("Bqk_top1_energy", "Bvo_top1_energy", "rep_macro_top1_energy"),
            ("Bqk top-1 energy", "Bvo top-1 energy", "macro representation top-1 energy"),
        ):
            steps, values = mean_curve(all_history, primary_dim, cond, metric)
            if len(steps) > 0:
                axis.plot(steps, values, color=colors[cond], label=labels[cond])
            axis.set_title(title)
            axis.set_ylim(0, 1.02)
            axis.set_xlabel("optimization step")
            axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"dim={primary_dim}: spectral concentration (Phase 2)")
    fig.tight_layout()
    fig.savefig(output_dir / "spectral_curves.png", dpi=180)
    plt.close(fig)

    # ── Gradient contributions ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for cond in conditions:
        selected = [
            row for row in all_gradient
            if int(row["dim"]) == primary_dim
            and row.get("condition") == cond
            and int(row["is_tail"]) == 1
        ]
        by_step_raw: Dict[int, List[float]] = {}
        by_step_weighted: Dict[int, List[float]] = {}
        for row in selected:
            step = int(row["step"])
            by_step_raw.setdefault(step, []).append(float(row["raw_grad_norm"]))
            by_step_weighted.setdefault(step, []).append(float(row["weighted_grad_norm"]))
        steps = sorted(by_step_raw)
        if steps:
            axes[0].plot(
                steps, [np.mean(by_step_raw[step]) for step in steps],
                color=colors[cond], label=labels[cond],
            )
            axes[1].plot(
                steps, [np.mean(by_step_weighted[step]) for step in steps],
                color=colors[cond], label=labels[cond],
            )
    axes[0].set_title("raw per-tail-pattern gradient norm")
    axes[1].set_title("frequency-weighted tail gradient contribution")
    for axis in axes:
        axis.set_yscale("log")
        axis.set_xlabel("optimization step")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"dim={primary_dim}: raw gradient vs optimizer-visible contribution")
    fig.tight_layout()
    fig.savefig(output_dir / "gradient_contributions.png", dpi=180)
    plt.close(fig)

    # ── Representation geometry at convergence ──
    phase2_summary = [row for row in all_summary if row.get("phase") == "phase2"]
    groups = [(cond,) for cond in conditions]
    grouped = [
        [row for row in phase2_summary
         if int(row["dim"]) == primary_dim and row["condition"] == cond]
        for cond in conditions
    ]
    x = np.arange(len(conditions))
    x_labels = ["Natural\n(Zipf)", "Reweight\n(baseline)", "Forced\nOrthogonal"]

    def mean_metric(rows, key):
        values = [float(row[key]) for row in rows if key in row and row[key] is not None]
        return float(np.mean(values)) if values else 0.0

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    width = 0.35

    # Panel 1: tail common energy fraction
    axes[0].bar(x, [mean_metric(rows, "converged_tail_common_energy_fraction") for rows in grouped])
    axes[0].set_title("tail hidden energy\nin common direction")
    axes[0].set_ylim(0.0, 1.05)

    # Panel 2: causal ablation
    axes[1].bar(
        x - width / 2,
        [mean_metric(rows, "converged_common_only_cake_loss") for rows in grouped],
        width, label="common only",
    )
    axes[1].bar(
        x + width / 2,
        [mean_metric(rows, "converged_residual_only_cake_loss") for rows in grouped],
        width, label="residual only",
    )
    axes[1].set_title("causal cake loss\nafter subspace ablation")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=7)

    # Panel 3: static embedding cosine
    axes[2].bar(
        x - width / 2,
        [mean_metric(rows, "converged_cos_cake_moon_tail") for rows in grouped],
        width, label="cake vs moon",
    )
    axes[2].bar(
        x + width / 2,
        [0.5 * (mean_metric(rows, "converged_cos_cake_banana") + mean_metric(rows, "converged_cos_cake_fruit"))
         for rows in grouped],
        width, label="cake vs banana/fruit",
    )
    axes[2].set_title("static embedding cosine")
    axes[2].legend(fontsize=7)

    # Panel 4: convergence step comparison
    stable_steps = [mean_metric(rows, "first_stable_all_step") for rows in grouped]
    bar_colors = [colors[cond] for cond in conditions]
    axes[3].bar(x, stable_steps, color=bar_colors)
    axes[3].set_title("steps to stable convergence\n(lower = faster)")
    axes[3].set_ylabel("step")

    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=8)
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle(f"dim={primary_dim}: representation geometry at convergence (seed mean)")
    fig.tight_layout()
    fig.savefig(output_dir / "representation_geometry.png", dpi=180)
    plt.close(fig)

    # ── Tail projection onto Phase 1 common direction ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for cond in conditions:
        for axis, metric, title in zip(
            axes,
            ("tail_proj_onto_phase1_c", "cake_emb_proj_onto_c"),
            ("tail hidden proj onto Phase-1 c", "cake embedding proj onto Phase-1 c"),
        ):
            selected = [
                row for row in all_history
                if int(row["dim"]) == primary_dim
                and row.get("condition") == cond
                and row.get("phase") == "phase2"
                and metric in row
            ]
            by_step: Dict[int, List[float]] = {}
            for row in selected:
                by_step.setdefault(int(row["step"]), []).append(float(row[metric]))
            if by_step:
                steps = sorted(by_step)
                axis.plot(
                    steps, [np.mean(by_step[s]) for s in steps],
                    color=colors[cond], label=labels[cond],
                )
            axis.set_title(title)
            axis.set_xlabel("optimization step")
            axis.set_ylim(0, 1.05)
            axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"dim={primary_dim}: tail projection onto Phase-1 common direction c")
    fig.tight_layout()
    fig.savefig(output_dir / "common_direction_projection.png", dpi=180)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

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
    a = parser.parse_args()
    return Config(
        dims=a.dims,
        seeds=a.seeds,
        steps=a.steps,
        eval_interval=a.eval_interval,
        stable_evals=a.stable_evals,
        bayes_gap_threshold=a.bayes_gap_threshold,
        cake_loss_threshold=a.cake_loss_threshold,
        lr=a.lr,
        device=a.device,
        output_dir=a.output_dir,
    )


def main() -> None:
    cfg = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_history: List[Dict[str, object]] = []
    all_gradient: List[Dict[str, object]] = []
    all_summary: List[Dict[str, object]] = []

    for dim in cfg.dims:
        for seed in cfg.seeds:
            print(f"\ndim={dim} seed={seed}")
            hist, grad_hist, summaries = run_experiment(cfg, dim, seed, output_dir)
            all_history.extend(hist)
            all_gradient.extend(grad_hist)
            all_summary.extend(summaries)

            # Print per-condition results
            for s in summaries:
                stable = s.get("first_stable_all_step", None)
                cake = float(s.get("final_cake_loss", float("nan")))
                print(f"  {s['condition']:20s}  stable_all={stable}  final_cake_loss={cake:.4g}")

    # Write CSVs
    write_csv(output_dir / "history.csv", all_history)
    write_csv(output_dir / "gradient_history.csv", all_gradient)
    write_csv(output_dir / "summary.csv", all_summary)

    with (output_dir / "config.json").open("w") as handle:
        json.dump(asdict(cfg), handle, indent=2)

    # Generate plots
    make_plots(all_history, all_gradient, all_summary, output_dir, primary_dim=cfg.dims[0])

    print(f"\n✅ Results written to {output_dir}")


if __name__ == "__main__":
    main()
