#!/usr/bin/env python3
"""Muon vs Adam on the shared-K trigram tied-attention toy.

This experiment keeps the user-aligned data/model contract:

  - shared-K trigram data: A0,A1->K ; A1,K->A2 ; K,A2->A0 ; A2,A0->A1
  - tied input/output embedding
  - one attention-only layer with residual connection

It compares:

  - optimizer: AdamW all-params vs canonical-hybrid Muon
  - distribution: withK_uniform vs withK_zipf
  - batch regime: exact population vs stochastic minibatch
  - loss: raw CE vs sqrt inverse-target-frequency reweighting
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    dim: int = 4
    theta_deg: float = 12.0
    init_noise: float = 0.005
    residual_alpha: float = 1.0
    use_o_proj: bool = True
    steps: int = 400
    stable_window: int = 10
    reweight_alpha: float = 0.5
    muon_momentum: float = 0.95
    ns_steps: int = 5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rot(v: np.ndarray, degrees: float, dim: int) -> np.ndarray:
    center = np.array(v, dtype=np.float32)
    center /= np.linalg.norm(center)
    perp = np.zeros(dim, dtype=np.float32)
    perp[0 if abs(center[0]) < 0.9 else 1] = 1.0
    perp -= float(np.dot(perp, center)) * center
    perp /= np.linalg.norm(perp)
    theta = math.radians(degrees)
    return (math.cos(theta) * center + math.sin(theta) * perp).astype(np.float32)


def build_data(cfg: Config, condition: str) -> Dict[str, object]:
    centers = {
        "A": np.eye(cfg.dim, dtype=np.float32)[0],
        "B": np.eye(cfg.dim, dtype=np.float32)[1],
        "C": np.eye(cfg.dim, dtype=np.float32)[2],
        "D": np.eye(cfg.dim, dtype=np.float32)[3],
    }
    e_rows: List[np.ndarray] = [np.ones(cfg.dim, dtype=np.float32) / math.sqrt(cfg.dim)]
    token_names: List[str] = ["K"]
    token_groups: List[str] = ["K"]
    group_ids: Dict[str, List[int]] = {}
    for group, center in centers.items():
        ids = []
        for local_i, off in enumerate([0.0, cfg.theta_deg, -cfg.theta_deg]):
            ids.append(len(e_rows))
            e_rows.append(rot(center, off, cfg.dim))
            token_names.append(f"{group}{local_i}")
            token_groups.append(group)
        group_ids[group] = ids

    c1: List[int] = []
    c2: List[int] = []
    targets: List[int] = []
    groups: List[str] = []
    families: List[str] = []
    pattern_names: List[str] = []
    for group in ["A", "B", "C", "D"]:
        i0, i1, i2 = group_ids[group]
        k = 0
        patterns = [
            (i0, i1, k, "to_K", f"{group}0{group}1_K"),
            (i1, k, i2, "from_K", f"{group}1K_{group}2"),
            (k, i2, i0, "from_K", f"K{group}2_{group}0"),
            (i2, i0, i1, "internal", f"{group}2{group}0_{group}1"),
        ]
        for a, b, y, family, name in patterns:
            c1.append(a)
            c2.append(b)
            targets.append(y)
            groups.append(group)
            families.append(family)
            pattern_names.append(name)

    if condition == "withK_uniform":
        group_probs = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    elif condition == "withK_zipf":
        group_probs = {"A": 0.70, "B": 0.10, "C": 0.10, "D": 0.10}
    else:
        raise ValueError(condition)

    example_probs = torch.tensor(
        [group_probs[g] / sum(x == g for x in groups) for g in groups],
        dtype=torch.float32,
    )
    example_probs = example_probs / example_probs.sum()

    target_probs = torch.zeros(len(e_rows), dtype=torch.float32)
    for p, y in zip(example_probs, targets):
        target_probs[y] += float(p)

    return {
        "E0": torch.tensor(np.stack(e_rows), dtype=torch.float32),
        "token_names": token_names,
        "token_groups": token_groups,
        "c1": torch.tensor(c1, dtype=torch.long),
        "c2": torch.tensor(c2, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "groups": groups,
        "families": families,
        "pattern_names": pattern_names,
        "example_probs": example_probs,
        "target_probs": target_probs,
    }


class AttnLM(torch.nn.Module):
    def __init__(self, e0: torch.Tensor, cfg: Config, seed: int):
        super().__init__()
        self.cfg = cfg
        self.E = torch.nn.Parameter(e0.clone())
        eye = torch.eye(cfg.dim, dtype=torch.float32) * 0.1
        gen = torch.Generator().manual_seed(seed + 1729)
        self.Wq = torch.nn.Parameter(eye + cfg.init_noise * torch.randn(cfg.dim, cfg.dim, generator=gen))
        self.Wk = torch.nn.Parameter(eye + cfg.init_noise * torch.randn(cfg.dim, cfg.dim, generator=gen))
        self.Wv = torch.nn.Parameter(eye + cfg.init_noise * torch.randn(cfg.dim, cfg.dim, generator=gen))
        self.Wo = torch.nn.Parameter(eye + cfg.init_noise * torch.randn(cfg.dim, cfg.dim, generator=gen))
        self.scale = math.sqrt(cfg.dim)

    def forward(self, c1: torch.Tensor, c2: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h1 = self.E[c1]
        h2 = self.E[c2]
        q = h2 @ self.Wq.T
        k1 = h1 @ self.Wk.T
        k2 = h2 @ self.Wk.T
        v1 = h1 @ self.Wv.T
        v2 = h2 @ self.Wv.T
        scores = torch.stack([(q * k1).sum(dim=-1), (q * k2).sum(dim=-1)], dim=-1) / self.scale
        attn = F.softmax(scores, dim=-1)
        attn_pre_o = attn[:, 0:1] * v1 + attn[:, 1:2] * v2
        attn_out = attn_pre_o @ self.Wo.T if self.cfg.use_o_proj else attn_pre_o
        final_h = attn_out + self.cfg.residual_alpha * h2
        logits = final_h @ self.E.T
        cache = {
            "final_h": final_h.detach(),
            "attn": attn.detach(),
            "q": q.detach(),
            "k": torch.cat([k1, k2], dim=0).detach(),
            "Bqk": (self.Wq.detach().T @ self.Wk.detach()),
        }
        return logits, cache


def loss_coefficients(data: Dict[str, object], loss_name: str, alpha: float) -> torch.Tensor:
    probs = data["target_probs"]
    targets = data["targets"]
    if loss_name == "raw":
        coeff = torch.ones_like(targets, dtype=torch.float32)
    elif loss_name == "sqrt_reweight":
        coeff = probs[targets].clamp_min(1e-12).pow(-alpha)
        coeff = coeff / (data["example_probs"] * coeff).sum()
    else:
        raise ValueError(loss_name)
    return coeff


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    if g.ndim != 2:
        raise ValueError(f"Muon expects a 2D matrix, got {tuple(g.shape)}")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        poly = b * gram + c * (gram @ gram)
        x = a * x + poly @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class ManualAdamW:
    def __init__(self, shape: torch.Size, cfg: Config):
        self.m = torch.zeros(shape)
        self.v = torch.zeros(shape)
        self.t = 0
        self.cfg = cfg

    def update(self, grad: torch.Tensor) -> torch.Tensor:
        self.t += 1
        b1, b2 = self.cfg.adam_beta1, self.cfg.adam_beta2
        self.m.lerp_(grad, 1 - b1)
        self.v.lerp_(grad.square(), 1 - b2)
        mhat = self.m / (1 - b1**self.t)
        vhat = self.v / (1 - b2**self.t)
        return mhat / (vhat.sqrt() + self.cfg.adam_eps)


class ManualMuon:
    def __init__(self, shape: torch.Size, cfg: Config):
        self.momentum = torch.zeros(shape)
        self.cfg = cfg

    def update(self, grad: torch.Tensor) -> torch.Tensor:
        beta = self.cfg.muon_momentum
        self.momentum.lerp_(grad, 1 - beta)
        nesterov = torch.lerp(grad, self.momentum, beta)
        update = zeropower_via_newtonschulz5(nesterov, self.cfg.ns_steps)
        update *= math.sqrt(max(1.0, update.shape[-2] / update.shape[-1]))
        return update


def make_optimizers(model: AttnLM, optimizer_name: str, cfg: Config) -> Dict[str, object]:
    opts: Dict[str, object] = {}
    if optimizer_name == "adam":
        for name in ["E", "Wq", "Wk", "Wv", "Wo"]:
            opts[name] = ManualAdamW(getattr(model, name).shape, cfg)
        return opts
    if optimizer_name == "muon":
        # Canonical hybrid: embedding/output table stays on AdamW, hidden 2D mats use Muon.
        opts["E"] = ManualAdamW(model.E.shape, cfg)
        for name in ["Wq", "Wk", "Wv", "Wo"]:
            opts[name] = ManualMuon(getattr(model, name).shape, cfg)
        return opts
    raise ValueError(optimizer_name)


def spectrum_metrics(matrix: torch.Tensor, prefix: str) -> Dict[str, float]:
    s = torch.linalg.svdvals(matrix.detach().float())
    energy = s.square()
    total = float(energy.sum())
    if total <= 1e-20:
        return {
            f"{prefix}_top1_energy": 0.0,
            f"{prefix}_effective_rank": 0.0,
            f"{prefix}_condition_nonzero": 0.0,
        }
    p = energy / energy.sum()
    nz = s[s > max(float(s[0]) * 1e-6, 1e-12)]
    condition = float(nz[0] / nz[-1]) if len(nz) else 0.0
    return {
        f"{prefix}_top1_energy": float(p[0]),
        f"{prefix}_effective_rank": float(torch.exp(-(p * (p + 1e-30).log()).sum())),
        f"{prefix}_condition_nonzero": condition,
    }


def sample_indices(
    data: Dict[str, object],
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.multinomial(data["example_probs"], batch_size, replacement=True, generator=generator)


def compute_loss(
    model: AttnLM,
    data: Dict[str, object],
    loss_name: str,
    cfg: Config,
    indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    coeff = loss_coefficients(data, loss_name, cfg.reweight_alpha)
    if indices is None:
        logits, cache = model(data["c1"], data["c2"])
        losses = F.cross_entropy(logits, data["targets"], reduction="none")
        loss = (data["example_probs"] * coeff * losses).sum()
        return loss, cache
    logits, cache = model(data["c1"][indices], data["c2"][indices])
    losses = F.cross_entropy(logits, data["targets"][indices], reduction="none")
    loss = (coeff[indices] * losses).mean()
    return loss, cache


def grad_snapshot(model: AttnLM) -> Dict[str, torch.Tensor]:
    grads: Dict[str, torch.Tensor] = {}
    for name in ["E", "Wq", "Wk", "Wv", "Wo"]:
        param = getattr(model, name)
        grads[name] = param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
    return grads


def evaluate(model: AttnLM, data: Dict[str, object], cfg: Config) -> Dict[str, float]:
    logits, cache = model(data["c1"], data["c2"])
    targets = data["targets"]
    probs = data["example_probs"]
    raw_losses = F.cross_entropy(logits, targets, reduction="none")
    rew_coeff = loss_coefficients(data, "sqrt_reweight", cfg.reweight_alpha)
    preds = logits.argmax(dim=-1)
    correct = preds.eq(targets).float()
    common = torch.tensor([g == "A" for g in data["groups"]], dtype=torch.bool)
    tail = ~common
    internal = torch.tensor([f == "internal" for f in data["families"]], dtype=torch.bool)

    group_accs = []
    for group in ["A", "B", "C", "D"]:
        mask = torch.tensor([g == group for g in data["groups"]], dtype=torch.bool)
        group_accs.append(float(correct[mask].mean()))

    pattern_accs = {}
    for family in ["to_K", "from_K", "internal"]:
        mask = torch.tensor([f == family for f in data["families"]], dtype=torch.bool)
        pattern_accs[f"{family}_accuracy"] = float(correct[mask].mean())
        pattern_accs[f"{family}_loss"] = float(raw_losses[mask].mean().detach())

    metrics = {
        "population_raw_loss": float((probs * raw_losses).sum().detach()),
        "population_reweighted_loss": float((probs * rew_coeff * raw_losses).sum().detach()),
        "all_accuracy": float(correct.mean()),
        "common_accuracy": float(correct[common].mean()),
        "tail_accuracy": float(correct[tail].mean()),
        "internal_accuracy": float(correct[internal].mean()),
        "common_loss": float(raw_losses[common].mean().detach()),
        "tail_loss": float(raw_losses[tail].mean().detach()),
        "A_group_accuracy": group_accs[0],
        "B_group_accuracy": group_accs[1],
        "C_group_accuracy": group_accs[2],
        "D_group_accuracy": group_accs[3],
        "all_groups_full_accuracy": float(min(group_accs) >= 1.0 - 1e-12),
        "all_examples_full_accuracy": float(correct.min() >= 1.0 - 1e-12),
        "Bqk_norm": float(cache["Bqk"].norm()),
        **pattern_accs,
        **spectrum_metrics(model.E, "E"),
        **spectrum_metrics(cache["Bqk"], "Bqk"),
        **spectrum_metrics(cache["q"], "Q_out"),
        **spectrum_metrics(cache["final_h"], "Final_h"),
    }
    return metrics


def first_window_step(history: Sequence[Dict[str, float]], key: str, window: int) -> Optional[int]:
    flags = [float(row[key]) >= 1.0 - 1e-12 for row in history]
    for i in range(0, len(flags) - window + 1):
        if all(flags[i : i + window]):
            return int(history[i]["step"])
    return None


def first_step_stats(
    grads: Dict[str, torch.Tensor],
    updates: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    grad_matrix = torch.cat([grads["Wq"], grads["Wk"], grads["Wv"], grads["Wo"]], dim=0)
    update_matrix = torch.cat([updates["Wq"], updates["Wk"], updates["Wv"], updates["Wo"]], dim=0)
    out = {}
    out.update(spectrum_metrics(grad_matrix, "step1_hidden_grad"))
    out.update(spectrum_metrics(update_matrix, "step1_hidden_update"))
    return out


def train_one(
    cfg: Config,
    condition: str,
    optimizer_name: str,
    loss_name: str,
    batch_size: int,
    lr: float,
    seed: int,
    steps: Optional[int] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
    steps = steps or cfg.steps
    set_seed(seed)
    data = build_data(cfg, condition)
    model = AttnLM(data["E0"], cfg, seed)
    optimizers = make_optimizers(model, optimizer_name, cfg)
    generator = torch.Generator().manual_seed(100_000 + seed + 97 * batch_size)
    history: List[Dict[str, float]] = []
    step1_diag: Dict[str, float] = {}

    history.append({"step": 0, **evaluate(model, data, cfg)})
    for step in range(1, steps + 1):
        model.zero_grad(set_to_none=True)
        if batch_size == 0:
            indices = None
            batch_tail_fraction = 1.0
            batch_internal_fraction = 1.0
            batch_unique_fraction = 1.0
        else:
            indices = sample_indices(data, batch_size, generator)
            tail_mask = torch.tensor([data["groups"][int(i)] != "A" for i in indices], dtype=torch.bool)
            internal_mask = torch.tensor([data["families"][int(i)] == "internal" for i in indices], dtype=torch.bool)
            batch_tail_fraction = float(tail_mask.float().mean()) if len(indices) else 0.0
            batch_internal_fraction = float(internal_mask.float().mean()) if len(indices) else 0.0
            batch_unique_fraction = float(len(indices.unique()) / len(data["targets"]))
        loss, _ = compute_loss(model, data, loss_name, cfg, indices)
        loss.backward()
        grads = grad_snapshot(model)
        updates: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name in ["E", "Wq", "Wk", "Wv", "Wo"]:
                update = optimizers[name].update(grads[name])
                getattr(model, name).sub_(lr * update)
                updates[name] = update
        row = {
            "step": step,
            **evaluate(model, data, cfg),
            "batch_unique_fraction": batch_unique_fraction,
            "batch_tail_fraction": batch_tail_fraction,
            "batch_internal_fraction": batch_internal_fraction,
            **spectrum_metrics(torch.cat([grads["Wq"], grads["Wk"], grads["Wv"], grads["Wo"]], dim=0), "hidden_grad"),
            **spectrum_metrics(torch.cat([updates["Wq"], updates["Wk"], updates["Wv"], updates["Wo"]], dim=0), "hidden_update"),
            **spectrum_metrics(grads["E"], "E_grad"),
            **spectrum_metrics(updates["E"], "E_update"),
        }
        history.append(row)
        if step == 1:
            step1_diag = first_step_stats(grads, updates)

    summary = {
        "condition": condition,
        "optimizer": optimizer_name,
        "loss": loss_name,
        "batch": "population" if batch_size == 0 else str(batch_size),
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "first_stable_all_examples_full_accuracy_step": first_window_step(
            history, "all_examples_full_accuracy", cfg.stable_window
        ),
        "first_stable_all_groups_full_accuracy_step": first_window_step(
            history, "all_groups_full_accuracy", cfg.stable_window
        ),
        "first_stable_internal_accuracy_step": first_window_step(
            history, "internal_accuracy", cfg.stable_window
        ),
        **step1_diag,
        **{f"final_{k}": v for k, v in history[-1].items() if k != "step"},
    }
    return history, summary


def select_learning_rates(
    cfg: Config,
    seeds: Sequence[int],
    adam_lrs: Sequence[float],
    muon_lrs: Sequence[float],
    steps: int,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    selected: Dict[str, float] = {}
    evidence: List[Dict[str, object]] = []
    for optimizer_name, candidates in (("adam", adam_lrs), ("muon", muon_lrs)):
        best_score = (float("inf"), float("inf"))
        best_lr = None
        for lr in candidates:
            final_fail_scores = []
            stable_scores = []
            final_losses = []
            for seed in seeds:
                _, summary = train_one(
                    cfg,
                    "withK_uniform",
                    optimizer_name,
                    "raw",
                    0,
                    lr,
                    seed,
                    steps=steps,
                )
                stable = summary["first_stable_all_groups_full_accuracy_step"]
                final_fail_scores.append(1.0 - float(summary["final_all_examples_full_accuracy"]))
                stable_scores.append(float(stable) if stable is not None else float(steps + 1))
                final_losses.append(float(summary["final_population_raw_loss"]))
            mean_final_fail = float(np.mean(final_fail_scores))
            mean_stable = float(np.mean(stable_scores))
            mean_final_loss = float(np.mean(final_losses))
            score = (mean_final_fail, mean_stable, mean_final_loss)
            evidence.append(
                {
                    "optimizer": optimizer_name,
                    "lr": lr,
                    "mean_final_full_accuracy_fail": mean_final_fail,
                    "mean_stable_group_step": mean_stable,
                    "mean_final_population_raw_loss": mean_final_loss,
                }
            )
            if score < best_score:
                best_score = score
                best_lr = lr
        assert best_lr is not None
        selected[optimizer_name] = float(best_lr)
    return selected, evidence


def estimator_diagnostics(
    cfg: Config,
    batch_sizes: Sequence[int],
    mc_samples: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for condition in ["withK_uniform", "withK_zipf"]:
        for loss_name in ["raw", "sqrt_reweight"]:
            data = build_data(cfg, condition)
            model = AttnLM(data["E0"], cfg, seed=777)
            model.zero_grad(set_to_none=True)
            loss, _ = compute_loss(model, data, loss_name, cfg, indices=None)
            loss.backward()
            global_grads = grad_snapshot(model)
            global_hidden_grad = torch.cat(
                [global_grads["Wq"], global_grads["Wk"], global_grads["Wv"], global_grads["Wo"]],
                dim=0,
            )
            global_hidden_muon = zeropower_via_newtonschulz5(global_hidden_grad, cfg.ns_steps)
            for batch_size in batch_sizes:
                gen = torch.Generator().manual_seed(9000 + batch_size)
                batch_grads = []
                batch_muons = []
                tail_fracs = []
                internal_fracs = []
                for _ in range(mc_samples):
                    idx = sample_indices(data, batch_size, gen)
                    model.zero_grad(set_to_none=True)
                    loss, _ = compute_loss(model, data, loss_name, cfg, indices=idx)
                    loss.backward()
                    grads = grad_snapshot(model)
                    hidden_grad = torch.cat([grads["Wq"], grads["Wk"], grads["Wv"], grads["Wo"]], dim=0)
                    batch_grads.append(hidden_grad)
                    batch_muons.append(zeropower_via_newtonschulz5(hidden_grad, cfg.ns_steps))
                    tail_fracs.append(float(np.mean([data["groups"][int(i)] != "A" for i in idx.tolist()])))
                    internal_fracs.append(float(np.mean([data["families"][int(i)] == "internal" for i in idx.tolist()])))
                grad_stack = torch.stack(batch_grads)
                muon_stack = torch.stack(batch_muons)
                grad_mean = grad_stack.mean(dim=0)
                muon_mean = muon_stack.mean(dim=0)
                rows.append(
                    {
                        "condition": condition,
                        "loss": loss_name,
                        "batch_size": batch_size,
                        "mc_samples": mc_samples,
                        "mean_batch_tail_fraction": float(np.mean(tail_fracs)),
                        "mean_batch_internal_fraction": float(np.mean(internal_fracs)),
                        "hidden_grad_relative_bias": float(
                            (grad_mean - global_hidden_grad).norm() / (global_hidden_grad.norm() + 1e-12)
                        ),
                        "hidden_grad_relative_rms_error": float(
                            ((grad_stack - global_hidden_grad).square().sum(dim=(1, 2)).mean().sqrt())
                            / (global_hidden_grad.norm() + 1e-12)
                        ),
                        "hidden_muon_relative_bias": float(
                            (muon_mean - global_hidden_muon).norm() / (global_hidden_muon.norm() + 1e-12)
                        ),
                        "hidden_muon_relative_rms_error": float(
                            ((muon_stack - global_hidden_muon).square().sum(dim=(1, 2)).mean().sqrt())
                            / (global_hidden_muon.norm() + 1e-12)
                        ),
                    }
                )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(history_rows: Sequence[Dict[str, object]], summary_rows: Sequence[Dict[str, object]], outdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    panels = [
        ("adam", "raw", "population"),
        ("adam", "sqrt_reweight", "16"),
        ("muon", "raw", "population"),
        ("muon", "sqrt_reweight", "16"),
    ]
    for ax, (optimizer, loss_name, batch) in zip(axes.reshape(-1), panels):
        selected = [
            r
            for r in history_rows
            if r["condition"] == "withK_zipf"
            and r["optimizer"] == optimizer
            and r["loss"] == loss_name
            and r["batch"] == batch
        ]
        if not selected:
            ax.axis("off")
            continue
        steps = sorted({int(r["step"]) for r in selected})
        for metric, label in (
            ("common_accuracy", "common"),
            ("tail_accuracy", "tail"),
            ("internal_accuracy", "internal"),
        ):
            means = []
            for step in steps:
                vals = [float(r[metric]) for r in selected if int(r["step"]) == step]
                means.append(np.mean(vals))
            ax.plot(steps, means, label=label, linewidth=2)
        ax.set_title(f"{optimizer} / {loss_name} / {batch}")
        ax.set_xlabel("step")
        ax.set_ylabel("mean accuracy")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("shared-K Zipf trigram: common, tail, and internal accuracy")
    fig.tight_layout()
    fig.savefig(outdir / "learning_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    zipf = [r for r in summary_rows if r["condition"] == "withK_zipf"]
    labels = sorted({f'{r["optimizer"]}_{r["loss"]}' for r in zipf})
    batches = ["population", "64", "16"]
    xloc = np.arange(len(batches))
    for label in labels:
        medians = []
        for batch in batches:
            vals = [
                float(r["first_stable_all_groups_full_accuracy_step"])
                for r in zipf
                if f'{r["optimizer"]}_{r["loss"]}' == label
                and r["batch"] == batch
                and r["first_stable_all_groups_full_accuracy_step"] is not None
            ]
            medians.append(np.median(vals) if vals else np.nan)
        ax.plot(xloc, medians, marker="o", linewidth=2, label=label)
    ax.set_xticks(xloc, batches)
    ax.set_xlabel("batch regime")
    ax.set_ylabel("median stable all-group-full-accuracy step")
    ax.set_title("shared-K Zipf trigram: convergence cost vs batch coverage")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "batch_gap.png", dpi=180)
    plt.close(fig)


def parse_float_list(value: str) -> List[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr_search_steps", type=int, default=200)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--batch_sizes", default="64,16")
    parser.add_argument("--adam_lrs", default="0.01,0.03,0.1,0.18")
    parser.add_argument("--muon_lrs", default="0.003,0.01,0.03,0.1")
    parser.add_argument("--mc_samples", type=int, default=400)
    args = parser.parse_args()

    cfg = Config(steps=args.steps)
    seeds = parse_int_list(args.seeds)
    batch_sizes = parse_int_list(args.batch_sizes)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selected_lrs, lr_evidence = select_learning_rates(
        cfg,
        seeds,
        parse_float_list(args.adam_lrs),
        parse_float_list(args.muon_lrs),
        args.lr_search_steps,
    )
    print(f"Selected learning rates: {selected_lrs}", flush=True)

    conditions = [
        ("withK_uniform", "adam", "raw"),
        ("withK_uniform", "muon", "raw"),
        ("withK_zipf", "adam", "raw"),
        ("withK_zipf", "adam", "sqrt_reweight"),
        ("withK_zipf", "muon", "raw"),
        ("withK_zipf", "muon", "sqrt_reweight"),
    ]

    all_history: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    for condition, optimizer_name, loss_name in conditions:
        for batch_size in [0, *batch_sizes]:
            for seed in seeds:
                history, summary = train_one(
                    cfg,
                    condition,
                    optimizer_name,
                    loss_name,
                    batch_size,
                    selected_lrs[optimizer_name],
                    seed,
                )
                meta = {
                    "condition": condition,
                    "optimizer": optimizer_name,
                    "loss": loss_name,
                    "batch": "population" if batch_size == 0 else str(batch_size),
                    "batch_size": batch_size,
                    "lr": selected_lrs[optimizer_name],
                    "seed": seed,
                }
                all_history.extend([{**meta, **row} for row in history])
                summaries.append(summary)
                print(
                    f'{condition:13s} {optimizer_name:4s} {loss_name:13s} '
                    f'{meta["batch"]:10s} seed={seed} '
                    f'group={summary["first_stable_all_groups_full_accuracy_step"]} '
                    f'internal={summary["first_stable_internal_accuracy_step"]}',
                    flush=True,
                )

    diagnostics = estimator_diagnostics(cfg, batch_sizes, args.mc_samples)
    write_csv(outdir / "history.csv", all_history)
    write_csv(outdir / "summary.csv", summaries)
    write_csv(outdir / "lr_selection.csv", lr_evidence)
    write_csv(outdir / "estimator_diagnostics.csv", diagnostics)
    plot_results(all_history, summaries, outdir)
    payload = {
        "config": asdict(cfg),
        "selected_lrs": selected_lrs,
        "summaries": summaries,
        "estimator_diagnostics": diagnostics,
    }
    (outdir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote results to {outdir}", flush=True)


if __name__ == "__main__":
    main()
