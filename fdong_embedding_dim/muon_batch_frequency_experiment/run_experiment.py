#!/usr/bin/env python3
"""Controlled Muon x frequency x batch-coverage experiment.

The experiment trains one dense matrix on fixed orthogonal feature directions.
It compares exact population gradients with stochastic batches under AdamW and
canonical momentum-plus-Newton-Schulz Muon updates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    num_features: int = 16
    num_common: int = 4
    common_mass: float = 0.90
    steps: int = 400
    stable_window: int = 10
    muon_momentum: float = 0.95
    ns_steps: int = 5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_features(n: int, seed: int = 12345) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(n, n, generator=gen))
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    return q * signs.unsqueeze(0)


def make_probs(cfg: Config, distribution: str) -> torch.Tensor:
    n, nc = cfg.num_features, cfg.num_common
    if distribution == "uniform":
        return torch.full((n,), 1.0 / n)
    if distribution == "zipf90":
        p = torch.empty(n)
        p[:nc] = cfg.common_mass / nc
        p[nc:] = (1.0 - cfg.common_mass) / (n - nc)
        return p
    raise ValueError(distribution)


def loss_factors(probs: torch.Tensor, balanced: bool) -> torch.Tensor:
    if not balanced:
        return torch.ones_like(probs)
    return 1.0 / (len(probs) * probs)


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Reference NS5 polynomial used by Muon, evaluated in float32 here."""
    if g.ndim != 2:
        raise ValueError(f"Muon diagnostic expects a 2D matrix, got {g.shape}")
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


def spectrum_metrics(matrix: torch.Tensor, prefix: str) -> Dict[str, float]:
    s = torch.linalg.svdvals(matrix.float())
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


def exact_gradient(
    w: torch.Tensor,
    x: torch.Tensor,
    probs: torch.Tensor,
    balanced: bool,
) -> torch.Tensor:
    logits = x @ w
    err = logits.softmax(dim=-1) - torch.eye(len(probs))
    coeff = probs * loss_factors(probs, balanced)
    return x.T @ (coeff[:, None] * err)


def sampled_gradient(
    w: torch.Tensor,
    x: torch.Tensor,
    probs: torch.Tensor,
    balanced: bool,
    batch_size: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.multinomial(probs, batch_size, replacement=True, generator=generator)
    logits = x[idx] @ w
    err = logits.softmax(dim=-1) - F.one_hot(idx, num_classes=len(probs)).float()
    factors = loss_factors(probs, balanced)[idx]
    grad = x[idx].T @ (factors[:, None] * err) / batch_size
    return grad, idx


def evaluate(w: torch.Tensor, x: torch.Tensor, probs: torch.Tensor, cfg: Config) -> Dict[str, float]:
    logits = x @ w
    labels = torch.arange(cfg.num_features)
    losses = F.cross_entropy(logits, labels, reduction="none")
    correct = logits.argmax(dim=-1).eq(labels).float()
    nc = cfg.num_common
    metrics = {
        "population_loss": float((probs * losses).sum()),
        "macro_loss": float(losses.mean()),
        "common_loss": float(losses[:nc].mean()),
        "tail_loss": float(losses[nc:].mean()),
        "common_accuracy": float(correct[:nc].mean()),
        "tail_accuracy": float(correct[nc:].mean()),
        "all_accuracy": float(correct.mean()),
    }
    metrics.update(spectrum_metrics(w, "parameter"))
    return metrics


def first_window_step(history: Sequence[Dict[str, float]], key: str, window: int) -> Optional[int]:
    values = [row[key] >= 1.0 - 1e-12 for row in history]
    for i in range(0, len(values) - window + 1):
        if all(values[i : i + window]):
            return int(history[i]["step"])
    return None


def train_one(
    cfg: Config,
    x: torch.Tensor,
    distribution: str,
    optimizer_name: str,
    balanced: bool,
    batch_size: int,
    lr: float,
    seed: int,
    steps: Optional[int] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
    steps = steps or cfg.steps
    set_seed(seed)
    probs = make_probs(cfg, distribution)
    gen = torch.Generator().manual_seed(100_000 + seed)
    w = 0.01 * torch.randn(cfg.num_features, cfg.num_features)
    opt = ManualAdamW(w.shape, cfg) if optimizer_name == "adam" else ManualMuon(w.shape, cfg)
    history: List[Dict[str, float]] = []

    initial = evaluate(w, x, probs, cfg)
    history.append({"step": 0, **initial})
    for step in range(1, steps + 1):
        if batch_size == 0:
            grad = exact_gradient(w, x, probs, balanced)
            unique_fraction = 1.0
            tail_unique_fraction = 1.0
        else:
            grad, idx = sampled_gradient(w, x, probs, balanced, batch_size, gen)
            unique_fraction = len(idx.unique()) / cfg.num_features
            tail_seen = idx[idx >= cfg.num_common].unique()
            tail_unique_fraction = len(tail_seen) / (cfg.num_features - cfg.num_common)
        update = opt.update(grad)
        with torch.no_grad():
            w -= lr * update
        metrics = evaluate(w, x, probs, cfg)
        functional_update = x @ update
        common_update = functional_update[: cfg.num_common].norm(dim=1).mean()
        tail_update = functional_update[cfg.num_common :].norm(dim=1).mean()
        row = {
            "step": step,
            **metrics,
            **spectrum_metrics(grad, "gradient"),
            **spectrum_metrics(update, "update"),
            "common_function_update_norm": float(common_update),
            "tail_function_update_norm": float(tail_update),
            "tail_to_common_update_ratio": float(tail_update / (common_update + 1e-12)),
            "batch_unique_fraction": float(unique_fraction),
            "batch_tail_unique_fraction": float(tail_unique_fraction),
        }
        history.append(row)

    common_step = first_window_step(history, "common_accuracy", cfg.stable_window)
    tail_step = first_window_step(history, "tail_accuracy", cfg.stable_window)
    all_step = first_window_step(history, "all_accuracy", cfg.stable_window)
    summary: Dict[str, object] = {
        "distribution": distribution,
        "optimizer": optimizer_name,
        "loss": "balanced" if balanced else "raw",
        "batch": "population" if batch_size == 0 else str(batch_size),
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "common_stable_step": common_step,
        "tail_stable_step": tail_step,
        "all_stable_step": all_step,
        "tail_common_step_ratio": (
            float(tail_step / common_step) if tail_step is not None and common_step not in (None, 0) else None
        ),
        **{f"final_{k}": v for k, v in history[-1].items() if k != "step"},
    }
    return history, summary


def select_learning_rates(
    cfg: Config,
    x: torch.Tensor,
    seeds: Sequence[int],
    adam_lrs: Sequence[float],
    muon_lrs: Sequence[float],
    steps: int,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    selected: Dict[str, float] = {}
    evidence: List[Dict[str, object]] = []
    for optimizer_name, candidates in (("adam", adam_lrs), ("muon", muon_lrs)):
        best_score, best_lr = (float("inf"), float("inf")), None
        for lr in candidates:
            stable_scores = []
            final_losses = []
            for seed in seeds:
                _, summary = train_one(
                    cfg, x, "uniform", optimizer_name, False, 0, lr, seed, steps=steps
                )
                stable = summary["all_stable_step"]
                stable_score = float(stable) if stable is not None else float(steps + 1)
                stable_scores.append(stable_score)
                final_losses.append(float(summary["final_macro_loss"]))
            mean_stable = float(np.mean(stable_scores))
            mean_final_loss = float(np.mean(final_losses))
            score = (mean_stable, mean_final_loss)
            evidence.append(
                {
                    "optimizer": optimizer_name,
                    "lr": lr,
                    "mean_stable_step_score": mean_stable,
                    "mean_final_macro_loss": mean_final_loss,
                }
            )
            if score < best_score:
                best_score, best_lr = score, lr
        assert best_lr is not None
        selected[optimizer_name] = float(best_lr)
    return selected, evidence


def estimator_diagnostics(
    cfg: Config,
    x: torch.Tensor,
    batch_sizes: Sequence[int],
    mc_samples: int,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    set_seed(777)
    w = 0.01 * torch.randn(cfg.num_features, cfg.num_features)
    uniform = make_probs(cfg, "uniform")
    zipf = make_probs(cfg, "zipf90")
    objective_error = float(
        (exact_gradient(w, x, zipf, True) - exact_gradient(w, x, uniform, False)).norm()
        / (exact_gradient(w, x, uniform, False).norm() + 1e-12)
    )
    rows: List[Dict[str, object]] = []
    for distribution in ("uniform", "zipf90"):
        probs = make_probs(cfg, distribution)
        for balanced in (False, True):
            global_grad = exact_gradient(w, x, probs, balanced)
            global_muon = zeropower_via_newtonschulz5(global_grad, cfg.ns_steps)
            for batch_size in batch_sizes:
                gen = torch.Generator().manual_seed(9000 + batch_size + (100 if balanced else 0))
                raw_samples, muon_samples = [], []
                tail_coverages = []
                for _ in range(mc_samples):
                    grad, idx = sampled_gradient(w, x, probs, balanced, batch_size, gen)
                    raw_samples.append(grad)
                    muon_samples.append(zeropower_via_newtonschulz5(grad, cfg.ns_steps))
                    tail_coverages.append(
                        len(idx[idx >= cfg.num_common].unique()) / (cfg.num_features - cfg.num_common)
                    )
                raw_stack = torch.stack(raw_samples)
                muon_stack = torch.stack(muon_samples)
                raw_mean = raw_stack.mean(dim=0)
                muon_mean = muon_stack.mean(dim=0)
                raw_bias = float((raw_mean - global_grad).norm() / (global_grad.norm() + 1e-12))
                muon_bias = float((muon_mean - global_muon).norm() / (global_muon.norm() + 1e-12))
                raw_rms = float(
                    ((raw_stack - global_grad).square().sum(dim=(1, 2)).mean().sqrt())
                    / (global_grad.norm() + 1e-12)
                )
                muon_rms = float(
                    ((muon_stack - global_muon).square().sum(dim=(1, 2)).mean().sqrt())
                    / (global_muon.norm() + 1e-12)
                )
                rows.append(
                    {
                        "distribution": distribution,
                        "loss": "balanced" if balanced else "raw",
                        "batch_size": batch_size,
                        "mc_samples": mc_samples,
                        "mean_tail_feature_coverage": float(np.mean(tail_coverages)),
                        "raw_relative_bias": raw_bias,
                        "raw_relative_rms_error": raw_rms,
                        "muon_relative_bias": muon_bias,
                        "muon_relative_rms_error": muon_rms,
                    }
                )
    return rows, {"balanced_objective_gradient_relative_error": objective_error}


def algorithm_check(cfg: Config) -> Dict[str, object]:
    diag = torch.diag(torch.tensor([10.0, 1.0, 0.1, 0.0]))
    out = zeropower_via_newtonschulz5(diag, cfg.ns_steps)
    return {
        "input_singular_values": torch.linalg.svdvals(diag).tolist(),
        "ns5_singular_values": torch.linalg.svdvals(out).tolist(),
        "zero_direction_abs": float(out[-1, -1].abs()),
    }


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
        ("adam", "balanced", "16"),
        ("muon", "raw", "population"),
        ("muon", "balanced", "16"),
    ]
    for ax, (optimizer, loss_name, batch) in zip(axes.reshape(-1), panels):
        selected = [
            r for r in history_rows
            if r["distribution"] == "zipf90" and r["optimizer"] == optimizer
            and r["loss"] == loss_name and r["batch"] == batch
        ]
        if not selected:
            ax.axis("off")
            continue
        steps = sorted({int(r["step"]) for r in selected})
        for metric, label in (("common_accuracy", "common"), ("tail_accuracy", "tail")):
            means = []
            for step in steps:
                vals = [float(r[metric]) for r in selected if int(r["step"]) == step]
                means.append(np.mean(vals))
            ax.plot(steps, means, label=label, linewidth=2)
        ax.set_title(f"{optimizer} / {loss_name} / {batch}")
        ax.set_xlabel("step")
        ax.set_ylabel("mean accuracy")
        ax.set_xlim(-2, 100)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("90/10 data: common and tail accuracy across seeds")
    fig.tight_layout()
    fig.savefig(outdir / "learning_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    zipf = [r for r in summary_rows if r["distribution"] == "zipf90"]
    labels = sorted({f'{r["optimizer"]}_{r["loss"]}' for r in zipf})
    batches = ["population", "64", "16"]
    xloc = np.arange(len(batches))
    for label in labels:
        medians = []
        for batch in batches:
            vals = [
                float(r["all_stable_step"])
                for r in zipf
                if f'{r["optimizer"]}_{r["loss"]}' == label and r["batch"] == batch
                and r["all_stable_step"] is not None
            ]
            medians.append(np.median(vals) if vals else np.nan)
        ax.plot(xloc, medians, marker="o", linewidth=2, label=label)
    ax.set_xticks(xloc, batches)
    ax.set_xlabel("batch regime")
    ax.set_ylabel("median stable all-feature step")
    ax.set_title("90/10 data: convergence cost as batch coverage decreases")
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
    parser.add_argument("--adam_lrs", default="0.003,0.01,0.03,0.1")
    parser.add_argument("--muon_lrs", default="0.003,0.01,0.03,0.1")
    parser.add_argument("--mc_samples", type=int, default=400)
    args = parser.parse_args()

    cfg = Config(steps=args.steps)
    seeds = parse_int_list(args.seeds)
    batch_sizes = parse_int_list(args.batch_sizes)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    x = make_features(cfg.num_features)

    selected_lrs, lr_evidence = select_learning_rates(
        cfg,
        x,
        seeds,
        parse_float_list(args.adam_lrs),
        parse_float_list(args.muon_lrs),
        args.lr_search_steps,
    )
    print("Selected learning rates:", selected_lrs, flush=True)

    all_history: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    conditions = [
        ("uniform", "adam", False),
        ("uniform", "muon", False),
        ("zipf90", "adam", False),
        ("zipf90", "adam", True),
        ("zipf90", "muon", False),
        ("zipf90", "muon", True),
    ]
    for distribution, optimizer_name, balanced in conditions:
        for batch_size in [0, *batch_sizes]:
            for seed in seeds:
                history, summary = train_one(
                    cfg,
                    x,
                    distribution,
                    optimizer_name,
                    balanced,
                    batch_size,
                    selected_lrs[optimizer_name],
                    seed,
                )
                meta = {
                    "distribution": distribution,
                    "optimizer": optimizer_name,
                    "loss": "balanced" if balanced else "raw",
                    "batch": "population" if batch_size == 0 else str(batch_size),
                    "batch_size": batch_size,
                    "lr": selected_lrs[optimizer_name],
                    "seed": seed,
                }
                all_history.extend([{**meta, **row} for row in history])
                summaries.append(summary)
                print(
                    f'{distribution:7s} {optimizer_name:4s} {meta["loss"]:8s} '
                    f'{meta["batch"]:10s} seed={seed} all={summary["all_stable_step"]} '
                    f'common={summary["common_stable_step"]} tail={summary["tail_stable_step"]}',
                    flush=True,
                )

    diagnostics, objective_check = estimator_diagnostics(cfg, x, batch_sizes, args.mc_samples)
    algo_check = algorithm_check(cfg)
    write_csv(outdir / "history.csv", all_history)
    write_csv(outdir / "summary.csv", summaries)
    write_csv(outdir / "lr_selection.csv", lr_evidence)
    write_csv(outdir / "estimator_diagnostics.csv", diagnostics)
    payload = {
        "config": asdict(cfg),
        "selected_lrs": selected_lrs,
        "objective_check": objective_check,
        "algorithm_check": algo_check,
        "summaries": summaries,
        "estimator_diagnostics": diagnostics,
    }
    (outdir / "summary.json").write_text(json.dumps(payload, indent=2))
    plot_results(all_history, summaries, outdir)
    print(f"Wrote results to {outdir}", flush=True)


if __name__ == "__main__":
    main()
