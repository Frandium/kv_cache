#!/usr/bin/env python3
"""Oracle ceiling for learning tail patterns in common vs residual B_vo subspaces.

Stage 1 learns only the frequent ``the -> {sun, moon}`` pattern.  Stage 2
freezes that checkpoint and gives two matched low-rank branches the same common
input coordinates and the same parameter count.  The only intended difference
is whether their updates are written into the leading or trailing left-singular
subspace of the stage-1 effective value map B_vo = W_o W_v.
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
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-orthogonal-tail-ceiling")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


PATTERNS = (
    ("the", "sun", "pad"),
    ("the", "moon", "pad"),
    ("a", "moon", "cake"),
    ("a", "banana", "cake"),
    ("a", "fruit", "cake"),
)
PATTERN_NAMES = ("the_sun", "the_moon", "a_moon_cake", "a_banana_cake", "a_fruit_cake")
VARIANTS = ("common_oracle", "residual_oracle", "full_output_oracle")


@dataclass(frozen=True)
class Config:
    dims: Tuple[int, ...] = (8, 16)
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    rank: int = 2
    pretrain_steps: int = 400
    stage2_steps: int = 1000
    eval_interval: int = 5
    stable_evals: int = 5
    gap_threshold: float = 0.03
    pretrain_lr: float = 0.03
    lrs: Tuple[float, ...] = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    variants: Tuple[str, ...] = VARIANTS
    init_scale: float = 0.35
    matrix_init_scale: float = 0.08
    device: str = "cpu"
    output_dir: str = "results"


def parse_int_tuple(text: str) -> Tuple[int, ...]:
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def parse_float_tuple(text: str) -> Tuple[float, ...]:
    values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    return values


def parse_str_tuple(text: str) -> Tuple[str, ...]:
    values = tuple(x.strip() for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated names")
    unknown = set(values) - set(VARIANTS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown variants: {sorted(unknown)}")
    return values


def build_data(dim: int, cfg: Config, seed: int, device: torch.device) -> Dict[str, object]:
    names = ("pad", "the", "a", "sun", "moon", "banana", "fruit", "cake")
    token_to_id = {name: i for i, name in enumerate(names)}
    generator = torch.Generator().manual_seed(7919 + seed * 101 + dim * 1009)
    embedding = F.normalize(torch.randn(len(names), dim, generator=generator), dim=-1)
    embedding = embedding * cfg.init_scale
    ids = torch.tensor(
        [[token_to_id[token] for token in pattern] for pattern in PATTERNS],
        dtype=torch.long,
        device=device,
    )
    inputs = ids[:, :-1]
    targets = ids[:, 1:]
    valid = targets.ne(token_to_id["pad"])
    return {
        "names": names,
        "token_to_id": token_to_id,
        "init_embedding": embedding.to(device),
        "inputs": inputs,
        "targets": targets,
        "valid": valid,
        "pad_id": token_to_id["pad"],
        "tail_token_ids": torch.tensor(
            [token_to_id[x] for x in ("a", "moon", "banana", "fruit", "cake")],
            dtype=torch.long,
            device=device,
        ),
    }


class TinyAttentionLM(torch.nn.Module):
    def __init__(self, init_embedding: torch.Tensor, cfg: Config, seed: int):
        super().__init__()
        vocab, dim = init_embedding.shape
        self.E = torch.nn.Parameter(init_embedding.clone())
        generator = torch.Generator().manual_seed(1543 + seed * 103 + dim * 2017)
        eye = torch.eye(dim, device=init_embedding.device)
        self.Wq = torch.nn.Parameter(
            eye + cfg.matrix_init_scale * torch.randn(dim, dim, generator=generator, device=eye.device)
        )
        self.Wk = torch.nn.Parameter(
            eye + cfg.matrix_init_scale * torch.randn(dim, dim, generator=generator, device=eye.device)
        )
        self.Wv = torch.nn.Parameter(
            eye + cfg.matrix_init_scale * torch.randn(dim, dim, generator=generator, device=eye.device)
        )
        self.Wo = torch.nn.Parameter(
            eye + cfg.matrix_init_scale * torch.randn(dim, dim, generator=generator, device=eye.device)
        )
        self.dim = dim
        self.vocab = vocab

    def run_with_embedding(
        self, inputs: torch.Tensor, pad_id: int, embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = embedding[inputs]
        q, k, v = x @ self.Wq.T, x @ self.Wk.T, x @ self.Wv.T
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dim)
        length = inputs.shape[1]
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        scores = scores.masked_fill(inputs.eq(pad_id).unsqueeze(1), float("-inf"))
        attention = F.softmax(scores, dim=-1)
        pooled_x = attention @ x
        attention_out = (attention @ v) @ self.Wo.T
        hidden = x + attention_out
        logits = hidden @ embedding.T
        logits = logits.clone()
        logits[..., pad_id] = -1e9
        return logits, {"hidden": hidden, "attention": attention, "pooled_x": pooled_x}

    def forward(self, inputs: torch.Tensor, pad_id: int):
        return self.run_with_embedding(inputs, pad_id, self.E)


class OracleSubspaceModel(torch.nn.Module):
    def __init__(
        self,
        base: TinyAttentionLM,
        tail_token_ids: torch.Tensor,
        common_input: torch.Tensor,
        output_basis: torch.Tensor,
        common_output: torch.Tensor,
        residual_output: torch.Tensor,
        variant: str,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.base = copy.deepcopy(base)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("tail_token_ids", tail_token_ids.clone())
        self.register_buffer("common_input", common_input.clone())
        self.register_buffer("output_basis", output_basis.clone())
        self.register_buffer("common_output", common_output.clone())
        self.register_buffer("residual_output", residual_output.clone())
        output_rank = output_basis.shape[1]
        input_rank = common_input.shape[1]
        self.A = torch.nn.Parameter(
            torch.zeros(output_rank, input_rank, device=output_basis.device)
        )
        self.Z = torch.nn.Parameter(
            torch.zeros(tail_token_ids.numel(), output_rank, device=output_basis.device)
        )
        self.variant = variant

    def effective_embedding(self) -> torch.Tensor:
        delta = torch.zeros_like(self.base.E)
        delta_rows = self.Z @ self.output_basis.T
        return delta.index_add(0, self.tail_token_ids, delta_rows) + self.base.E

    def forward(self, inputs: torch.Tensor, pad_id: int, tail_gate: bool = True):
        # The oracle gate makes the shared token ``moon`` sense-specific: the
        # tail delta is visible only on tail sequences, while the frozen high
        # task uses exactly the stage-1 tied embedding and logits.
        embedding = self.effective_embedding() if tail_gate else self.base.E
        logits, cache = self.base.run_with_embedding(inputs, pad_id, embedding)
        if tail_gate:
            adapter = ((cache["pooled_x"] @ self.common_input) @ self.A.T) @ self.output_basis.T
            hidden = cache["hidden"] + adapter
            logits = hidden @ embedding.T
            logits = logits.clone()
            logits[..., pad_id] = -1e9
            cache = dict(cache)
            cache["hidden"] = hidden
            cache["adapter"] = adapter
        return logits, cache


def sequence_losses(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    raw = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").reshape_as(targets)
    return (raw * valid.float()).sum(-1) / valid.sum(-1).clamp_min(1)


def tail_metrics(model: OracleSubspaceModel, data: Dict[str, object]) -> Dict[str, float]:
    with torch.no_grad():
        logits, _ = model(data["inputs"][2:], int(data["pad_id"]), tail_gate=True)
        targets, valid = data["targets"][2:], data["valid"][2:]
        raw = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").reshape_as(targets)
        a_gap = raw[:, 0].mean() - math.log(3.0)
        cake_loss = raw[:, 1].mean()
        seq = (raw * valid.float()).sum(-1) / valid.sum(-1)
        high_logits, _ = model(data["inputs"][:2], int(data["pad_id"]), tail_gate=False)
        high_raw = F.cross_entropy(
            high_logits.flatten(0, 1), data["targets"][:2].flatten(), reduction="none"
        ).reshape_as(data["targets"][:2])
        the_gap = high_raw[:, 0].mean() - math.log(2.0)
        embedding_delta = model.effective_embedding() - model.base.E
        common_energy = (embedding_delta @ model.common_output).square().sum()
        residual_energy = (embedding_delta @ model.residual_output).square().sum()
        total_energy = embedding_delta.square().sum().clamp_min(1e-30)
        delta_map = model.output_basis @ model.A @ model.common_input.T
        map_common = (model.common_output.T @ delta_map).square().sum()
        map_residual = (model.residual_output.T @ delta_map).square().sum()
        map_total = delta_map.square().sum().clamp_min(1e-30)
        return {
            "tail_loss": float(seq.mean()),
            "a_bayes_gap": float(a_gap),
            "cake_loss": float(cake_loss),
            "the_retention_gap": float(the_gap),
            "embedding_common_energy_fraction": float(common_energy / total_energy),
            "embedding_residual_energy_fraction": float(residual_energy / total_energy),
            "map_common_energy_fraction": float(map_common / map_total),
            "map_residual_energy_fraction": float(map_residual / map_total),
        }


def first_stable(rows: Sequence[Dict[str, object]], cfg: Config) -> Optional[int]:
    for i in range(len(rows) - cfg.stable_evals + 1):
        window = rows[i : i + cfg.stable_evals]
        if all(
            float(row["a_bayes_gap"]) <= cfg.gap_threshold
            and float(row["cake_loss"]) <= cfg.gap_threshold
            for row in window
        ):
            return int(window[0]["step"])
    return None


def tensor_state(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def max_state_drift(module: torch.nn.Module, reference: Dict[str, torch.Tensor]) -> float:
    return max(float((value.detach() - reference[name]).abs().max()) for name, value in module.state_dict().items())


def pretrain_base(data: Dict[str, object], cfg: Config, dim: int, seed: int) -> TinyAttentionLM:
    model = TinyAttentionLM(data["init_embedding"], cfg, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.pretrain_lr)
    for _ in range(cfg.pretrain_steps):
        logits, _ = model(data["inputs"][:2], int(data["pad_id"]))
        loss = sequence_losses(logits, data["targets"][:2], data["valid"][:2]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def spectral_bases(base: TinyAttentionLM, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bvo = base.Wo @ base.Wv
    u, s, vh = torch.linalg.svd(bvo.detach().float(), full_matrices=True)
    if 2 * rank > base.dim:
        raise ValueError(f"need dim >= 2*rank, got dim={base.dim}, rank={rank}")
    # The treatment is the literal spectral tail: the bottom-r left-singular
    # directions, not merely the next block after the common top-r space.
    return u[:, :rank], vh.T[:, :rank], u[:, -rank:], s


def train_stage2(
    base: TinyAttentionLM,
    data: Dict[str, object],
    cfg: Config,
    dim: int,
    seed: int,
    lr: float,
    variant: str,
    common_output: torch.Tensor,
    common_input: torch.Tensor,
    residual_output: torch.Tensor,
) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[str, float]]:
    if variant == "common_oracle":
        output_basis = common_output
    elif variant == "residual_oracle":
        output_basis = residual_output
    elif variant == "full_output_oracle":
        output_basis = torch.eye(base.dim, device=common_output.device)
    else:
        raise ValueError(variant)
    model = OracleSubspaceModel(
        base,
        data["tail_token_ids"],
        common_input,
        output_basis,
        common_output,
        residual_output,
        variant,
    )
    optimizer = torch.optim.Adam((model.A, model.Z), lr=lr)
    frozen_reference = tensor_state(model.base)
    history: List[Dict[str, object]] = []
    for step in range(cfg.stage2_steps + 1):
        if step % cfg.eval_interval == 0:
            row: Dict[str, object] = {"dim": dim, "seed": seed, "lr": lr, "variant": variant, "step": step}
            row.update(tail_metrics(model, data))
            history.append(row)
        if step == cfg.stage2_steps:
            break
        logits, _ = model(data["inputs"][2:], int(data["pad_id"]), tail_gate=True)
        loss = sequence_losses(logits, data["targets"][2:], data["valid"][2:]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    stable = first_stable(history, cfg)
    final = history[-1]
    summary: Dict[str, object] = {
        "dim": dim,
        "seed": seed,
        "lr": lr,
        "variant": variant,
        "trainable_parameters": model.A.numel() + model.Z.numel(),
        "first_stable_tail_step": stable,
        "success": int(stable is not None),
        "final_tail_loss": final["tail_loss"],
        "final_a_bayes_gap": final["a_bayes_gap"],
        "final_cake_loss": final["cake_loss"],
        "final_the_retention_gap": final["the_retention_gap"],
        "base_max_abs_drift": max_state_drift(model.base, frozen_reference),
        "final_embedding_common_energy_fraction": final["embedding_common_energy_fraction"],
        "final_embedding_residual_energy_fraction": final["embedding_residual_energy_fraction"],
        "final_map_common_energy_fraction": final["map_common_energy_fraction"],
        "final_map_residual_energy_fraction": final["map_residual_energy_fraction"],
    }
    return history, summary, tail_metrics(model, data)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[int, str, float], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((int(row["dim"]), str(row["variant"]), float(row["lr"])), []).append(row)
    result: List[Dict[str, object]] = []
    for (dim, variant, lr), group in sorted(grouped.items()):
        stable = [float(x["first_stable_tail_step"]) for x in group if x["first_stable_tail_step"] not in (None, "")]
        result.append(
            {
                "dim": dim,
                "variant": variant,
                "lr": lr,
                "successes": len(stable),
                "num_seeds": len(group),
                "median_first_stable_tail_step": float(np.median(stable)) if stable else None,
                "mean_final_tail_loss": float(np.mean([float(x["final_tail_loss"]) for x in group])),
                "mean_final_the_retention_gap": float(np.mean([float(x["final_the_retention_gap"]) for x in group])),
            }
        )
    return result


def select_best(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    keys = sorted({(int(x["dim"]), str(x["variant"])) for x in rows})
    for dim, variant in keys:
        candidates = [x for x in rows if int(x["dim"]) == dim and str(x["variant"]) == variant]
        candidates.sort(
            key=lambda x: (
                -int(x["successes"]),
                float("inf") if x["median_first_stable_tail_step"] in (None, "") else float(x["median_first_stable_tail_step"]),
                float(x["mean_final_tail_loss"]),
            )
        )
        result.append(dict(candidates[0]))
    return result


def make_plots(aggregate_rows: Sequence[Dict[str, object]], best_rows: Sequence[Dict[str, object]], output: Path) -> None:
    dims = sorted({int(x["dim"]) for x in aggregate_rows})
    present_variants = [variant for variant in VARIANTS if any(x["variant"] == variant for x in aggregate_rows)]
    markers = {"common_oracle": "o", "residual_oracle": "s", "full_output_oracle": "^"}
    fig, axes = plt.subplots(1, len(dims), figsize=(6 * len(dims), 4), squeeze=False)
    for ax, dim in zip(axes[0], dims):
        for variant in present_variants:
            rows = [x for x in aggregate_rows if int(x["dim"]) == dim and x["variant"] == variant]
            xs = [float(x["lr"]) for x in rows]
            ys = [float(x["median_first_stable_tail_step"]) if x["median_first_stable_tail_step"] not in (None, "") else np.nan for x in rows]
            ax.plot(xs, ys, marker=markers[variant], label=variant)
        ax.set_xscale("log")
        ax.set_xlabel("stage-2 Adam learning rate")
        ax.set_ylabel("median first stable tail step")
        ax.set_title(f"dim={dim}: independent LR sweep")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "lr_sweep.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.75 / len(present_variants)
    x = np.arange(len(dims))
    offsets = (np.arange(len(present_variants)) - (len(present_variants) - 1) / 2) * width
    for offset, variant in zip(offsets, present_variants):
        rows = [next(r for r in best_rows if int(r["dim"]) == dim and r["variant"] == variant) for dim in dims]
        heights = [float(r["median_first_stable_tail_step"]) if r["median_first_stable_tail_step"] not in (None, "") else np.nan for r in rows]
        ax.bar(x + offset, heights, width, label=variant)
    ax.set_xticks(x, [str(dim) for dim in dims])
    ax.set_xlabel("hidden dimension")
    ax.set_ylabel("best-LR median first stable tail step")
    ax.set_title("Gated isolated branches: constrained versus unrestricted output")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "best_lr_comparison.png", dpi=180)
    plt.close(fig)


def smoke_contract(cfg: Config, device: torch.device) -> Dict[str, object]:
    dim, seed = cfg.dims[0], cfg.seeds[0]
    data = build_data(dim, cfg, seed, device)
    base = pretrain_base(data, cfg, dim, seed)
    uc, vc, ur, singular_values = spectral_bases(base, cfg.rank)
    common = OracleSubspaceModel(base, data["tail_token_ids"], vc, uc, uc, ur, "common_oracle")
    residual = OracleSubspaceModel(base, data["tail_token_ids"], vc, ur, uc, ur, "residual_oracle")
    full = OracleSubspaceModel(
        base,
        data["tail_token_ids"],
        vc,
        torch.eye(dim, device=device),
        uc,
        ur,
        "full_output_oracle",
    )
    with torch.no_grad():
        lc, _ = common(data["inputs"], int(data["pad_id"]), tail_gate=True)
        lr, _ = residual(data["inputs"], int(data["pad_id"]), tail_gate=True)
        lf, _ = full(data["inputs"], int(data["pad_id"]), tail_gate=True)
    return {
        "common_tail_initial_logits_max_abs_diff": float((lc - lr).abs().max()),
        "common_full_initial_logits_max_abs_diff": float((lc - lf).abs().max()),
        "common_trainable_parameters": sum(p.numel() for p in common.parameters() if p.requires_grad),
        "residual_trainable_parameters": sum(p.numel() for p in residual.parameters() if p.requires_grad),
        "full_output_trainable_parameters": sum(p.numel() for p in full.parameters() if p.requires_grad),
        "common_residual_basis_max_abs_overlap": float((uc.T @ ur).abs().max()),
        "common_input_orthogonality_max_error": float((vc.T @ vc - torch.eye(cfg.rank, device=device)).abs().max()),
        "bvo_singular_values": [float(x) for x in singular_values],
    }


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=parse_int_tuple, default=Config.dims)
    parser.add_argument("--seeds", type=parse_int_tuple, default=Config.seeds)
    parser.add_argument("--rank", type=int, default=Config.rank)
    parser.add_argument("--pretrain-steps", type=int, default=Config.pretrain_steps)
    parser.add_argument("--stage2-steps", type=int, default=Config.stage2_steps)
    parser.add_argument("--eval-interval", type=int, default=Config.eval_interval)
    parser.add_argument("--stable-evals", type=int, default=Config.stable_evals)
    parser.add_argument("--gap-threshold", type=float, default=Config.gap_threshold)
    parser.add_argument("--pretrain-lr", type=float, default=Config.pretrain_lr)
    parser.add_argument("--lrs", type=parse_float_tuple, default=Config.lrs)
    parser.add_argument("--variants", type=parse_str_tuple, default=Config.variants)
    parser.add_argument("--device", default=Config.device)
    parser.add_argument("--output-dir", default=Config.output_dir)
    args = parser.parse_args()
    return Config(**vars(args))


def main() -> None:
    torch.set_num_threads(1)
    cfg = parse_args()
    device = torch.device(cfg.device)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    contract = smoke_contract(cfg, device)
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    if contract["common_tail_initial_logits_max_abs_diff"] != 0.0:
        raise RuntimeError("common and spectral-tail variants do not start from identical logits")
    if contract["common_full_initial_logits_max_abs_diff"] != 0.0:
        raise RuntimeError("common and full-output variants do not start from identical logits")
    if contract["common_trainable_parameters"] != contract["residual_trainable_parameters"]:
        raise RuntimeError("trainable parameter counts are not matched")

    history: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    for dim in cfg.dims:
        for seed in cfg.seeds:
            data = build_data(dim, cfg, seed, device)
            base = pretrain_base(data, cfg, dim, seed)
            uc, vc, ur, _ = spectral_bases(base, cfg.rank)
            for variant in cfg.variants:
                for lr in cfg.lrs:
                    run_history, summary, _ = train_stage2(
                        base, data, cfg, dim, seed, lr, variant, uc, vc, ur
                    )
                    history.extend(run_history)
                    summaries.append(summary)
                    print(
                        f"dim={dim} seed={seed} variant={variant} lr={lr:g} "
                        f"stable={summary['first_stable_tail_step']} final={summary['final_tail_loss']:.4g}"
                    )
    aggregate_rows = aggregate(summaries)
    best_rows = select_best(aggregate_rows)
    write_csv(output / "history.csv", history)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "aggregate_lr_sweep.csv", aggregate_rows)
    write_csv(output / "best_lr_summary.csv", best_rows)
    make_plots(aggregate_rows, best_rows, output)
    print("best settings")
    for row in best_rows:
        print(row)


if __name__ == "__main__":
    main()
