#!/usr/bin/env python3
"""Measure cross-layer alignment of attention-output dominant subspaces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from analyze import build_sequences, choose_device, choose_dtype


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="fdong/Qwen3-0.6B")
    parser.add_argument("--data-dir", default="/Users/bytedance/Desktop/dclm")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-sequences", type=int, default=8)
    parser.add_argument("--tokens-per-sequence", type=int, default=64)
    parser.add_argument("--top-ratio", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--pca-niter", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class AttentionCapture:
    def __init__(self, model, positions: torch.Tensor):
        self.positions = positions
        self.values: dict[int, torch.Tensor] = {}
        self.handles = []
        for layer_idx, layer in enumerate(model.layers):
            self.handles.append(layer.self_attn.register_forward_hook(self._hook(layer_idx)))

    def _hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            sampled = value[0, self.positions.to(value.device)]
            self.values[layer_idx] = sampled.detach().to("cpu", torch.float16)

        return hook

    def clear(self) -> None:
        self.values.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def pca_basis(matrix: torch.Tensor, rank: int, niter: int) -> tuple[torch.Tensor, torch.Tensor]:
    centered = matrix.float() - matrix.float().mean(dim=0, keepdim=True)
    _, singular_values, basis = torch.pca_lowrank(
        centered,
        q=min(rank, centered.shape[0] - 1, centered.shape[1]),
        center=False,
        niter=niter,
    )
    return basis, singular_values


def pairwise_pc1(bases: list[torch.Tensor]) -> np.ndarray:
    vectors = torch.stack([basis[:, 0] / basis[:, 0].norm() for basis in bases])
    return (vectors @ vectors.T).abs().numpy()


def pairwise_subspace(bases: list[torch.Tensor]) -> np.ndarray:
    count = len(bases)
    output = torch.empty((count, count), dtype=torch.float32)
    for i, left in enumerate(bases):
        for j, right in enumerate(bases):
            rank = min(left.shape[1], right.shape[1])
            output[i, j] = (left[:, :rank].T @ right[:, :rank]).square().sum() / rank
    return output.numpy()


def off_diagonal_mean(matrix: np.ndarray) -> float:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(matrix[mask].mean())


def adjacent_mean(matrix: np.ndarray) -> float:
    return float(np.diag(matrix, k=1).mean())


def same_token_cross_layer_cosine(
    matrices: list[torch.Tensor], bases: list[torch.Tensor]
) -> dict[str, float]:
    centered = [matrix.float() - matrix.float().mean(dim=0, keepdim=True) for matrix in matrices]
    projected = [matrix @ basis @ basis.T for matrix, basis in zip(centered, bases)]

    def summarize(values: list[torch.Tensor]) -> tuple[float, float]:
        adjacent = []
        all_pairs = []
        for left_idx, left in enumerate(values):
            for right_idx in range(left_idx + 1, len(values)):
                cosine = F.cosine_similarity(left, values[right_idx], dim=-1).mean()
                all_pairs.append(cosine)
                if right_idx == left_idx + 1:
                    adjacent.append(cosine)
        return float(torch.stack(adjacent).mean()), float(torch.stack(all_pairs).mean())

    full_adjacent, full_all = summarize(centered)
    top_adjacent, top_all = summarize(projected)
    return {
        "full_a_same_token_adjacent_layer_cosine": full_adjacent,
        "full_a_same_token_all_layer_cosine": full_all,
        "top1pct_a_same_token_adjacent_layer_cosine": top_adjacent,
        "top1pct_a_same_token_all_layer_cosine": top_all,
    }


def plot_matrices(pc1: np.ndarray, top: np.ndarray, output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1))
    for axis, matrix, title, label in (
        (axes[0], pc1, "Attention output: PC1 alignment", "Absolute cosine"),
        (axes[1], top, "Attention output: top-1% subspace overlap", "Subspace overlap"),
    ):
        image = axis.imshow(matrix, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_xlabel("Layer")
        axis.set_ylabel("Layer")
        axis.set_title(title)
        axis.set_xticks(range(0, matrix.shape[0], 3))
        axis.set_yticks(range(0, matrix.shape[0], 3))
        fig.colorbar(image, ax=axis, label=label, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    sequences, metadata = build_sequences(
        tokenizer, args.data_dir, args.seq_len, args.num_sequences
    )
    model = AutoModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device).eval()

    positions = torch.linspace(0, args.seq_len - 1, args.tokens_per_sequence).long()
    captured: dict[int, list[torch.Tensor]] = {
        layer_idx: [] for layer_idx in range(len(model.layers))
    }
    hooks = AttentionCapture(model, positions)
    try:
        with torch.no_grad():
            for seq_idx, sequence in enumerate(sequences):
                hooks.clear()
                model(input_ids=sequence.unsqueeze(0).to(device), use_cache=False, return_dict=True)
                for layer_idx in captured:
                    captured[layer_idx].append(hooks.values[layer_idx])
                print(f"[forward] {seq_idx + 1}/{len(sequences)}", flush=True)
    finally:
        hooks.close()

    hidden_size = next(iter(captured[0]))[0].numel()
    rank = max(1, math.ceil(args.top_ratio * hidden_size))
    bases: list[torch.Tensor] = []
    matrices: list[torch.Tensor] = []
    rows = []
    for layer_idx in range(len(model.layers)):
        matrix = torch.cat(captured[layer_idx], dim=0)
        matrices.append(matrix)
        basis, singular_values = pca_basis(matrix, rank, args.pca_niter)
        bases.append(basis)
        centered = matrix.float() - matrix.float().mean(dim=0, keepdim=True)
        rows.append(
            {
                "layer": layer_idx,
                "samples": int(matrix.shape[0]),
                "top_rank": rank,
                "pc1_energy_fraction": float(singular_values[0].square() / centered.square().sum()),
                "top1pct_energy_fraction": float(singular_values.square().sum() / centered.square().sum()),
            }
        )

    pc1 = pairwise_pc1(bases)
    top = pairwise_subspace(bases)
    random_overlap = rank / hidden_size
    summary = {
        "config": vars(args),
        "device": str(device),
        "effective_dtype": str(dtype),
        "hidden_size": hidden_size,
        "top_rank": rank,
        "sampled_tokens": args.num_sequences * args.tokens_per_sequence,
        "documents": metadata,
        "pc1_adjacent_layer_abs_cosine": adjacent_mean(pc1),
        "pc1_all_layer_abs_cosine": off_diagonal_mean(pc1),
        "top1pct_adjacent_layer_overlap": adjacent_mean(top),
        "top1pct_all_layer_overlap": off_diagonal_mean(top),
        "random_subspace_overlap": random_overlap,
        "top1pct_adjacent_overlap_over_random": adjacent_mean(top) / random_overlap,
        "top1pct_all_overlap_over_random": off_diagonal_mean(top) / random_overlap,
    }
    summary.update(same_token_cross_layer_cosine(matrices, bases))

    with (output_dir / "attention_cross_layer_by_layer.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savetxt(output_dir / "attention_pc1_abs_cosine.csv", pc1, delimiter=",")
    np.savetxt(output_dir / "attention_top1pct_subspace_overlap.csv", top, delimiter=",")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_matrices(pc1, top, output_dir / "attention_cross_layer_alignment.png")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
