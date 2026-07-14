#!/usr/bin/env python3
"""Rebuild the 0717 training and attention-continuity figures."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


STEP_RE = re.compile(
    r"\[step\s+(?P<step>\d+)\]\s+tokens=(?P<tokens>[\d,]+)\s+loss=(?P<loss>[\d.]+)"
)
CONFIG_RE = re.compile(r"^\[config\]\s+(\{.*\})$")
PARAM_RE = re.compile(r"^\[parameters\]\s+(\{.*\})$")
DIST_RE = re.compile(
    r"^\[distributed\]\s+world_size=(?P<world>\d+)\s+global_batch=(?P<batch>[\d,]+)"
    r"\s+tokens_per_step=(?P<tokens>[\d,]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--baseline-log", type=Path, default=Path(__file__).with_name("baseline-train.log"))
    parser.add_argument("--routing-log", type=Path, default=Path(__file__).with_name("routingonly-train.log"))
    parser.add_argument(
        "--band-csv",
        type=Path,
        default=root.parent
        / "fdong_embedding_dim/attention_residual_spectrum_experiment/results_source_own_continuity_seq1024/band_attribution_summary_layers1_27.csv",
    )
    parser.add_argument("--assets-dir", type=Path, default=root / "assets")
    parser.add_argument("--smooth-points", type=int, default=100)
    return parser.parse_args()


def parse_log(path: Path) -> dict:
    by_step: dict[int, dict] = {}
    config = None
    parameters = None
    distributed = None
    for line in path.read_text(errors="replace").splitlines():
        if match := CONFIG_RE.match(line):
            config = json.loads(match.group(1))
        elif match := PARAM_RE.match(line):
            parameters = json.loads(match.group(1))
        elif match := DIST_RE.match(line):
            distributed = {
                "world_size": int(match.group("world")),
                "global_batch": int(match.group("batch").replace(",", "")),
                "tokens_per_step": int(match.group("tokens").replace(",", "")),
            }
        elif match := STEP_RE.search(line):
            step = int(match.group("step"))
            by_step[step] = {
                "step": step,
                "tokens": int(match.group("tokens").replace(",", "")),
                "loss": float(match.group("loss")),
            }
    if not by_step or config is None or parameters is None or distributed is None:
        raise RuntimeError(f"missing records in {path}")
    return {
        "path": str(path),
        "config": config,
        "parameters": parameters,
        "distributed": distributed,
        "records": [by_step[step] for step in sorted(by_step)],
    }


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    cumulative = np.cumsum(values, dtype=np.float64)
    output = np.empty_like(values, dtype=np.float64)
    for idx in range(values.size):
        start = max(0, idx - window + 1)
        total = cumulative[idx] - (cumulative[start - 1] if start > 0 else 0.0)
        output[idx] = total / (idx - start + 1)
    return output


def plot_loss(baseline: dict, routing: dict, common_step: int, output: Path, smooth: int) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Residual routing": "#355C9A", "Attention-Top routing": "#D05A47"}
    fig, axis = plt.subplots(figsize=(9.2, 5.0))
    tail_summary = {}
    for label, parsed in (("Residual routing", baseline), ("Attention-Top routing", routing)):
        records = [record for record in parsed["records"] if record["step"] <= common_step]
        tokens = np.asarray([record["tokens"] for record in records], dtype=np.float64) / 1e9
        losses = np.asarray([record["loss"] for record in records], dtype=np.float64)
        smoothed = rolling_mean(losses, smooth)
        axis.plot(tokens, losses, color=colors[label], alpha=0.08, linewidth=0.7)
        axis.plot(tokens, smoothed, color=colors[label], linewidth=2.1, label=label)
        tail_summary[label] = {
            "last_step": int(records[-1]["step"]),
            "last_tokens": int(records[-1]["tokens"]),
            "last_100_record_mean_loss": float(losses[-100:].mean()),
            "last_500_record_mean_loss": float(losses[-500:].mean()),
        }

    axis.set_title(
        f"Training loss on the shared "
        f"{common_step * baseline['distributed']['tokens_per_step'] / 1e9:.2f}B-token interval"
    )
    axis.set_xlabel("Training tokens (billions)")
    axis.set_ylabel("Next-token training loss")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return tail_summary


def plot_attention_bands(csv_path: Path, output: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(csv_path.open()))
    order = ["0-1%", "1-2%", "2-5%", "5-10%", "10-20%", "20-50%", "50-100%"]

    def band_label(row: dict) -> str:
        for key in ("a_band", "band", "source_band"):
            if key in row:
                label = row[key].replace("top ", "").replace(" ", "").replace("_", "-")
                return label if label.endswith("%") else f"{label}%"
        raise KeyError(f"cannot find band label in {row.keys()}")

    value_key = "a_own_adjacent_cosine"
    values_by_band = {band_label(row): float(row[value_key]) for row in rows}
    values = [values_by_band[label] for label in order]
    full_a = 0.5182718325544287

    fig, axis = plt.subplots(figsize=(8.7, 4.7))
    x = np.arange(len(order))
    axis.plot(x, values, marker="o", linewidth=2.2, markersize=6, color="#D05A47", label="A spectral band")
    axis.axhline(full_a, color="#355C9A", linestyle="--", linewidth=1.8, label=f"Full A ({full_a:.3f})")
    axis.set_xticks(x, order)
    axis.set_xlabel("Spectral band within attention output A")
    axis.set_ylabel("Centered adjacent-token cosine")
    axis.set_ylim(-0.18, 0.72)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("Cross-token continuity is concentrated in A's head subspace")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"bands": order, "values": values, "full_a": full_a, "value_key": value_key}


def main() -> None:
    args = parse_args()
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    baseline = parse_log(args.baseline_log)
    routing = parse_log(args.routing_log)
    common_step = min(baseline["records"][-1]["step"], routing["records"][-1]["step"])

    loss_summary = plot_loss(
        baseline,
        routing,
        common_step,
        args.assets_dir / "0717_attention_top_routing_loss.png",
        args.smooth_points,
    )
    band_summary = plot_attention_bands(
        args.band_csv,
        args.assets_dir / "0717_attention_band_continuity.png",
    )

    parameters = baseline["parameters"]
    active_tail = parameters["tail"] // baseline["config"]["num_tail_experts"]
    active_parameters = (
        parameters["embedding"]
        + parameters["attention"]
        + parameters["common"]
        + parameters["router_norm"]
        + active_tail
    )
    summary = {
        "common_step": common_step,
        "common_tokens": common_step * baseline["distributed"]["tokens_per_step"],
        "baseline": {key: baseline[key] for key in ("config", "parameters", "distributed")},
        "routing": {key: routing[key] for key in ("config", "parameters", "distributed")},
        "active_parameters_including_tied_embedding_head": active_parameters,
        "active_parameter_fraction": active_parameters / parameters["total"],
        "loss_tail": loss_summary,
        "attention_band_continuity": band_summary,
    }
    output = Path(__file__).with_name("0717_result_summary.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
