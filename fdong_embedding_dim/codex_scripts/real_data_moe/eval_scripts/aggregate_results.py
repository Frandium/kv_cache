from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .common import load_manifest, read_csv, training_flops, write_csv


def combine_csvs(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(read_csv(path))
    return rows


def smoothed_train_loss(metrics_path: Path, step: int, window: int) -> float:
    values = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if int(record["step"]) <= step:
                values.append(float(record.get("ce_loss", record.get("loss"))))
    if not values:
        raise RuntimeError(f"no metrics at or before step {step}: {metrics_path}")
    return sum(values[-window:]) / min(len(values), window)


def metric_from_task(task_result: dict[str, Any]) -> tuple[str, float] | None:
    preferred = ("acc_norm,none", "acc,none", "exact_match,none", "word_perplexity,none")
    for key in preferred:
        if key in task_result:
            if key == "word_perplexity,none":
                return None
            return key.split(",", 1)[0], float(task_result[key])
    for key, value in task_result.items():
        if "stderr" not in key and isinstance(value, (int, float)):
            return key, float(value)
    return None


def step_dir(root: Path, size: str, method: str, step: int) -> Path:
    return root / size / method / f"step{step:07d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scaling-manifest", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--compact-dir", required=True)
    parser.add_argument("--train-loss-window", type=int, default=100)
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    compact = Path(args.compact_dir)
    compact.mkdir(parents=True, exist_ok=True)
    latest = load_manifest(args.manifest)
    scaling = load_manifest(args.scaling_manifest)
    shutil.copy2(Path(args.manifest).with_suffix(".csv"), compact / "checkpoint_manifest.csv")
    shutil.copy2(
        Path(args.scaling_manifest).with_suffix(".csv"),
        compact / "scaling_checkpoint_manifest.csv",
    )

    routing_dirs = [
        step_dir(raw / "routing" / "v1", row["size"], row["method"], int(row["step"]))
        for row in latest
    ]
    write_csv(
        compact / "expert_load_per_layer.csv",
        combine_csvs([directory / "expert_load.csv" for directory in routing_dirs]),
    )
    write_csv(
        compact / "expert_load_summary.csv",
        combine_csvs([directory / "expert_load_summary.csv" for directory in routing_dirs]),
    )
    write_csv(
        compact / "continuity_by_budget.csv",
        combine_csvs([directory / "continuity_by_budget.csv" for directory in routing_dirs]),
    )

    predictor_dirs = [
        step_dir(raw / "predictability" / "v1", row["size"], row["method"], int(row["step"]))
        for row in latest
    ]
    write_csv(
        compact / "predictability.csv",
        combine_csvs([directory / "predictability.csv" for directory in predictor_dirs]),
    )

    decode_paths = []
    ttft_paths = []
    for row in latest:
        directory = step_dir(
            raw / "swap_latency" / "decode_v1", row["size"], row["method"], int(row["step"])
        )
        ttft_directory = step_dir(
            raw / "ttft" / "v1", row["size"], row["method"], int(row["step"])
        )
        decode_paths.extend(directory / f"k{capacity}.csv" for capacity in (1, 2, 4, 8))
        ttft_paths.extend(ttft_directory / f"k{capacity}.csv" for capacity in (1, 2, 4, 8))

    decode_mean = [row for row in combine_csvs(decode_paths) if row["repeat"] == "mean"]
    decode_denominator = {
        row["size"]: float(row["milliseconds_per_token"])
        for row in decode_mean
        if row["method"] == "baseline" and row["is_unlimited"] == "1"
    }
    for row in decode_mean:
        reference = decode_denominator[row["size"]]
        ratio = float(row["milliseconds_per_token"]) / reference
        row["latency_ratio_vs_same_size_baseline_unlimited"] = ratio
        row["extra_latency_fraction_vs_same_size_baseline_unlimited"] = ratio - 1.0
    write_csv(compact / "ppu_latency_by_budget.csv", decode_mean)

    ttft_mean = [row for row in combine_csvs(ttft_paths) if row["repeat"] == "mean"]
    ttft_denominator = {
        (row["size"], row["cache_state"], row["prompt_length"]): float(row["ttft_milliseconds"])
        for row in ttft_mean
        if row["method"] == "baseline" and row["is_unlimited"] == "1"
    }
    for row in ttft_mean:
        key = (row["size"], row["cache_state"], row["prompt_length"])
        ratio = float(row["ttft_milliseconds"]) / ttft_denominator[key]
        row["ttft_ratio_vs_same_size_baseline_unlimited"] = ratio
        row["extra_ttft_fraction_vs_same_size_baseline_unlimited"] = ratio - 1.0
    write_csv(compact / "ppu_ttft_by_budget.csv", ttft_mean)

    downstream_rows = []
    scaling_rows = []
    flop_rows = []
    for run in scaling:
        size, method, step = run["size"], run["method"], int(run["step"])
        flop_row = {"size": size, "method": method, "step": step, **training_flops(run["metadata"], step)}
        flop_rows.append(flop_row)
        train_loss = smoothed_train_loss(
            Path(run["run_dir"]) / "metrics.jsonl", step, args.train_loss_window
        )
        loss_path = step_dir(raw / "scaling" / "test_loss_v1", size, method, step) / "result.json"
        with loss_path.open("r", encoding="utf-8") as handle:
            loss_payload = json.load(handle)
        if len(loss_payload) != 1:
            raise RuntimeError(f"expected one test-loss result in {loss_path}")
        test_loss = float(next(iter(loss_payload.values()))["test_loss"])

        lm_path = step_dir(raw / "scaling" / "lm_eval_v1", size, method, step) / "result.json"
        with lm_path.open("r", encoding="utf-8") as handle:
            lm_results = json.load(handle)["results"]
        accuracies = []
        for task, task_result in lm_results.items():
            selected = metric_from_task(task_result)
            if selected is None:
                continue
            metric, value = selected
            accuracies.append(value)
            downstream_rows.append(
                {"size": size, "method": method, "step": step, "task": task, "metric": metric, "value": value}
            )
            scaling_rows.append({**flop_row, "metric": f"downstream:{task}:{metric}", "value": value})
        if not accuracies:
            raise RuntimeError(f"no accuracy-like downstream metrics in {lm_path}")
        macro = sum(accuracies) / len(accuracies)
        downstream_rows.append(
            {"size": size, "method": method, "step": step, "task": "macro_average", "metric": "accuracy", "value": macro}
        )
        for metric, value in (
            ("downstream_macro_accuracy", macro),
            ("train_loss", train_loss),
            ("test_loss", test_loss),
        ):
            scaling_rows.append({**flop_row, "metric": metric, "value": value})

    write_csv(compact / "flops.csv", flop_rows)
    write_csv(compact / "downstream_accuracy.csv", downstream_rows)
    write_csv(compact / "scaling_points.csv", scaling_rows)
    print(f"[aggregate] compact CSV files written to {compact}", flush=True)


if __name__ == "__main__":
    main()
