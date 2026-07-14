#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Optional


RUNS = ("baseline", "proposed_total_matched", "routing_only")
RUN_LABELS = {
    "baseline": "baseline",
    "proposed_total_matched": "proposed",
    "routing_only": "routing_only",
}
SKIP_METRIC_PARTS = ("stderr", "perplexity", "alias", "exact_match_stderr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir",
        default="/mnt/workspace/fmoe_cuda_2b_outputs/lm_eval",
        help="Directory containing *_stepXXXXXXX_results.json files.",
    )
    parser.add_argument(
        "--step",
        default="auto",
        help="Checkpoint step to summarize, e.g. 56000. Default finds latest common step.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix. Default: <eval-dir>/summary_stepXXXXXXX",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Optional comma-separated task order/filter.",
    )
    return parser.parse_args()


def step_from_path(path: Path, run: str) -> Optional[int]:
    match = re.match(rf"{re.escape(run)}_step0*([0-9]+)_results\.json$", path.name)
    return int(match.group(1)) if match else None


def find_common_step(eval_dir: Path) -> int:
    step_sets = []
    for run in RUNS:
        steps = {
            step
            for path in eval_dir.glob(f"{run}_step*_results.json")
            if (step := step_from_path(path, run)) is not None
        }
        if not steps:
            raise FileNotFoundError(f"No result files found for {run} in {eval_dir}")
        step_sets.append(steps)
    common = set.intersection(*step_sets)
    if not common:
        raise FileNotFoundError(f"No common result step found across {', '.join(RUNS)}")
    return max(common)


def result_path(eval_dir: Path, run: str, step: int) -> Path:
    path = eval_dir / f"{run}_step{step:07d}_results.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def metric_name(raw_key: str) -> str:
    # lm-eval usually stores keys like "acc,none" or "acc_norm,none".
    return raw_key.split(",", 1)[0]


def useful_metrics(task_result: Dict[str, object]) -> Dict[str, float]:
    metrics = {}
    for key, value in task_result.items():
        name = metric_name(key)
        if any(part in name for part in SKIP_METRIC_PARTS):
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            metrics[name] = float(value)
    return metrics


def load_results(eval_dir: Path, step: int) -> Dict[str, Dict[str, Dict[str, float]]]:
    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for run in RUNS:
        with result_path(eval_dir, run, step).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        raw_results = payload.get("results", {})
        all_results[run] = {
            task: useful_metrics(task_result)
            for task, task_result in raw_results.items()
            if isinstance(task_result, dict)
        }
    return all_results


def ordered_tasks(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    task_filter: Optional[str],
) -> list[str]:
    if task_filter:
        return [task.strip() for task in task_filter.split(",") if task.strip()]
    tasks = sorted(set.intersection(*(set(all_results[run]) for run in RUNS)))
    return tasks


def choose_metric(
    task: str,
    all_results: Dict[str, Dict[str, Dict[str, float]]],
) -> str:
    candidates = sorted(
        set.intersection(*(set(all_results[run].get(task, {})) for run in RUNS))
    )
    if not candidates:
        raise ValueError(f"No common non-perplexity metric found for task {task}")
    # For tasks with acc and acc_norm, select the metric with the highest mean
    # across the three runs. This keeps one comparable metric per task.
    return max(
        candidates,
        key=lambda metric: sum(all_results[run][task][metric] for run in RUNS) / len(RUNS),
    )


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    headers = [
        "task",
        "metric",
        "baseline",
        "proposed",
        "routing_only",
        "proposed-baseline",
        "routing-baseline",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            handle.write(
                "| "
                + " | ".join(
                    str(row[header]) if not isinstance(row[header], float) else fmt(row[header])
                    for header in headers
                )
                + " |\n"
            )


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    headers = [
        "task",
        "metric",
        "baseline",
        "proposed",
        "routing_only",
        "proposed-baseline",
        "routing-baseline",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    step = find_common_step(eval_dir) if args.step == "auto" else int(args.step)
    all_results = load_results(eval_dir, step)

    rows = []
    for task in ordered_tasks(all_results, args.tasks):
        metric = choose_metric(task, all_results)
        baseline = all_results["baseline"][task][metric]
        proposed = all_results["proposed_total_matched"][task][metric]
        routing = all_results["routing_only"][task][metric]
        rows.append(
            {
                "task": task,
                "metric": metric,
                "baseline": baseline,
                "proposed": proposed,
                "routing_only": routing,
                "proposed-baseline": proposed - baseline,
                "routing-baseline": routing - baseline,
            }
        )

    if rows:
        rows.append(
            {
                "task": "average",
                "metric": "mean_selected",
                "baseline": sum(float(row["baseline"]) for row in rows) / len(rows),
                "proposed": sum(float(row["proposed"]) for row in rows) / len(rows),
                "routing_only": sum(float(row["routing_only"]) for row in rows) / len(rows),
                "proposed-baseline": sum(
                    float(row["proposed-baseline"]) for row in rows
                )
                / len(rows),
                "routing-baseline": sum(float(row["routing-baseline"]) for row in rows)
                / len(rows),
            }
        )

    prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else eval_dir / f"summary_step{step:07d}"
    )
    markdown_path = prefix.with_suffix(".md")
    csv_path = prefix.with_suffix(".csv")
    write_markdown(rows, markdown_path)
    write_csv(rows, csv_path)

    print(f"[summary] step={step}")
    print(f"[summary] markdown={markdown_path}")
    print(f"[summary] csv={csv_path}")
    print(markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
