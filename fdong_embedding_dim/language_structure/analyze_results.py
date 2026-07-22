#!/usr/bin/env python3
"""Summarize real/null hierarchy mining CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percentiles(rows: List[dict], field: str) -> Dict[str, float]:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    if values.size == 0:
        return {}
    return {
        str(percentile): float(np.percentile(values, percentile))
        for percentile in (10, 25, 50, 75, 90, 99)
    }


def composition_class(row: dict, level1: List[dict]) -> str:
    left = level1[int(row["left_pattern_id"])]
    right = level1[int(row["right_pattern_id"])]
    text = "".join(
        (
            left["left_text"],
            left["right_text"],
            right["left_text"],
            right["right_text"],
        )
    )
    if any(character.isdigit() for character in text):
        return "contains_digit"
    if any(character.isalpha() for character in text):
        return "contains_alpha_no_digit"
    return "punctuation_or_space_only"


def class_counts(rows: List[dict], level1: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        name = composition_class(row, level1)
        counts[name] = counts.get(name, 0) + 1
    return counts


def summarize(result_dir: Path, top_n: int) -> dict:
    level1 = read_csv(result_dir / "c4_patterns.csv")
    real_all = read_csv(result_dir / "c8_real_patterns.csv")
    null_all = read_csv(result_dir / "c8_null_patterns.csv")
    real = [row for row in real_all if row["kind"] == "hetero"]
    null = [row for row in null_all if row["kind"] == "hetero"]
    reuse = np.asarray([int(row["parent_reuse"]) for row in level1], dtype=np.int64)

    return {
        "real_hetero_patterns": len(real),
        "null_hetero_patterns": len(null),
        "real_over_null_count_ratio": len(real) / len(null) if null else None,
        "real": {
            "support_percentiles": percentiles(real, "train_support"),
            "npmi_percentiles": percentiles(real, "npmi"),
            "document_coverage_percentiles": percentiles(real, "document_coverage"),
            "composition_classes": class_counts(real, level1),
            f"top_{top_n}_composition_classes": class_counts(real[:top_n], level1),
        },
        "null": {
            "support_percentiles": percentiles(null, "train_support"),
            "npmi_percentiles": percentiles(null, "npmi"),
            "document_coverage_percentiles": percentiles(null, "document_coverage"),
            "composition_classes": class_counts(null, level1),
        },
        "level1_parent_reuse": {
            "patterns": int(reuse.size),
            "at_least_1": int((reuse >= 1).sum()),
            "at_least_2": int((reuse >= 2).sum()),
            "at_least_5": int((reuse >= 5).sum()),
            "percentiles": {
                str(percentile): float(np.percentile(reuse, percentile))
                for percentile in (50, 75, 90, 95, 99)
            }
            if reuse.size
            else {},
            "max": int(reuse.max()) if reuse.size else 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    summary = summarize(result_dir, args.top_n)
    output_path = result_dir / "analysis_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
