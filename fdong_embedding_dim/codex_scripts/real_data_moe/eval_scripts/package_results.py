from __future__ import annotations

import argparse
import csv
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-bytes", type=int, default=5_000_000)
    args = parser.parse_args()

    compact = Path(args.compact_dir)
    required = (
        "checkpoint_manifest.csv",
        "scaling_checkpoint_manifest.csv",
        "expert_load_per_layer.csv",
        "expert_load_summary.csv",
        "continuity_by_budget.csv",
        "ppu_latency_by_budget.csv",
        "ppu_ttft_by_budget.csv",
        "predictability.csv",
        "flops.csv",
        "downstream_accuracy.csv",
        "scaling_points.csv",
    )
    for name in required:
        path = compact / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing compact result: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            if not next(csv.reader(handle), None):
                raise RuntimeError(f"empty CSV: {path}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for name in required:
            archive.add(compact / name, arcname=name)
    size = output.stat().st_size
    if size > args.max_bytes:
        output.unlink()
        raise RuntimeError(f"result archive would exceed limit: {size} > {args.max_bytes}")
    print(f"[package] {output} bytes={size}", flush=True)


if __name__ == "__main__":
    main()
