from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import (
    checkpoint_metadata,
    parse_flop_targets,
    selected_runs,
    selected_scaling_runs,
    training_flops,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = selected_runs()
    csv_rows = []
    for run in runs:
        metadata = checkpoint_metadata(run["checkpoint"])
        if metadata["step"] != run["step"]:
            raise RuntimeError(f"filename/payload step mismatch: {run['checkpoint']}")
        config = metadata["model_config"]
        train_args = metadata["train_args"]
        row = {
            **run,
            "world_size": metadata["world_size"],
            "hidden_size": config["hidden_size"],
            "num_layers": config["num_hidden_layers"],
            "num_attention_heads": config["num_attention_heads"],
            "num_kv_heads": config["num_key_value_heads"],
            "head_dim": config["head_dim"],
            "num_tail_experts": config["num_tail_experts"],
            "experts_per_token": config["num_experts_per_token"],
            "common_intermediate_size": config["common_intermediate_size"],
            "tail_intermediate_size": config["tail_intermediate_size"],
            "router_input": config["router_input"],
            "sequence_length": train_args.get("sequence_length", ""),
            "micro_batch_size": train_args.get("batch_size", ""),
            "gradient_accumulation": train_args.get("gradient_accumulation", ""),
        }
        run["metadata"] = metadata
        csv_rows.append(row)
        print(
            f"[manifest] {run['size']}/{run['method']} step={run['step']} "
            f"checkpoint={run['checkpoint']}",
            flush=True,
        )

    (output_dir / "checkpoint_manifest.json").write_text(
        json.dumps(runs, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "checkpoint_manifest.csv", csv_rows)

    scaling = selected_scaling_runs(runs, parse_flop_targets())
    scaling_rows = []
    for run in scaling:
        values = training_flops(run["metadata"], int(run["step"]))
        scaling_rows.append(
            {
                "size": run["size"],
                "method": run["method"],
                "run_name": run["run_name"],
                "run_dir": run["run_dir"],
                "step": run["step"],
                "checkpoint": run["checkpoint"],
                "target_flops": run["target_flops"],
                "is_latest": run["is_latest"],
                **values,
            }
        )
        print(
            f"[scaling-manifest] {run['size']}/{run['method']} step={run['step']} "
            f"flops={values['training_flops']:.6e} latest={run['is_latest']}",
            flush=True,
        )
    (output_dir / "scaling_manifest.json").write_text(
        json.dumps(scaling, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "scaling_manifest.csv", scaling_rows)


if __name__ == "__main__":
    main()
