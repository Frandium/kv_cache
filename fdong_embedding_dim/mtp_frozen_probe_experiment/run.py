from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from .data import make_cartesian_patterns, make_compositional_patterns
from .experiment import (
    evaluate_next_token_positions,
    evaluate_offsets,
    evaluate_probe,
    model_metadata,
    prefix_spectrum,
    train_frozen_probe,
    train_model,
)


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_str_list(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbones", default="linear,mlp,attention")
    parser.add_argument("--hidden-sizes", default="2,4,8")
    parser.add_argument("--seeds", default="971,972,973")
    parser.add_argument("--num-prefixes", type=int, default=8)
    parser.add_argument("--num-bones", type=int, default=8)
    parser.add_argument("--holdout-stride", type=int, default=4)
    parser.add_argument("--dataset", choices=("lookup", "compositional"), default="compositional")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=20260705)
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--probe-steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--probe-learning-rate", type=float, default=3e-2)
    parser.add_argument("--probe-kinds", default="linear,mlp")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "lookup":
        data = make_cartesian_patterns(
            num_prefixes=args.num_prefixes,
            num_bones=args.num_bones,
            holdout_stride=args.holdout_stride,
        )
    else:
        data = make_compositional_patterns(
            num_x=args.num_prefixes,
            num_y=args.num_prefixes,
            num_bones=args.num_bones,
            test_fraction=args.test_fraction,
            split_seed=args.split_seed,
        )
    device = torch.device(args.device)
    records: list[dict] = []

    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    config["mtp_convention"] = "mtp=1 is NTP; mtp=3 predicts offsets 1,2,3"
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "vocabulary.json").write_text(
        json.dumps(list(data.token_names), indent=2), encoding="utf-8"
    )

    for backbone_kind in parse_str_list(args.backbones):
        for hidden_size in parse_int_list(args.hidden_sizes):
            for seed in parse_int_list(args.seeds):
                for mtp in (1, 3):
                    run_name = f"{backbone_kind}_d{hidden_size}_seed{seed}_mtp{mtp}"
                    model, train_history = train_model(
                        data=data,
                        backbone_kind=backbone_kind,
                        hidden_size=hidden_size,
                        mtp=mtp,
                        steps=args.train_steps,
                        learning_rate=args.learning_rate,
                        seed=seed,
                        device=device,
                    )
                    train_tokens = data.sequences[data.train_mask].to(device)
                    test_tokens = data.sequences[data.test_mask].to(device)
                    record = {
                        "run_name": run_name,
                        "seed": seed,
                        **model_metadata(model),
                        "train_history": train_history,
                        "train_offsets": evaluate_offsets(model, train_tokens),
                        "test_offsets": evaluate_offsets(model, test_tokens),
                        "train_ntp_positions": evaluate_next_token_positions(model, train_tokens),
                        "test_ntp_positions": evaluate_next_token_positions(model, test_tokens),
                        "prefix_spectrum": prefix_spectrum(model, data, device),
                        "probes": {},
                    }
                    for probe_kind in parse_str_list(args.probe_kinds):
                        probe, probe_history = train_frozen_probe(
                            model=model,
                            data=data,
                            probe_kind=probe_kind,
                            steps=args.probe_steps,
                            learning_rate=args.probe_learning_rate,
                            seed=seed + 10_000,
                            device=device,
                        )
                        record["probes"][probe_kind] = {
                            "history": probe_history,
                            "train": evaluate_probe(
                                model, probe, train_tokens, data.probe_position, data.target_position
                            ),
                            "test": evaluate_probe(
                                model, probe, test_tokens, data.probe_position, data.target_position
                            ),
                        }
                    if args.save_checkpoints:
                        torch.save(model.state_dict(), output_dir / f"{run_name}.pt")
                    records.append(record)
                    (output_dir / "results.json").write_text(
                        json.dumps(records, indent=2), encoding="utf-8"
                    )
                    print(
                        run_name,
                        "linear_probe_test_acc=",
                        f"{record['probes']['linear']['test']['accuracy']:.3f}",
                        "effective_rank=",
                        f"{record['prefix_spectrum']['effective_rank']:.3f}",
                        flush=True,
                    )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_name",
                "backbone_kind",
                "hidden_size",
                "seed",
                "mtp",
                "suffix_ntp_test_accuracy",
                "native_offset3_test_accuracy",
                "linear_probe_test_accuracy",
                "mlp_probe_test_accuracy",
                "top1_energy_fraction",
                "effective_rank",
                "stable_rank",
            ],
        )
        writer.writeheader()
        for record in records:
            offset3 = record["test_offsets"].get("offset_3", {})
            writer.writerow(
                {
                    "run_name": record["run_name"],
                    "backbone_kind": record["backbone_kind"],
                    "hidden_size": record["hidden_size"],
                    "seed": record["seed"],
                    "mtp": record["mtp"],
                    "suffix_ntp_test_accuracy": record["test_ntp_positions"][
                        f"position_{data.target_position - 1}_to_{data.target_position}"
                    ]["accuracy"],
                    "native_offset3_test_accuracy": offset3.get("accuracy", ""),
                    "linear_probe_test_accuracy": record["probes"]["linear"]["test"]["accuracy"],
                    "mlp_probe_test_accuracy": record["probes"]["mlp"]["test"]["accuracy"],
                    "top1_energy_fraction": record["prefix_spectrum"]["top1_energy_fraction"],
                    "effective_rank": record["prefix_spectrum"]["effective_rank"],
                    "stable_rank": record["prefix_spectrum"]["stable_rank"],
                }
            )


if __name__ == "__main__":
    main()
