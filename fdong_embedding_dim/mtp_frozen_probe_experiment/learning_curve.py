from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from .data import make_cartesian_patterns, make_compositional_patterns, make_variable_lookup_patterns
from .experiment import (
    evaluate_offsets,
    model_metadata,
    multi_token_loss,
    set_seed,
)
from .model import CausalBackbone, MultiTokenModel


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_str_list(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def flatten_offset_metrics(
    prefix: str, metrics: dict[str, dict[str, float]], mtp: int
) -> dict[str, float | str]:
    row: dict[str, float | str] = {}
    ce_values: list[float] = []
    for offset in range(1, mtp + 1):
        key = f"offset_{offset}"
        values = metrics.get(key)
        if values is None:
            row[f"{prefix}_offset{offset}_ce"] = ""
            row[f"{prefix}_offset{offset}_acc"] = ""
            continue
        ce = values["cross_entropy"]
        row[f"{prefix}_offset{offset}_ce"] = ce
        row[f"{prefix}_offset{offset}_acc"] = values["accuracy"]
        ce_values.append(ce)
    row[f"{prefix}_total_ce"] = sum(ce_values) / len(ce_values) if ce_values else ""
    return row


@torch.no_grad()
def evaluate_bone_length_suffix_metrics(
    model: MultiTokenModel,
    tokens: torch.Tensor,
    bone_lengths: torch.Tensor | None,
    split_prefix: str,
) -> dict[str, float | str]:
    """Measure suffix prediction by bone length for variable-length lookup data.

    `suffix_ntp_L*` uses the offset-1 head at the final bone position.
    `prefix_to_suffix_L*` uses the matching MTP offset head at the prefix
    position when the suffix is inside the model's MTP horizon.
    """
    if bone_lengths is None:
        return {}
    model.eval()
    _, logits = model(tokens)
    row: dict[str, float | str] = {}
    unique_lengths = sorted(int(value) for value in torch.unique(bone_lengths).cpu())
    for bone_length in unique_lengths:
        examples = bone_lengths == bone_length
        suffix_position = bone_length + 1
        final_bone_position = bone_length
        suffix_target = tokens[examples, suffix_position]

        ntp_prediction = logits[0][examples, final_bone_position]
        ntp_ce = torch.nn.functional.cross_entropy(ntp_prediction, suffix_target)
        ntp_acc = (ntp_prediction.argmax(dim=-1) == suffix_target).float().mean()
        row[f"{split_prefix}_suffix_ntp_L{bone_length}_ce"] = float(ntp_ce.cpu())
        row[f"{split_prefix}_suffix_ntp_L{bone_length}_acc"] = float(ntp_acc.cpu())

        suffix_offset = bone_length + 1
        if suffix_offset <= model.mtp:
            prefix_prediction = logits[suffix_offset - 1][examples, 0]
            prefix_ce = torch.nn.functional.cross_entropy(prefix_prediction, suffix_target)
            prefix_acc = (prefix_prediction.argmax(dim=-1) == suffix_target).float().mean()
            row[f"{split_prefix}_prefix_to_suffix_L{bone_length}_ce"] = float(prefix_ce.cpu())
            row[f"{split_prefix}_prefix_to_suffix_L{bone_length}_acc"] = float(prefix_acc.cpu())
        else:
            row[f"{split_prefix}_prefix_to_suffix_L{bone_length}_ce"] = ""
            row[f"{split_prefix}_prefix_to_suffix_L{bone_length}_acc"] = ""
    return row


@torch.no_grad()
def evaluate_curve_row(
    model: MultiTokenModel,
    train_tokens: torch.Tensor,
    test_tokens: torch.Tensor,
    train_loss_mask: torch.Tensor | None,
    test_loss_mask: torch.Tensor | None,
    train_bone_lengths: torch.Tensor | None,
    test_bone_lengths: torch.Tensor | None,
    step: int,
    run_name: str,
    seed: int,
) -> dict[str, float | int | str]:
    train_metrics = evaluate_offsets(model, train_tokens, train_loss_mask)
    test_metrics = evaluate_offsets(model, test_tokens, test_loss_mask)
    row: dict[str, float | int | str] = {
        "run_name": run_name,
        "step": step,
        "seed": seed,
        **model_metadata(model),
    }
    row.update(flatten_offset_metrics("train", train_metrics, model.mtp))
    row.update(flatten_offset_metrics("test", test_metrics, model.mtp))
    row.update(
        evaluate_bone_length_suffix_metrics(model, train_tokens, train_bone_lengths, "train")
    )
    row.update(evaluate_bone_length_suffix_metrics(model, test_tokens, test_bone_lengths, "test"))
    return row


def train_one_curve(
    *,
    data,
    backbone_kind: str,
    hidden_size: int,
    mtp: int,
    seed: int,
    steps: int,
    learning_rate: float,
    log_every: int,
    device: torch.device,
    checkpoint_steps: set[int],
    checkpoint_dir: Path | None,
    run_config: dict,
) -> list[dict[str, float | int | str]]:
    set_seed(seed)
    backbone = CausalBackbone(
        vocab_size=data.vocab_size,
        hidden_size=hidden_size,
        kind=backbone_kind,
        max_seq_len=data.sequences.size(1),
    )
    model = MultiTokenModel(backbone, data.vocab_size, mtp).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    train_tokens = data.sequences[data.train_mask].to(device)
    test_tokens = data.sequences[data.test_mask].to(device)
    train_loss_mask = data.loss_mask[data.train_mask].to(device) if data.loss_mask is not None else None
    test_loss_mask = data.loss_mask[data.test_mask].to(device) if data.loss_mask is not None else None
    train_bone_lengths = (
        data.bone_lengths[data.train_mask].to(device) if data.bone_lengths is not None else None
    )
    test_bone_lengths = (
        data.bone_lengths[data.test_mask].to(device) if data.bone_lengths is not None else None
    )
    run_name = f"{backbone_kind}_d{hidden_size}_seed{seed}_mtp{mtp}"

    def current_row(current_step: int) -> dict[str, float | int | str]:
        return evaluate_curve_row(
            model,
            train_tokens,
            test_tokens,
            train_loss_mask,
            test_loss_mask,
            train_bone_lengths,
            test_bone_lengths,
            current_step,
            run_name,
            seed,
        )

    def save_checkpoint(current_step: int, row: dict[str, float | int | str]) -> None:
        if checkpoint_dir is None or current_step not in checkpoint_steps:
            return
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{run_name}_step{current_step:06d}.pt"
        torch.save(
            {
                "run_name": run_name,
                "step": current_step,
                "seed": seed,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": row,
                "config": run_config,
                "model_metadata": model_metadata(model),
            },
            path,
        )

    first_row = current_row(0)
    save_checkpoint(0, first_row)
    rows = [first_row]
    for step in range(1, steps + 1):
        model.train()
        _, logits = model(train_tokens)
        loss = multi_token_loss(logits, train_tokens, train_loss_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        should_log = step == 1 or step % log_every == 0 or step == steps
        should_checkpoint = checkpoint_dir is not None and step in checkpoint_steps
        if should_log or should_checkpoint:
            row = current_row(step)
            if should_log:
                rows.append(row)
            save_checkpoint(step, row)
    return rows


def write_curve_csv(
    path: Path,
    rows: list[dict[str, float | int | str]],
    max_mtp: int,
    bone_length_values: list[int],
) -> None:
    fieldnames = [
        "run_name",
        "step",
        "seed",
        "backbone_kind",
        "hidden_size",
        "mtp",
        "total_parameters",
        "backbone_parameters",
    ]
    for split in ("train", "test"):
        for offset in range(1, max_mtp + 1):
            fieldnames.extend([f"{split}_offset{offset}_ce", f"{split}_offset{offset}_acc"])
        fieldnames.append(f"{split}_total_ce")
        for bone_length in bone_length_values:
            fieldnames.extend(
                [
                    f"{split}_suffix_ntp_L{bone_length}_ce",
                    f"{split}_suffix_ntp_L{bone_length}_acc",
                    f"{split}_prefix_to_suffix_L{bone_length}_ce",
                    f"{split}_prefix_to_suffix_L{bone_length}_acc",
                ]
            )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset", choices=("lookup", "variable_lookup", "compositional"), default="lookup"
    )
    parser.add_argument("--backbones", default="mlp")
    parser.add_argument("--hidden-sizes", default="4")
    parser.add_argument("--seeds", default="971")
    parser.add_argument("--mtps", default="1,3")
    parser.add_argument("--num-prefixes", type=int, default=8)
    parser.add_argument("--num-bones", type=int, default=8)
    parser.add_argument("--min-bone-length", type=int, default=1)
    parser.add_argument("--max-bone-length", type=int, default=4)
    parser.add_argument("--holdout-stride", type=int, default=4)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=20260705)
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--checkpoint-steps",
        default="",
        help="comma-separated training steps to save, e.g. 0,20,100,500,3000",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="directory for checkpoint .pt files; default disables checkpoint saving",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "lookup":
        data = make_cartesian_patterns(
            num_prefixes=args.num_prefixes,
            num_bones=args.num_bones,
            holdout_stride=args.holdout_stride,
        )
    elif args.dataset == "variable_lookup":
        data = make_variable_lookup_patterns(
            num_prefixes=args.num_prefixes,
            num_bones=args.num_bones,
            min_bone_length=args.min_bone_length,
            max_bone_length=args.max_bone_length,
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

    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    if config["checkpoint_dir"] is not None:
        config["checkpoint_dir"] = str(config["checkpoint_dir"])
    config["loss_weighting"] = "total_ce is the arithmetic mean of valid offset CE losses"
    config["mtp_convention"] = "mtp=1 is NTP; mtp=3 predicts offsets 1,2,3"
    (args.output_dir / "learning_curve_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (args.output_dir / "vocabulary.json").write_text(
        json.dumps(list(data.token_names), indent=2), encoding="utf-8"
    )

    device = torch.device(args.device)
    all_rows: list[dict[str, float | int | str]] = []
    max_mtp = max(parse_int_list(args.mtps))
    checkpoint_steps = set(parse_int_list(args.checkpoint_steps))
    bone_length_values = (
        sorted(int(value) for value in torch.unique(data.bone_lengths).cpu())
        if data.bone_lengths is not None
        else []
    )
    for backbone_kind in parse_str_list(args.backbones):
        for hidden_size in parse_int_list(args.hidden_sizes):
            for seed in parse_int_list(args.seeds):
                for mtp in parse_int_list(args.mtps):
                    rows = train_one_curve(
                        data=data,
                        backbone_kind=backbone_kind,
                        hidden_size=hidden_size,
                        mtp=mtp,
                        seed=seed,
                        steps=args.train_steps,
                        learning_rate=args.learning_rate,
                        log_every=args.log_every,
                        device=device,
                        checkpoint_steps=checkpoint_steps,
                        checkpoint_dir=args.checkpoint_dir,
                        run_config=config,
                    )
                    all_rows.extend(rows)
                    write_curve_csv(
                        args.output_dir / "learning_curve.csv",
                        all_rows,
                        max_mtp,
                        bone_length_values,
                    )
                    last = rows[-1]
                    print(
                        last["run_name"],
                        "step=",
                        last["step"],
                        "train_total_ce=",
                        f"{float(last['train_total_ce']):.4f}",
                        "test_total_ce=",
                        f"{float(last['test_total_ce']):.4f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
