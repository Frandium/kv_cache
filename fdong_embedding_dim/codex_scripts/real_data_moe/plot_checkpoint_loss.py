from __future__ import annotations

import argparse
import gc
import glob
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from .analysis_utils import fixed_token_sequences, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--proposed-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-sequences", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=256)
    return parser.parse_args()


@torch.no_grad()
def evaluate_checkpoints(
    directory: str,
    sequences: List[torch.Tensor],
    device: torch.device,
) -> List[Dict[str, float]]:
    results = []
    for checkpoint in sorted(glob.glob(os.path.join(directory, "checkpoint-*.pt"))):
        model, step = load_model(checkpoint, device)
        loss_sum = 0.0
        token_count = 0
        for sequence in sequences:
            tokens = sequence.unsqueeze(0).to(device)
            logits, _ = model(tokens[:, :-1])
            labels = tokens[:, 1:]
            loss_sum += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum"
            ).item()
            token_count += labels.numel()
        results.append({"step": float(step), "fixed_sample_loss": loss_sum / token_count})
        print(directory, results[-1], flush=True)
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
    return results


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    sequences, _ = fixed_token_sequences(
        args.data_dir, tokenizer, args.num_sequences, args.sequence_length
    )
    device = torch.device(args.device)
    results = {
        "baseline": evaluate_checkpoints(args.baseline_dir, sequences, device),
        "proposed": evaluate_checkpoints(args.proposed_dir, sequences, device),
    }
    output_json = os.path.join(args.output_dir, "checkpoint_fixed_sample_loss.json")
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "metric": "mean next-token cross-entropy on fixed DCLM samples",
                "num_sequences": args.num_sequences,
                "sequence_length": args.sequence_length,
                "results": results,
            },
            handle,
            indent=2,
        )

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for name, color in (("baseline", "#777777"), ("proposed", "#2a6fbb")):
        steps = [point["step"] for point in results[name]]
        losses = [point["fixed_sample_loss"] for point in results[name]]
        axis.plot(steps, losses, marker="o", linewidth=2, label=name, color=color)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Fixed-sample next-token loss")
    axis.set_title("Checkpoint loss on fixed DCLM samples")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(os.path.join(args.output_dir, "checkpoint_fixed_sample_loss.png"), dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
