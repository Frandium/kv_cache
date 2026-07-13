from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .analysis_utils import fixed_token_sequences, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="NAME=CHECKPOINT")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-sequences", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    runs = dict(item.split("=", 1) for item in args.run)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    sequences, _ = fixed_token_sequences(
        args.data_dir, tokenizer, args.num_sequences, args.sequence_length
    )
    device = torch.device(args.device)
    results = {}
    for name, checkpoint in runs.items():
        model, step = load_model(checkpoint, device)
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for start in range(0, len(sequences), args.batch_size):
                batch = torch.stack(sequences[start : start + args.batch_size]).to(device)
                logits, _ = model(batch[:, :-1])
                labels = batch[:, 1:]
                total_loss += F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="sum",
                ).float().item()
                total_tokens += labels.numel()
        loss = total_loss / total_tokens
        results[name] = {
            "checkpoint": os.path.abspath(checkpoint),
            "step": step,
            "tokens": total_tokens,
            "test_loss": loss,
            "perplexity": math.exp(loss),
        }
        print(name, json.dumps(results[name]), flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
