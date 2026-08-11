from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from .common import write_csv


class LayerPairLinearPredictor(nn.Module):
    """One linear 8-way classifier for every source/target layer pair."""

    def __init__(self, layers: int, hidden_size: int, experts: int) -> None:
        super().__init__()
        self.layers = layers
        self.experts = experts
        self.rows = nn.ModuleList(
            nn.Linear(hidden_size, layers * experts) for _ in range(layers)
        )

    def row(self, source_layer: int, features: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = features.shape
        return self.rows[source_layer](features).view(
            batch, tokens, self.layers, self.experts
        )


def setup() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, local_rank = 0, 0
    if not torch.cuda.is_available():
        raise RuntimeError("predictability evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank, torch.device("cuda", local_rank)


@torch.no_grad()
def extract_features(model: nn.Module, input_ids: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
    x = model.embed_tokens(input_ids)
    hidden_inputs = []
    routes = []
    for layer in model.layers:
        hidden_inputs.append(x.detach())
        x, diagnostics = layer(x)
        routes.append(diagnostics["route_indices"].detach())
    return hidden_inputs, torch.stack(routes, dim=0)


def predictor_loss(
    predictor: LayerPairLinearPredictor,
    hidden: list[torch.Tensor],
    routes: torch.Tensor,
) -> torch.Tensor:
    module = predictor
    layers = len(hidden)
    loss_sum = torch.zeros((), device=hidden[0].device)
    item_count = 0
    # The union of same-token i<j and next-token i>=j covers every layer pair,
    # so every classifier participates in every optimizer step.
    for source in range(layers):
        logits = module.row(source, hidden[source].float())
        if source + 1 < layers:
            same_logits = logits[:, :, source + 1 :, :]
            same_labels = routes[source + 1 :, :, :].permute(1, 2, 0)
            loss_sum = loss_sum + F.cross_entropy(
                same_logits.reshape(-1, module.experts),
                same_labels.reshape(-1),
                reduction="sum",
            )
            item_count += same_labels.numel()
        next_logits = logits[:, :-1, : source + 1, :]
        next_labels = routes[: source + 1, :, 1:].permute(1, 2, 0)
        loss_sum = loss_sum + F.cross_entropy(
            next_logits.reshape(-1, module.experts),
            next_labels.reshape(-1),
            reduction="sum",
        )
        item_count += next_labels.numel()
    return loss_sum / item_count


@torch.no_grad()
def accumulate_accuracy(
    predictor: LayerPairLinearPredictor,
    hidden: list[torch.Tensor],
    routes: torch.Tensor,
    correct: torch.Tensor,
    totals: torch.Tensor,
) -> None:
    module = predictor
    layers = len(hidden)
    ks = (1, 2, 4)
    for source in range(layers):
        logits = module.row(source, hidden[source].float())
        top4 = logits.topk(4, dim=-1).indices
        for target in range(layers):
            if source < target:
                task = 0  # same token
                labels = routes[target]
                predictions = top4[:, :, target]
            else:
                task = 1  # current token -> next token
                labels = routes[target, :, 1:]
                predictions = top4[:, :-1, target]
            totals[task, source, target] += labels.numel()
            for k_index, k in enumerate(ks):
                correct[task, source, target, k_index] += (
                    predictions[..., :k] == labels.unsqueeze(-1)
                ).any(dim=-1).sum().cpu()


def batches(items: list[torch.Tensor], batch_size: int, seed: int) -> list[list[torch.Tensor]]:
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    return [
        [items[index] for index in order[start : start + batch_size]]
        for start in range(0, len(order), batch_size)
    ]


def main() -> None:
    from transformers import AutoTokenizer
    try:
        from moe.analysis_utils import fixed_token_sequences, load_model
    except ModuleNotFoundError:  # local source-tree execution
        from fdong_embedding_dim.codex_scripts.real_data_moe.analysis_utils import (
            fixed_token_sequences,
            load_model,
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--size", required=True, choices=("L", "M"))
    parser.add_argument("--method", required=True, choices=("baseline", "proposed"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--train-sequences", type=int, default=512)
    parser.add_argument("--test-sequences", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rank, world, local_rank, device = setup()
    if args.train_sequences % world or args.test_sequences % world:
        raise ValueError("train/test sequence counts must be divisible by WORLD_SIZE")
    if (args.train_sequences // world) % args.batch_size:
        raise ValueError("per-rank train sequence count must be divisible by batch size")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    all_sequences, _ = fixed_token_sequences(
        args.data_dir,
        tokenizer,
        args.train_sequences + args.test_sequences,
        args.sequence_length,
    )
    train_sequences = all_sequences[: args.train_sequences][rank::world]
    test_sequences = all_sequences[args.train_sequences :][rank::world]
    if not train_sequences or not test_sequences:
        raise RuntimeError("each rank needs at least one train and test sequence")

    model, step = load_model(args.checkpoint, device)
    model.requires_grad_(False)
    layers = model.config.num_hidden_layers
    experts = model.config.num_tail_experts
    if experts < 4:
        raise ValueError("recall@4 requires at least four experts")
    predictor = LayerPairLinearPredictor(layers, model.config.hidden_size, experts).to(device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        predictor.train()
        for batch_items in batches(train_sequences, args.batch_size, args.seed + epoch + rank * 1000):
            input_ids = torch.stack(batch_items).to(device)
            # Use no_grad rather than inference_mode: predictor backward must
            # save these frozen-model features for its weight gradients.
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden, routes = extract_features(model, input_ids)
            optimizer.zero_grad(set_to_none=True)
            loss = predictor_loss(predictor, hidden, routes)
            loss.backward()
            if dist.is_initialized():
                for parameter in predictor.parameters():
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                    parameter.grad.div_(world)
            optimizer.step()
        if rank == 0:
            print(f"[predictability] {args.size}/{args.method} epoch={epoch + 1}/{args.epochs}", flush=True)

    predictor.eval()
    correct = torch.zeros(2, layers, layers, 3, dtype=torch.long)
    totals = torch.zeros(2, layers, layers, dtype=torch.long)
    for start in range(0, len(test_sequences), args.batch_size):
        input_ids = torch.stack(test_sequences[start : start + args.batch_size]).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden, routes = extract_features(model, input_ids)
        accumulate_accuracy(predictor, hidden, routes, correct, totals)

    correct_device = correct.to(device)
    totals_device = totals.to(device)
    if dist.is_initialized():
        dist.all_reduce(correct_device, op=dist.ReduceOp.SUM)
        dist.all_reduce(totals_device, op=dist.ReduceOp.SUM)
    correct = correct_device.cpu()
    totals = totals_device.cpu()

    if rank == 0:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        task_names = ("same_token", "next_token")
        for task, task_name in enumerate(task_names):
            for source in range(layers):
                for target in range(layers):
                    valid = source < target if task_name == "same_token" else source >= target
                    if not valid:
                        continue
                    total = int(totals[task, source, target])
                    for k_index, k in enumerate((1, 2, 4)):
                        hits = int(correct[task, source, target, k_index])
                        rows.append(
                            {
                                "size": args.size,
                                "method": args.method,
                                "step": step,
                                "task": task_name,
                                "source_layer": source,
                                "target_layer": target,
                                "target_token_offset": 0 if task_name == "same_token" else 1,
                                "recall_k": k,
                                "correct": hits,
                                "total": total,
                                "recall": hits / total if total else float("nan"),
                            }
                        )
        write_csv(output / "predictability.csv", rows)
        torch.save(predictor.state_dict(), output / "predictor.pt")
        print(f"[predictability] completed {args.size}/{args.method} step={step}", flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
