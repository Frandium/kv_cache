from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from .data import PatternData
from .model import CausalBackbone, MultiTokenModel, Probe, parameter_count


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def offset_cross_entropy_losses(
    logits: list[torch.Tensor], tokens: torch.Tensor, loss_mask: torch.Tensor | None = None
) -> list[torch.Tensor]:
    losses = []
    for offset, offset_logits in enumerate(logits, start=1):
        if offset >= tokens.size(1):
            break
        predictions = offset_logits[:, :-offset]
        targets = tokens[:, offset:]
        if loss_mask is None:
            valid = torch.ones_like(targets, dtype=torch.bool)
        else:
            valid = loss_mask[:, :-offset] & loss_mask[:, offset:]
        if not torch.any(valid):
            continue
        losses.append(
            F.cross_entropy(
                predictions[valid],
                targets[valid],
            )
        )
    if not losses:
        raise ValueError("sequence is too short for requested MTP horizon")
    return losses


def multi_token_loss(
    logits: list[torch.Tensor], tokens: torch.Tensor, loss_mask: torch.Tensor | None = None
) -> torch.Tensor:
    losses = offset_cross_entropy_losses(logits, tokens, loss_mask)
    return torch.stack(losses).mean()


def train_model(
    data: PatternData,
    backbone_kind: str,
    hidden_size: int,
    mtp: int,
    steps: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    num_attention_heads: int = 1,
) -> tuple[MultiTokenModel, list[float]]:
    set_seed(seed)
    backbone = CausalBackbone(
        vocab_size=data.vocab_size,
        hidden_size=hidden_size,
        kind=backbone_kind,
        num_attention_heads=num_attention_heads,
    )
    model = MultiTokenModel(backbone, data.vocab_size, mtp).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    train_tokens = data.sequences[data.train_mask].to(device)
    train_loss_mask = data.loss_mask[data.train_mask].to(device) if data.loss_mask is not None else None
    history: list[float] = []
    for step in range(steps):
        _, logits = model(train_tokens)
        loss = multi_token_loss(logits, train_tokens, train_loss_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 20) == 0:
            history.append(float(loss.detach().cpu()))
    return model, history


@torch.no_grad()
def evaluate_offsets(
    model: MultiTokenModel, tokens: torch.Tensor, loss_mask: torch.Tensor | None = None
) -> dict[str, dict[str, float]]:
    model.eval()
    _, logits = model(tokens)
    result: dict[str, dict[str, float]] = {}
    for offset, offset_logits in enumerate(logits, start=1):
        if offset >= tokens.size(1):
            break
        predictions = offset_logits[:, :-offset]
        targets = tokens[:, offset:]
        if loss_mask is None:
            valid = torch.ones_like(targets, dtype=torch.bool)
        else:
            valid = loss_mask[:, :-offset] & loss_mask[:, offset:]
        if not torch.any(valid):
            continue
        ce = F.cross_entropy(predictions[valid], targets[valid])
        accuracy = (predictions.argmax(dim=-1)[valid] == targets[valid]).float().mean()
        result[f"offset_{offset}"] = {
            "cross_entropy": float(ce.cpu()),
            "perplexity": float(torch.exp(ce).cpu()),
            "accuracy": float(accuracy.cpu()),
        }
    return result


@torch.no_grad()
def evaluate_next_token_positions(
    model: MultiTokenModel, tokens: torch.Tensor
) -> dict[str, dict[str, float]]:
    """Report NTP-head metrics separately for every sequence transition."""
    model.eval()
    _, logits = model(tokens)
    head = logits[0]
    result: dict[str, dict[str, float]] = {}
    for position in range(tokens.size(1) - 1):
        label = f"position_{position}_to_{position + 1}"
        prediction = head[:, position]
        target = tokens[:, position + 1]
        ce = F.cross_entropy(prediction, target)
        result[label] = {
            "cross_entropy": float(ce.cpu()),
            "perplexity": float(torch.exp(ce).cpu()),
            "accuracy": float((prediction.argmax(dim=-1) == target).float().mean().cpu()),
        }
    return result


def train_frozen_probe(
    model: MultiTokenModel,
    data: PatternData,
    probe_kind: str,
    steps: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[Probe, list[float]]:
    """Train a fresh probe h(prefix)->suffix while the entire model is frozen."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    with torch.no_grad():
        train_tokens = data.sequences[data.train_mask].to(device)
        train_hidden = model.backbone(train_tokens)[:, data.probe_position].detach()
        train_hidden = F.layer_norm(train_hidden, (train_hidden.size(-1),))
        train_targets = train_tokens[:, data.target_position]

    set_seed(seed)
    probe = Probe(model.backbone.hidden_size, data.vocab_size, probe_kind).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=0.0)
    history: list[float] = []
    for step in range(steps):
        loss = F.cross_entropy(probe(train_hidden), train_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 20) == 0:
            history.append(float(loss.detach().cpu()))
    return probe, history


@torch.no_grad()
def evaluate_probe(
    model: MultiTokenModel,
    probe: Probe,
    tokens: torch.Tensor,
    probe_position: int,
    target_position: int,
) -> dict[str, float]:
    model.eval()
    probe.eval()
    hidden = model.backbone(tokens)[:, probe_position]
    hidden = F.layer_norm(hidden, (hidden.size(-1),))
    logits = probe(hidden)
    targets = tokens[:, target_position]
    ce = F.cross_entropy(logits, targets)
    return {
        "cross_entropy": float(ce.cpu()),
        "perplexity": float(torch.exp(ce).cpu()),
        "accuracy": float((logits.argmax(dim=-1) == targets).float().mean().cpu()),
    }


@torch.no_grad()
def prefix_spectrum(
    model: MultiTokenModel,
    data: PatternData,
    device: torch.device,
) -> dict[str, float | list[float]]:
    """Spectrum of one centered h(prefix_i) per unique prefix."""
    model.eval()
    rows = []
    for prefix in torch.unique(data.prefix_index):
        index = torch.nonzero(data.prefix_index == prefix, as_tuple=False)[0, 0]
        token = data.sequences[index : index + 1].to(device)
        rows.append(model.backbone(token)[:, data.probe_position].squeeze(0))
    matrix = torch.stack(rows)
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(matrix.float())
    squared = singular_values.square()
    total = squared.sum()
    if float(total) == 0.0:
        normalized_energy = torch.zeros_like(squared)
        effective_rank = 0.0
        stable_rank = 0.0
    else:
        normalized_energy = squared / total
        nonzero = normalized_energy[normalized_energy > 0]
        effective_rank = float(torch.exp(-(nonzero * nonzero.log()).sum()).cpu())
        stable_rank = float((total / squared.max()).cpu())
    return {
        "singular_values": [float(value) for value in singular_values.cpu()],
        "normalized_energy": [float(value) for value in normalized_energy.cpu()],
        "top1_energy_fraction": float(normalized_energy[0].cpu()) if len(normalized_energy) else 0.0,
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
    }


def model_metadata(model: MultiTokenModel) -> dict[str, int | str]:
    return {
        "backbone_kind": model.backbone.kind,
        "hidden_size": model.backbone.hidden_size,
        "mtp": model.mtp,
        "total_parameters": parameter_count(model),
        "backbone_parameters": parameter_count(model.backbone),
    }
