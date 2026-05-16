import argparse
import json
import math
import os
import sys
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from models import MyQwen3ForCausalLM
from utils import HierarchicalPatternData


def parse_int_list(value):
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def add_args():
    parser = argparse.ArgumentParser(description="Offline inverse-KV routing and attention analysis.")
    parser.add_argument("--config_dir", type=str, default="../Qwen3-0.6B")
    parser.add_argument("--ckpt_file", type=str, default="")
    parser.add_argument("--output_path", type=str, default="../experiments/inverse_kv_analysis.json")

    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--synthetic_block_size", type=int, default=4)
    parser.add_argument("--synthetic_num_hierarchy_layers", type=int, default=2)
    parser.add_argument("--synthetic_content_token_count", type=int, default=128)
    parser.add_argument("--synthetic_num_units_per_layer", type=int, default=32)
    parser.add_argument("--synthetic_seed", type=int, default=0)
    parser.add_argument("--synthetic_min_token_id", type=int, default=1)
    parser.add_argument(
        "--analysis_feature_layer",
        type=int,
        default=0,
        help="Which metadata layer to use as the feature/slot id. 0 means smallest slot.",
    )

    parser.add_argument("--debug_vocab_size", type=int, default=256)
    parser.add_argument("--debug_hidden_size", type=int, default=64)
    parser.add_argument("--debug_intermediate_size", type=int, default=128)
    parser.add_argument("--debug_num_hidden_layers", type=int, default=2)
    parser.add_argument("--debug_num_attention_heads", type=int, default=4)
    parser.add_argument("--debug_num_key_value_heads", type=int, default=2)
    parser.add_argument("--debug_head_dim", type=int, default=16)
    parser.add_argument("--debug_max_position_embeddings", type=int, default=256)

    parser.add_argument("--use_moe", action="store_true", default=True)
    parser.add_argument("--moe_num_unique_experts", type=int, default=4)
    parser.add_argument("--moe_num_experts_per_tok", type=int, default=2)
    parser.add_argument("--moe_intermediate_size", type=int, default=64)
    parser.add_argument("--moe_use_common_expert", action="store_true", default=False)
    parser.add_argument("--moe_common_intermediate_size", type=int, default=64)

    parser.add_argument("--attention_stride_pattern", type=parse_int_list, default=None)
    parser.add_argument("--residual_source_pattern", type=parse_int_list, default=None)
    parser.add_argument("--shuffle_seed", type=int, default=123)
    return parser.parse_args()


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def maybe_override(config, name, value):
    if value != -1:
        setattr(config, name, value)


def build_config(args):
    config = AutoConfig.from_pretrained(args.config_dir, trust_remote_code=True)
    maybe_override(config, "vocab_size", args.debug_vocab_size)
    maybe_override(config, "hidden_size", args.debug_hidden_size)
    maybe_override(config, "intermediate_size", args.debug_intermediate_size)
    maybe_override(config, "num_hidden_layers", args.debug_num_hidden_layers)
    maybe_override(config, "num_attention_heads", args.debug_num_attention_heads)
    maybe_override(config, "num_key_value_heads", args.debug_num_key_value_heads)
    maybe_override(config, "head_dim", args.debug_head_dim)
    maybe_override(config, "max_position_embeddings", args.debug_max_position_embeddings)

    if getattr(config, "pad_token_id", None) is None or config.pad_token_id >= config.vocab_size:
        config.pad_token_id = 0
    if getattr(config, "bos_token_id", None) is not None and config.bos_token_id >= config.vocab_size:
        config.bos_token_id = 1
    if getattr(config, "eos_token_id", None) is not None and config.eos_token_id >= config.vocab_size:
        config.eos_token_id = 2

    config._attn_implementation = "eager"
    config.use_cache = False
    config.attention_stride_pattern = args.attention_stride_pattern or [1] * config.num_hidden_layers
    config.residual_source_pattern = args.residual_source_pattern or [-1] * config.num_hidden_layers
    config.use_moe = bool(args.use_moe)
    config.moe_num_unique_experts = args.moe_num_unique_experts
    config.moe_num_experts_per_tok = args.moe_num_experts_per_tok
    config.moe_intermediate_size = (
        args.moe_intermediate_size if args.moe_intermediate_size != -1 else config.intermediate_size
    )
    config.moe_use_common_expert = bool(args.moe_use_common_expert)
    config.moe_common_intermediate_size = (
        args.moe_common_intermediate_size
        if args.moe_common_intermediate_size != -1
        else config.moe_intermediate_size
    )
    config.moe_router_bias = False
    config.moe_normalize_topk_prob = True
    return config


def load_model(args, device):
    config = build_config(args)
    model = MyQwen3ForCausalLM(config).to(device)
    if args.ckpt_file:
        state = torch.load(args.ckpt_file, map_location=device)
        model.load_state_dict(state)
    model.eval()
    return model


def entropy_from_counts(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [count / total for count in counts if count > 0]
    return float(-sum(p * math.log(p + 1e-12) for p in probs))


def mutual_information(x, y):
    pairs = list(zip(x, y))
    total = len(pairs)
    if total == 0:
        return 0.0
    x_counts = Counter(x)
    y_counts = Counter(y)
    xy_counts = Counter(pairs)
    mi = 0.0
    for (xi, yi), count in xy_counts.items():
        pxy = count / total
        px = x_counts[xi] / total
        py = y_counts[yi] / total
        mi += pxy * math.log((pxy + 1e-12) / (px * py + 1e-12))
    return float(mi)


def mapping_purity(source, target):
    grouped = {}
    for s, t in zip(source, target):
        grouped.setdefault(int(s), Counter())[int(t)] += 1
    total = sum(sum(counter.values()) for counter in grouped.values())
    if total == 0:
        return 0.0
    correct = sum(max(counter.values()) for counter in grouped.values())
    return float(correct / total)


def attention_mass_by_mask(attn, mask):
    denom = attn.sum().clamp_min(1e-12)
    return float((attn * mask.to(attn.dtype)).sum().item() / denom.item())


def attention_mean_by_mask(attn, mask):
    mask = mask.to(torch.bool)
    denom = mask.sum().clamp_min(1)
    return float(attn.masked_select(mask).sum().item() / denom.item())


def pairwise_match_rate_by_mask(labels, mask):
    same_label = labels[:, :, None] == labels[:, None, :]
    valid = mask & (labels[:, :, None] >= 0) & (labels[:, None, :] >= 0)
    denom = valid.sum().clamp_min(1)
    return float((same_label & valid).float().sum().item() / denom.item())


def pairwise_same_expert_given_same_slot(expert_ids, feature_ids):
    same_slot = feature_ids[:, :, None] == feature_ids[:, None, :]
    valid = same_slot & (feature_ids[:, :, None] >= 0) & (feature_ids[:, None, :] >= 0)
    not_self = ~torch.eye(feature_ids.size(1), dtype=torch.bool, device=feature_ids.device)[None, :, :]
    valid = valid & not_self
    denom = valid.sum().clamp_min(1)
    same_expert = expert_ids[:, :, None] == expert_ids[:, None, :]
    return float((same_expert & valid).float().sum().item() / denom.item())


def analyze_attention_layer(attn_layer, expert_ids, feature_ids, shuffle_seed):
    # attn_layer: [batch, heads, seq, seq], expert_ids/feature_ids: [batch, seq]
    batch, heads, seq_len, _ = attn_layer.shape
    device = attn_layer.device
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=-1)
    if causal.sum().item() == 0:
        return {}

    same_expert = expert_ids[:, :, None] == expert_ids[:, None, :]
    same_feature = feature_ids[:, :, None] == feature_ids[:, None, :]
    valid_feature = (feature_ids[:, :, None] >= 0) & (feature_ids[:, None, :] >= 0)
    causal = causal[None, :, :]
    causal_heads = causal[:, None, :, :]
    valid_slot = valid_feature[:, None, :, :] & causal_heads
    same_slot_mask = same_feature[:, None, :, :] & valid_slot
    diff_slot_mask = (~same_feature[:, None, :, :]) & valid_slot
    same_expert_mask = same_expert[:, None, :, :] & causal_heads
    diff_expert_mask = (~same_expert[:, None, :, :]) & causal_heads

    generator = torch.Generator(device=device)
    generator.manual_seed(shuffle_seed)
    shuffled_flat = expert_ids.reshape(-1)[torch.randperm(expert_ids.numel(), generator=generator, device=device)]
    shuffled = shuffled_flat.reshape_as(expert_ids)
    same_expert_shuffled = shuffled[:, :, None] == shuffled[:, None, :]

    top_idx = attn_layer.argmax(dim=-1)
    gathered_expert = expert_ids[:, None, :].expand(batch, heads, seq_len).gather(-1, top_idx)
    query_expert = expert_ids[:, None, :].expand(batch, heads, seq_len)
    top1_same_expert = ((gathered_expert == query_expert) & (torch.arange(seq_len, device=device)[None, None, :] > 0))

    result = {
        "attention_mass_same_expert": attention_mass_by_mask(attn_layer, same_expert_mask),
        "attention_mass_same_slot": attention_mass_by_mask(attn_layer, same_slot_mask),
        "attention_mean_same_slot": attention_mean_by_mask(attn_layer, same_slot_mask),
        "attention_mean_diff_slot": attention_mean_by_mask(attn_layer, diff_slot_mask),
        "attention_same_slot_lift": attention_mean_by_mask(attn_layer, same_slot_mask)
        / max(attention_mean_by_mask(attn_layer, diff_slot_mask), 1e-12),
        "attention_mean_same_expert": attention_mean_by_mask(attn_layer, same_expert_mask),
        "attention_mean_diff_expert": attention_mean_by_mask(attn_layer, diff_expert_mask),
        "attention_same_expert_lift": attention_mean_by_mask(attn_layer, same_expert_mask)
        / max(attention_mean_by_mask(attn_layer, diff_expert_mask), 1e-12),
        "attention_mass_same_expert_shuffled": attention_mass_by_mask(
            attn_layer, same_expert_shuffled[:, None, :, :] & causal_heads
        ),
        "top1_same_expert_rate": float(top1_same_expert.float().mean().item()),
    }
    return result


@torch.no_grad()
def main():
    args = add_args()
    device = choose_device()
    dataset = HierarchicalPatternData(
        max_seq_len=args.seq_len,
        num_samples=args.num_samples,
        block_size=args.synthetic_block_size,
        num_hierarchy_layers=args.synthetic_num_hierarchy_layers,
        content_token_count=args.synthetic_content_token_count,
        num_units_per_layer=args.synthetic_num_units_per_layer,
        seed=args.synthetic_seed,
        min_token_id=args.synthetic_min_token_id,
        return_metadata=True,
    )
    model = load_model(args, device)

    all_layer_experts = [[] for _ in range(model.config.num_hidden_layers)]
    all_feature_ids = []
    attention_metrics = [[] for _ in range(model.config.num_hidden_layers)]
    losses = []
    correct_tokens = 0
    total_tokens = 0

    if args.analysis_feature_layer < 0 or args.analysis_feature_layer >= args.synthetic_num_hierarchy_layers:
        raise ValueError(
            "`analysis_feature_layer` must be in [0, synthetic_num_hierarchy_layers), "
            f"got {args.analysis_feature_layer}."
        )

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    for start in range(0, args.num_samples, args.batch_size):
        batch_items = [dataset[i] for i in range(start, min(start + args.batch_size, args.num_samples))]
        source = torch.stack([item[0] for item in batch_items]).to(device)
        target = torch.stack([item[1] for item in batch_items]).to(device)
        metadata = torch.stack([item[3] for item in batch_items]).to(device)
        feature_ids = metadata[:, :, args.analysis_feature_layer]

        outputs = model(
            source,
            output_attentions=True,
            output_expert_labels=True,
            use_cache=False,
        )
        loss = loss_fn(outputs.logits.reshape(-1, outputs.logits.size(-1)), target.reshape(-1))
        losses.append(float(loss.item()))
        pred = outputs.logits.argmax(dim=-1)
        valid = target != 0
        correct_tokens += int(((pred == target) & valid).sum().item())
        total_tokens += int(valid.sum().item())
        all_feature_ids.append(feature_ids.cpu())

        for layer_idx, expert_labels in enumerate(outputs.expert_labels):
            primary_expert = expert_labels[..., 0]
            all_layer_experts[layer_idx].append(primary_expert.cpu())
            attention_metrics[layer_idx].append(
                analyze_attention_layer(
                    outputs.attentions[layer_idx].detach(),
                    primary_expert.detach(),
                    feature_ids.detach(),
                    args.shuffle_seed + layer_idx,
                )
            )

    feature_ids_all = torch.cat(all_feature_ids, dim=0).reshape(-1).tolist()
    summary = {
        "config": vars(args),
        "device": str(device),
        "loss_mean": float(sum(losses) / max(len(losses), 1)),
        "token_accuracy": float(correct_tokens / max(total_tokens, 1)),
        "analysis_feature_layer": args.analysis_feature_layer,
        "layers": [],
    }
    for layer_idx, layer_experts in enumerate(all_layer_experts):
        expert_tensor = torch.cat(layer_experts, dim=0)
        feature_tensor = torch.cat(all_feature_ids, dim=0)
        experts = expert_tensor.reshape(-1).tolist()
        counts = [0] * int(model.config.moe_num_unique_experts)
        for expert in experts:
            if 0 <= int(expert) < len(counts):
                counts[int(expert)] += 1
        layer_metric_rows = attention_metrics[layer_idx]
        mean_attention_metrics = {}
        if layer_metric_rows:
            for key in layer_metric_rows[0]:
                mean_attention_metrics[key] = float(
                    sum(row[key] for row in layer_metric_rows) / len(layer_metric_rows)
                )
        summary["layers"].append(
            {
                "layer": layer_idx,
                "expert_load": counts,
                "expert_entropy": entropy_from_counts(counts),
                "slot_expert_mi": mutual_information(feature_ids_all, experts),
                "slot_to_expert_purity": mapping_purity(feature_ids_all, experts),
                "expert_to_slot_purity": mapping_purity(experts, feature_ids_all),
                "same_slot_same_expert_rate": pairwise_same_expert_given_same_slot(
                    expert_tensor, feature_tensor
                ),
                **mean_attention_metrics,
            }
        )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
