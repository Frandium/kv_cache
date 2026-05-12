import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from models import MyQwen3ForCausalLM
from utils import TokenizedJSONLData


def parse_int_list(value):
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Teacher-forced KV-cache decode test for MyQwen3.")
    parser.add_argument("--local_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--prefill_len", type=int, default=100)
    parser.add_argument("--decode_steps", type=int, default=128)

    parser.add_argument("--config_dir", type=str, default="../../Qwen3-0.6B")
    parser.add_argument("--data_dir", type=str, default="../../dclm/global-shard_01_of_10")
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--ckpt_step", type=int, default=None)
    parser.add_argument("--ckpt_file", type=str, default="")

    parser.add_argument("--attention_stride_pattern", type=parse_int_list, default=None)
    parser.add_argument("--residual_source_pattern", type=parse_int_list, default=None)
    parser.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa"])

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_shuffle", action="store_true", default=False)
    parser.add_argument("--no_data_shuffle", action="store_false", dest="data_shuffle")
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--no_use_bf16", action="store_false", dest="use_bf16")
    parser.add_argument("--ignore_index", type=int, default=151643)
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


def runtime_config_path(args):
    if args.ckpt_dir:
        return os.path.join(args.ckpt_dir, "runtime_config.json")
    if args.ckpt_file:
        return os.path.join(os.path.dirname(args.ckpt_file), "runtime_config.json")
    return ""


def load_runtime_config(args):
    path = runtime_config_path(args)
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        runtime_config = json.load(f)
    print(f"Loaded runtime config: {path}", flush=True)
    return runtime_config


def resolve_pattern(cli_pattern, runtime_config, key, default_pattern):
    runtime_pattern = runtime_config.get(key)
    if cli_pattern is not None and runtime_pattern is not None and cli_pattern != runtime_pattern:
        raise ValueError(
            f"`{key}` mismatch between command line and runtime_config.json.\n"
            f"command line: {cli_pattern}\n"
            f"runtime_config: {runtime_pattern}"
        )
    return cli_pattern or runtime_pattern or default_pattern


def prepare_model(args, device):
    config = AutoConfig.from_pretrained(args.config_dir, trust_remote_code=True)
    runtime_config = load_runtime_config(args)
    config._attn_implementation = args.attn_implementation
    config.attention_stride_pattern = resolve_pattern(
        args.attention_stride_pattern,
        runtime_config,
        "attention_stride_pattern",
        [1 for _ in range(config.num_hidden_layers)],
    )
    config.residual_source_pattern = resolve_pattern(
        args.residual_source_pattern,
        runtime_config,
        "residual_source_pattern",
        [-1 for _ in range(config.num_hidden_layers)],
    )

    print(f"Model attention_stride_pattern: {config.attention_stride_pattern}", flush=True)
    print(f"Model residual_source_pattern: {config.residual_source_pattern}", flush=True)

    model = MyQwen3ForCausalLM(config).to(device)
    ckpt_file = resolve_checkpoint(args)
    if ckpt_file is not None:
        print(f"Loading checkpoint: {ckpt_file}", flush=True)
        state_dict = torch.load(ckpt_file, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print("No checkpoint provided; evaluating randomly initialized model.", flush=True)

    model.eval()
    return model


def resolve_checkpoint(args):
    if args.ckpt_file:
        return args.ckpt_file
    if args.ckpt_dir and args.ckpt_step is not None:
        return os.path.join(args.ckpt_dir, f"{args.ckpt_step}.pth")
    return None


def prepare_data(args):
    tokenizer = AutoTokenizer.from_pretrained(args.config_dir, trust_remote_code=True)
    dataset = TokenizedJSONLData(args.data_dir, args.seq_len, tokenizer)
    print(f"Construct dataset, total {len(dataset)} samples.", flush=True)
    return DataLoader(
        dataset,
        batch_size=args.local_batch_size,
        num_workers=args.num_workers,
        shuffle=args.data_shuffle,
    )


def ce_sum_and_count(logits, labels, ignore_index):
    loss_sum = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )
    count = labels.ne(ignore_index).sum()
    return loss_sum, count


def ce_sum_and_count_last(logits, labels, ignore_index):
    loss_sum = F.cross_entropy(
        logits.float(),
        labels,
        ignore_index=ignore_index,
        reduction="sum",
    )
    count = labels.ne(ignore_index).sum()
    return loss_sum, count


def cache_lengths(past_key_values):
    if past_key_values is None:
        return {}
    if hasattr(past_key_values, "get_cache_lengths"):
        return past_key_values.get_cache_lengths()
    if hasattr(past_key_values, "key_cache"):
        return {
            layer_idx: 0 if key_states is None else key_states.shape[-2]
            for layer_idx, key_states in enumerate(past_key_values.key_cache)
        }
    return {}


def total_cache_tokens(lengths):
    return sum(int(length) for length in lengths.values())


def analyzed_strides(attention_stride_pattern):
    return sorted({int(stride) for stride in attention_stride_pattern if int(stride) > 1})


def init_modulo_stats(attention_stride_pattern):
    return {
        stride: {modulo: {"loss_sum": 0.0, "count": 0} for modulo in range(stride)}
        for stride in analyzed_strides(attention_stride_pattern)
    }


def update_modulo_stats(stats, logits, labels, positions, ignore_index):
    if not stats or logits.shape[1] == 0:
        return
    token_losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(labels.shape)
    valid_mask = labels.ne(ignore_index)
    for stride, stride_stats in stats.items():
        modulo_positions = torch.remainder(positions, stride)
        for modulo, modulo_stats in stride_stats.items():
            mask = valid_mask & (modulo_positions.unsqueeze(0) == modulo)
            count = int(mask.sum().detach().cpu())
            if count == 0:
                continue
            modulo_stats["loss_sum"] += float(token_losses[mask].sum().detach().cpu())
            modulo_stats["count"] += count


def summarize_modulo_stats(stats):
    summary = {}
    for stride, stride_stats in stats.items():
        stride_summary = {}
        for modulo, values in stride_stats.items():
            count = values["count"]
            stride_summary[str(modulo)] = {
                "loss": None if count == 0 else values["loss_sum"] / count,
                "count": count,
                "is_anchor": modulo == stride - 1,
            }
        summary[str(stride)] = stride_summary
    return summary


@torch.no_grad()
def full_sequence_loss(model, source, target, args, modulo_stats=None):
    output = model(source, use_cache=False, output_hidden_states=False)
    if modulo_stats is not None:
        positions = torch.arange(source.shape[1], device=source.device)
        update_modulo_stats(
            modulo_stats,
            output.logits,
            target,
            positions,
            args.ignore_index,
        )
    return ce_sum_and_count(output.logits, target, args.ignore_index)


@torch.no_grad()
def teacher_forced_decode(model, source, target, args, anchor_only_kv_cache):
    batch_size, seq_len = source.shape
    prefill_len = min(args.prefill_len, seq_len)
    if prefill_len < 1:
        raise ValueError("prefill_len must be at least 1.")

    if args.decode_steps < 0:
        decode_end = seq_len
    else:
        decode_end = min(seq_len, prefill_len + args.decode_steps)

    output = model(
        input_ids=source[:, :prefill_len],
        use_cache=True,
        anchor_only_kv_cache=anchor_only_kv_cache,
        output_hidden_states=False,
    )
    past_key_values = output.past_key_values

    prefill_loss_sum, prefill_count = ce_sum_and_count(
        output.logits,
        target[:, :prefill_len],
        args.ignore_index,
    )
    next_token_loss_sum, next_token_count = ce_sum_and_count_last(
        output.logits[:, -1, :],
        target[:, prefill_len - 1],
        args.ignore_index,
    )

    decode_loss_sum = source.new_tensor(0, dtype=torch.float32)
    decode_count = source.new_tensor(0)
    for input_pos in range(prefill_len, decode_end):
        output = model(
            input_ids=source[:, input_pos : input_pos + 1],
            past_key_values=past_key_values,
            use_cache=True,
            anchor_only_kv_cache=anchor_only_kv_cache,
            output_hidden_states=False,
        )
        past_key_values = output.past_key_values
        step_logits = output.logits[:, -1, :]
        step_loss_sum, step_count = ce_sum_and_count_last(
            step_logits,
            target[:, input_pos],
            args.ignore_index,
        )
        decode_loss_sum = decode_loss_sum + step_loss_sum
        decode_count = decode_count + step_count

    total_loss_sum = prefill_loss_sum + decode_loss_sum
    total_count = prefill_count + decode_count

    return {
        "loss_sum": total_loss_sum,
        "count": total_count,
        "prefill_loss_sum": prefill_loss_sum,
        "prefill_count": prefill_count,
        "prefill_next_token_loss_sum": next_token_loss_sum,
        "prefill_next_token_count": next_token_count,
        "decode_loss_sum": decode_loss_sum,
        "decode_count": decode_count,
        "cache_lengths": cache_lengths(past_key_values),
    }


@torch.no_grad()
def teacher_forced_decode_pair(model, source, target, args):
    batch_size, seq_len = source.shape
    prefill_len = min(args.prefill_len, seq_len)
    if prefill_len < 1:
        raise ValueError("prefill_len must be at least 1.")

    if args.decode_steps < 0:
        decode_end = seq_len
    else:
        decode_end = min(seq_len, prefill_len + args.decode_steps)

    full_output = model(
        input_ids=source[:, :prefill_len],
        use_cache=True,
        anchor_only_kv_cache=False,
        output_hidden_states=False,
    )
    anchor_output = model(
        input_ids=source[:, :prefill_len],
        use_cache=True,
        anchor_only_kv_cache=True,
        output_hidden_states=False,
    )
    full_past = full_output.past_key_values
    anchor_past = anchor_output.past_key_values

    full_prefill_loss_sum, full_prefill_count = ce_sum_and_count(
        full_output.logits,
        target[:, :prefill_len],
        args.ignore_index,
    )
    anchor_prefill_loss_sum, anchor_prefill_count = ce_sum_and_count(
        anchor_output.logits,
        target[:, :prefill_len],
        args.ignore_index,
    )
    full_next_loss_sum, full_next_count = ce_sum_and_count_last(
        full_output.logits[:, -1, :],
        target[:, prefill_len - 1],
        args.ignore_index,
    )
    anchor_next_loss_sum, anchor_next_count = ce_sum_and_count_last(
        anchor_output.logits[:, -1, :],
        target[:, prefill_len - 1],
        args.ignore_index,
    )

    full_decode_loss_sum = source.new_tensor(0, dtype=torch.float32)
    full_decode_count = source.new_tensor(0)
    anchor_decode_loss_sum = source.new_tensor(0, dtype=torch.float32)
    anchor_decode_count = source.new_tensor(0)

    max_abs_diff = 0.0
    abs_diff_sum = 0.0
    abs_diff_count = 0
    top1_match_sum = 0
    top1_match_count = 0
    full_modulo_stats = init_modulo_stats(model.model.attention_stride_pattern)
    anchor_modulo_stats = init_modulo_stats(model.model.attention_stride_pattern)

    prefill_positions = torch.arange(prefill_len, device=source.device)
    update_modulo_stats(
        full_modulo_stats,
        full_output.logits,
        target[:, :prefill_len],
        prefill_positions,
        args.ignore_index,
    )
    update_modulo_stats(
        anchor_modulo_stats,
        anchor_output.logits,
        target[:, :prefill_len],
        prefill_positions,
        args.ignore_index,
    )

    prefill_diff = (full_output.logits[:, -1, :].float() - anchor_output.logits[:, -1, :].float()).abs()
    max_abs_diff = max(max_abs_diff, float(prefill_diff.max().detach().cpu()))
    abs_diff_sum += float(prefill_diff.sum().detach().cpu())
    abs_diff_count += prefill_diff.numel()
    top1_match_sum += int(
        full_output.logits[:, -1, :].argmax(dim=-1).eq(anchor_output.logits[:, -1, :].argmax(dim=-1)).sum().detach().cpu()
    )
    top1_match_count += batch_size

    for input_pos in range(prefill_len, decode_end):
        full_output = model(
            input_ids=source[:, input_pos : input_pos + 1],
            past_key_values=full_past,
            use_cache=True,
            anchor_only_kv_cache=False,
            output_hidden_states=False,
        )
        anchor_output = model(
            input_ids=source[:, input_pos : input_pos + 1],
            past_key_values=anchor_past,
            use_cache=True,
            anchor_only_kv_cache=True,
            output_hidden_states=False,
        )
        full_past = full_output.past_key_values
        anchor_past = anchor_output.past_key_values

        full_logits = full_output.logits[:, -1, :]
        anchor_logits = anchor_output.logits[:, -1, :]
        full_step_loss_sum, full_step_count = ce_sum_and_count_last(
            full_logits,
            target[:, input_pos],
            args.ignore_index,
        )
        anchor_step_loss_sum, anchor_step_count = ce_sum_and_count_last(
            anchor_logits,
            target[:, input_pos],
            args.ignore_index,
        )
        full_decode_loss_sum = full_decode_loss_sum + full_step_loss_sum
        full_decode_count = full_decode_count + full_step_count
        anchor_decode_loss_sum = anchor_decode_loss_sum + anchor_step_loss_sum
        anchor_decode_count = anchor_decode_count + anchor_step_count
        step_position = torch.tensor([input_pos], device=source.device)
        update_modulo_stats(
            full_modulo_stats,
            full_logits.unsqueeze(1),
            target[:, input_pos : input_pos + 1],
            step_position,
            args.ignore_index,
        )
        update_modulo_stats(
            anchor_modulo_stats,
            anchor_logits.unsqueeze(1),
            target[:, input_pos : input_pos + 1],
            step_position,
            args.ignore_index,
        )

        diff = (full_logits.float() - anchor_logits.float()).abs()
        max_abs_diff = max(max_abs_diff, float(diff.max().detach().cpu()))
        abs_diff_sum += float(diff.sum().detach().cpu())
        abs_diff_count += diff.numel()
        top1_match_sum += int(full_logits.argmax(dim=-1).eq(anchor_logits.argmax(dim=-1)).sum().detach().cpu())
        top1_match_count += batch_size

    full_result = {
        "loss_sum": full_prefill_loss_sum + full_decode_loss_sum,
        "count": full_prefill_count + full_decode_count,
        "prefill_loss_sum": full_prefill_loss_sum,
        "prefill_count": full_prefill_count,
        "prefill_next_token_loss_sum": full_next_loss_sum,
        "prefill_next_token_count": full_next_count,
        "decode_loss_sum": full_decode_loss_sum,
        "decode_count": full_decode_count,
        "cache_lengths": cache_lengths(full_past),
        "modulo_stats": full_modulo_stats,
    }
    anchor_result = {
        "loss_sum": anchor_prefill_loss_sum + anchor_decode_loss_sum,
        "count": anchor_prefill_count + anchor_decode_count,
        "prefill_loss_sum": anchor_prefill_loss_sum,
        "prefill_count": anchor_prefill_count,
        "prefill_next_token_loss_sum": anchor_next_loss_sum,
        "prefill_next_token_count": anchor_next_count,
        "decode_loss_sum": anchor_decode_loss_sum,
        "decode_count": anchor_decode_count,
        "cache_lengths": cache_lengths(anchor_past),
        "modulo_stats": anchor_modulo_stats,
    }
    comparison = {
        "max_abs_diff": max_abs_diff,
        "abs_diff_sum": abs_diff_sum,
        "abs_diff_count": abs_diff_count,
        "top1_match_sum": top1_match_sum,
        "top1_match_count": top1_match_count,
    }
    return full_result, anchor_result, comparison


def add_metric(acc, prefix, loss_sum, count):
    acc[f"{prefix}_loss_sum"] += float(loss_sum.detach().cpu())
    acc[f"{prefix}_count"] += int(count.detach().cpu())


def mean_from_acc(acc, prefix):
    count = acc[f"{prefix}_count"]
    if count == 0:
        return None
    return acc[f"{prefix}_loss_sum"] / count


def update_cache_stats(acc, prefix, lengths):
    total = total_cache_tokens(lengths)
    acc[f"{prefix}_cache_total_tokens_sum"] += total
    acc[f"{prefix}_cache_observations"] += 1
    for layer_idx, length in lengths.items():
        key = f"{prefix}_layer_{layer_idx}_cache_tokens_sum"
        acc[key] = acc.get(key, 0) + int(length)


def merge_modulo_stats(acc_stats, batch_stats):
    for stride, stride_stats in batch_stats.items():
        if stride not in acc_stats:
            acc_stats[stride] = {modulo: {"loss_sum": 0.0, "count": 0} for modulo in stride_stats}
        for modulo, values in stride_stats.items():
            acc_stats[stride][modulo]["loss_sum"] += values["loss_sum"]
            acc_stats[stride][modulo]["count"] += values["count"]


def summarize_cache(acc, prefix, num_layers):
    observations = acc[f"{prefix}_cache_observations"]
    if observations == 0:
        return {"total_tokens": None, "per_layer_tokens": {}}
    per_layer = {
        str(layer_idx): acc.get(f"{prefix}_layer_{layer_idx}_cache_tokens_sum", 0) / observations
        for layer_idx in range(num_layers)
    }
    return {
        "total_tokens": acc[f"{prefix}_cache_total_tokens_sum"] / observations,
        "per_layer_tokens": per_layer,
    }


@torch.no_grad()
def evaluate(model, dataloader, args, device):
    num_layers = model.config.num_hidden_layers
    full_sequence_modulo_stats = init_modulo_stats(model.model.attention_stride_pattern)
    full_kv_modulo_stats = init_modulo_stats(model.model.attention_stride_pattern)
    anchor_kv_modulo_stats = init_modulo_stats(model.model.attention_stride_pattern)
    acc = {
        "full_sequence_loss_sum": 0.0,
        "full_sequence_count": 0,
        "full_kv_loss_sum": 0.0,
        "full_kv_count": 0,
        "full_kv_prefill_loss_sum": 0.0,
        "full_kv_prefill_count": 0,
        "full_kv_prefill_next_token_loss_sum": 0.0,
        "full_kv_prefill_next_token_count": 0,
        "full_kv_decode_loss_sum": 0.0,
        "full_kv_decode_count": 0,
        "anchor_kv_loss_sum": 0.0,
        "anchor_kv_count": 0,
        "anchor_kv_prefill_loss_sum": 0.0,
        "anchor_kv_prefill_count": 0,
        "anchor_kv_prefill_next_token_loss_sum": 0.0,
        "anchor_kv_prefill_next_token_count": 0,
        "anchor_kv_decode_loss_sum": 0.0,
        "anchor_kv_decode_count": 0,
        "full_kv_cache_total_tokens_sum": 0,
        "full_kv_cache_observations": 0,
        "anchor_kv_cache_total_tokens_sum": 0,
        "anchor_kv_cache_observations": 0,
        "max_decode_logit_abs_diff": 0.0,
        "decode_logit_abs_diff_sum": 0.0,
        "decode_logit_abs_diff_count": 0,
        "decode_top1_match_sum": 0,
        "decode_top1_match_count": 0,
        "num_batches": 0,
        "num_samples": 0,
    }

    max_batches = max(1, args.test_batch_size // args.local_batch_size)
    for batch_idx, (source, target, real_lens) in enumerate(dataloader, 1):
        source = source.to(device)
        target = target.to(device)
        acc["num_batches"] += 1
        acc["num_samples"] += source.shape[0]

        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=args.use_bf16):
            full_loss_sum, full_count = full_sequence_loss(model, source, target, args, full_sequence_modulo_stats)
            full_kv, anchor_kv, comparison = teacher_forced_decode_pair(model, source, target, args)

        add_metric(acc, "full_sequence", full_loss_sum, full_count)
        for prefix, result in [("full_kv", full_kv), ("anchor_kv", anchor_kv)]:
            add_metric(acc, prefix, result["loss_sum"], result["count"])
            add_metric(acc, f"{prefix}_prefill", result["prefill_loss_sum"], result["prefill_count"])
            add_metric(
                acc,
                f"{prefix}_prefill_next_token",
                result["prefill_next_token_loss_sum"],
                result["prefill_next_token_count"],
            )
            add_metric(acc, f"{prefix}_decode", result["decode_loss_sum"], result["decode_count"])
            update_cache_stats(acc, prefix, result["cache_lengths"])
            if prefix == "full_kv":
                merge_modulo_stats(full_kv_modulo_stats, result["modulo_stats"])
            else:
                merge_modulo_stats(anchor_kv_modulo_stats, result["modulo_stats"])

        acc["max_decode_logit_abs_diff"] = max(acc["max_decode_logit_abs_diff"], comparison["max_abs_diff"])
        acc["decode_logit_abs_diff_sum"] += comparison["abs_diff_sum"]
        acc["decode_logit_abs_diff_count"] += comparison["abs_diff_count"]
        acc["decode_top1_match_sum"] += comparison["top1_match_sum"]
        acc["decode_top1_match_count"] += comparison["top1_match_count"]

        print(
            f"batch {batch_idx}: "
            f"full_seq={float(full_loss_sum / full_count):.4f}, "
            f"full_kv={float(full_kv['loss_sum'] / full_kv['count']):.4f}, "
            f"anchor_kv={float(anchor_kv['loss_sum'] / anchor_kv['count']):.4f}, "
            f"anchor_cache_tokens={total_cache_tokens(anchor_kv['cache_lengths'])}",
            flush=True,
        )

        if batch_idx >= max_batches:
            break

    full_cache = summarize_cache(acc, "full_kv", num_layers)
    anchor_cache = summarize_cache(acc, "anchor_kv", num_layers)
    full_total = full_cache["total_tokens"] or 0
    anchor_total = anchor_cache["total_tokens"] or 0
    cache_ratio = None if full_total == 0 else anchor_total / full_total

    mean_logit_diff = None
    if acc["decode_logit_abs_diff_count"] > 0:
        mean_logit_diff = acc["decode_logit_abs_diff_sum"] / acc["decode_logit_abs_diff_count"]
    top1_match_rate = None
    if acc["decode_top1_match_count"] > 0:
        top1_match_rate = acc["decode_top1_match_sum"] / acc["decode_top1_match_count"]

    return {
        "num_batches": acc["num_batches"],
        "num_samples": acc["num_samples"],
        "prefill_len": args.prefill_len,
        "decode_steps": args.decode_steps,
        "loss": {
            "full_sequence": mean_from_acc(acc, "full_sequence"),
            "full_kv_teacher_forced": mean_from_acc(acc, "full_kv"),
            "anchor_kv_teacher_forced": mean_from_acc(acc, "anchor_kv"),
            "full_kv_prefill": mean_from_acc(acc, "full_kv_prefill"),
            "anchor_kv_prefill": mean_from_acc(acc, "anchor_kv_prefill"),
            "full_kv_prefill_next_token": mean_from_acc(acc, "full_kv_prefill_next_token"),
            "anchor_kv_prefill_next_token": mean_from_acc(acc, "anchor_kv_prefill_next_token"),
            "full_kv_decode_only": mean_from_acc(acc, "full_kv_decode"),
            "anchor_kv_decode_only": mean_from_acc(acc, "anchor_kv_decode"),
        },
        "logit_equivalence": {
            "max_decode_logit_abs_diff": acc["max_decode_logit_abs_diff"],
            "mean_decode_logit_abs_diff": mean_logit_diff,
            "decode_top1_match_rate": top1_match_rate,
        },
        "cache": {
            "full_kv": full_cache,
            "anchor_kv": anchor_cache,
            "anchor_vs_full_total_token_ratio": cache_ratio,
            "cache_reduction": None if cache_ratio is None else 1.0 - cache_ratio,
        },
        "position_loss_by_modulo": {
            "full_sequence": summarize_modulo_stats(full_sequence_modulo_stats),
            "full_kv_teacher_forced": summarize_modulo_stats(full_kv_modulo_stats),
            "anchor_kv_teacher_forced": summarize_modulo_stats(anchor_kv_modulo_stats),
        },
        "config": {
            "config_dir": args.config_dir,
            "data_dir": args.data_dir,
            "ckpt_file": resolve_checkpoint(args),
            "attention_stride_pattern": model.model.attention_stride_pattern,
            "residual_source_pattern": model.model.residual_source_pattern,
            "attn_implementation": args.attn_implementation,
        },
    }


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print("Test Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print(f"device: {device}", flush=True)

    model = prepare_model(args, device)
    dataloader = prepare_data(args)
    summary = evaluate(model, dataloader, args, device)

    print("Teacher-forced decode KV test summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Wrote summary to {output_path}", flush=True)


if __name__ == "__main__":
    main()
