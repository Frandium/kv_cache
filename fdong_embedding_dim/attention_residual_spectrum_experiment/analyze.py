import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="fdong/Qwen3-0.6B")
    parser.add_argument("--data-dir", default="/Users/bytedance/Desktop/dclm")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-sequences", type=int, default=8)
    parser.add_argument("--spectral-tokens-per-sequence", type=int, default=64)
    parser.add_argument("--pair-samples-per-sequence", type=int, default=1024)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--pca-niter", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def choose_device(value):
    if value != "auto":
        return torch.device(value)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def choose_dtype(value, device):
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[value]
    if device.type == "cpu" and dtype != torch.float32:
        return torch.float32
    return dtype


def parse_layers(value, count):
    if value == "all":
        return list(range(count))
    layers = [int(item) for item in value.split(",") if item.strip()]
    if any(layer < 0 or layer >= count for layer in layers):
        raise ValueError(f"layers must be in [0, {count})")
    return layers


def iter_documents(data_dir):
    for path in sorted(Path(data_dir).glob("part-*.txt")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    value = line
                if isinstance(value, str) and value.strip():
                    yield value.strip(), str(path)


def build_sequences(tokenizer, data_dir, seq_len, count):
    sequences = []
    metadata = []
    for text, path in iter_documents(data_dir):
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if ids.numel() < seq_len:
            continue
        sequences.append(ids[:seq_len].clone())
        metadata.append({"path": path, "chars": len(text), "tokens": int(ids.numel())})
        if len(sequences) == count:
            break
    if len(sequences) < count:
        raise RuntimeError(f"only found {len(sequences)} documents with at least {seq_len} tokens")
    return sequences, metadata


class Capture:
    def __init__(self, model, layers):
        self.x = {}
        self.a = {}
        self.handles = []
        for layer_idx in layers:
            layer = model.layers[layer_idx]
            self.handles.append(layer.register_forward_pre_hook(self._pre_hook(layer_idx)))
            self.handles.append(layer.self_attn.register_forward_hook(self._attn_hook(layer_idx)))

    def _pre_hook(self, layer_idx):
        def hook(_module, args):
            self.x[layer_idx] = args[0].detach().to("cpu", torch.float16)
        return hook

    def _attn_hook(self, layer_idx):
        def hook(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            self.a[layer_idx] = value.detach().to("cpu", torch.float16)
        return hook

    def clear(self):
        self.x.clear()
        self.a.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def cosine_mean(left, right):
    return float((F.normalize(left.float(), dim=-1) * F.normalize(right.float(), dim=-1)).sum(-1).mean())


def representation_metrics(values, pair_count, seed):
    # values: [N, S, D]
    values = values.float()
    nseq, seqlen, _ = values.shape
    global_mean = values.reshape(-1, values.shape[-1]).mean(0, keepdim=True)
    centered = values - global_mean
    generator = torch.Generator().manual_seed(seed)
    output = {}
    for name, tensor in (("raw", values), ("centered", centered)):
        output[f"{name}_adjacent_cosine"] = cosine_mean(tensor[:, :-1], tensor[:, 1:])
        within_left = []
        within_right = []
        between_left = []
        between_right = []
        for seq_idx in range(nseq):
            count = min(pair_count, seqlen)
            i = torch.randint(0, seqlen, (count,), generator=generator)
            j = torch.randint(0, seqlen, (count,), generator=generator)
            within_left.append(tensor[seq_idx, i])
            within_right.append(tensor[seq_idx, j])
            other = (seq_idx + 1) % nseq
            between_left.append(tensor[seq_idx, i])
            between_right.append(tensor[other, j])
        within = cosine_mean(torch.cat(within_left), torch.cat(within_right))
        between = cosine_mean(torch.cat(between_left), torch.cat(between_right))
        output[f"{name}_within_cosine"] = within
        output[f"{name}_between_cosine"] = between
        output[f"{name}_sequence_gap"] = within - between

        centroids = F.normalize(tensor[:, ::2].mean(1), dim=-1)
        queries = F.normalize(tensor[:, 1::2], dim=-1)
        logits = torch.einsum("nsd,md->nsm", queries, centroids)
        labels = torch.arange(nseq)[:, None].expand(nseq, queries.shape[1])
        output[f"{name}_sequence_centroid_accuracy"] = float((logits.argmax(-1) == labels).float().mean())
    return output


def pca_basis(matrix, rank, niter):
    matrix = matrix.float()
    q = min(rank, matrix.shape[0] - 1, matrix.shape[1])
    if q < 1:
        raise ValueError("not enough samples for PCA")
    _, singular_values, basis = torch.pca_lowrank(matrix, q=q, center=False, niter=niter)
    return basis, singular_values


def projected_energy(matrix, basis):
    return float((matrix @ basis).square().sum())


def subspace_overlap(left, right):
    k = min(left.shape[1], right.shape[1])
    return float((left[:, :k].T @ right[:, :k]).square().sum() / k)


def band_metrics(a_values, h_basis, k_common, k_middle, pair_count, seed):
    centered = a_values.float() - a_values.float().reshape(-1, a_values.shape[-1]).mean(0)
    total = float(centered.square().sum())
    bands = {
        "common": h_basis[:, :k_common],
        "middle": h_basis[:, k_common:k_middle],
    }
    result = {}
    reconstructed_top = torch.zeros_like(centered)
    for name, basis in bands.items():
        projected = centered @ basis
        reconstructed = projected @ basis.T
        reconstructed_top += reconstructed
        result[f"attention_energy_{name}"] = float(reconstructed.square().sum()) / max(total, 1e-12)
        metrics = representation_metrics(reconstructed, pair_count, seed)
        result[f"attention_{name}_centered_sequence_gap"] = metrics["centered_sequence_gap"]
        result[f"attention_{name}_centered_sequence_accuracy"] = metrics["centered_sequence_centroid_accuracy"]
    tail = centered - reconstructed_top
    result["attention_energy_tail"] = float(tail.square().sum()) / max(total, 1e-12)
    metrics = representation_metrics(tail, pair_count, seed)
    result["attention_tail_centered_sequence_gap"] = metrics["centered_sequence_gap"]
    result["attention_tail_centered_sequence_accuracy"] = metrics["centered_sequence_centroid_accuracy"]
    return result


def analyze_layer(layer_idx, x_values, a_values, args):
    h_values = x_values.float() + a_values.float()
    row = {"layer": layer_idx}
    for name, values in (("x", x_values), ("a", a_values), ("h", h_values)):
        metrics = representation_metrics(values, args.pair_samples_per_sequence, args.seed + layer_idx)
        row.update({f"{name}_{key}": value for key, value in metrics.items()})

    nseq, seqlen, dim = h_values.shape
    per_seq = min(args.spectral_tokens_per_sequence, seqlen)
    positions = torch.linspace(0, seqlen - 1, per_seq).long()
    spectral = {}
    for name, values in (("x", x_values), ("a", a_values), ("h", h_values)):
        sampled = values[:, positions].reshape(-1, dim).float()
        sampled -= sampled.mean(0, keepdim=True)
        spectral[name] = sampled

    max_rank = max(2, math.ceil(0.10 * dim))
    bases = {}
    singular_values = {}
    for name in ("x", "a", "h"):
        bases[name], singular_values[name] = pca_basis(spectral[name], max_rank, args.pca_niter)

    total_h = float(spectral["h"].square().sum())
    for ratio in (0.01, 0.05, 0.10):
        k = max(1, math.ceil(ratio * dim))
        key = f"top{int(ratio * 100)}"
        optimal = projected_energy(spectral["h"], bases["h"][:, :k])
        x_energy = projected_energy(spectral["h"], bases["x"][:, :k])
        a_energy = projected_energy(spectral["h"], bases["a"][:, :k])
        row[f"hx_overlap_{key}"] = subspace_overlap(bases["h"][:, :k], bases["x"][:, :k])
        row[f"ha_overlap_{key}"] = subspace_overlap(bases["h"][:, :k], bases["a"][:, :k])
        row[f"h_recovery_x_{key}"] = x_energy / max(optimal, 1e-12)
        row[f"h_recovery_a_{key}"] = a_energy / max(optimal, 1e-12)
        row[f"h_optimal_energy_{key}"] = optimal / max(total_h, 1e-12)

    k_common = max(1, math.ceil(0.01 * dim))
    k_middle = max(k_common + 1, math.ceil(0.10 * dim))
    row.update(
        band_metrics(
            a_values.float(), bases["h"], k_common, k_middle,
            args.pair_samples_per_sequence, args.seed + layer_idx,
        )
    )
    row["x_to_a_norm_ratio"] = float(x_values.float().norm() / max(float(a_values.float().norm()), 1e-12))
    return row


def write_csv(rows, path):
    fields = sorted({key for row in rows for key in row})
    fields.remove("layer")
    fields.insert(0, "layer")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    result = {}
    keys = [key for key in rows[0] if key != "layer"]
    for key in keys:
        values = [float(row[key]) for row in rows]
        result[key] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    return result


def plot(rows, output_dir):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [row["layer"] for row in rows]

    def save(lines, ylabel, filename, ylim=None):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for label, key in lines:
            ax.plot(layers, [row[key] for row in rows], marker="o", markersize=3, label=label)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        if ylim:
            ax.set_ylim(*ylim)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=180)
        plt.close(fig)

    save(
        [("Residual input X", "x_centered_adjacent_cosine"),
         ("Attention output A", "a_centered_adjacent_cosine"),
         ("Post-attention H", "h_centered_adjacent_cosine")],
        "Centered adjacent-token cosine", "continuity_by_layer.png",
    )
    save(
        [("Residual input X", "x_centered_sequence_centroid_accuracy"),
         ("Attention output A", "a_centered_sequence_centroid_accuracy"),
         ("Post-attention H", "h_centered_sequence_centroid_accuracy")],
        "Sequence centroid accuracy", "sequence_accuracy_by_layer.png", (0, 1.02),
    )
    save(
        [("H recovered by X top-5%", "h_recovery_x_top5"),
         ("H recovered by A top-5%", "h_recovery_a_top5")],
        "Fraction of optimal H top-5% energy", "top_subspace_dominance_by_layer.png", (0, 1.05),
    )
    save(
        [("Common 0-1%", "attention_energy_common"),
         ("Middle 1-10%", "attention_energy_middle"),
         ("Tail 10-100%", "attention_energy_tail")],
        "Attention energy fraction in H spectrum", "attention_energy_in_h_bands.png", (0, 1.02),
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for label, key, width in (
        ("Common 0-1%", "attention_energy_common", 0.01),
        ("Middle 1-10%", "attention_energy_middle", 0.09),
        ("Tail 10-100%", "attention_energy_tail", 0.90),
    ):
        ax.plot(
            layers,
            [row[key] / width for row in rows],
            marker="o",
            markersize=3,
            label=label,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Isotropic baseline")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Energy density / isotropic density")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "attention_energy_density_by_band.png"), dpi=180)
    plt.close(fig)
    save(
        [("Common 0-1%", "attention_common_centered_sequence_gap"),
         ("Middle 1-10%", "attention_middle_centered_sequence_gap"),
         ("Tail 10-100%", "attention_tail_centered_sequence_gap")],
        "Within-sequence minus between-sequence cosine", "attention_band_sequence_gap.png",
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    print(f"[setup] device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    sequences, metadata = build_sequences(tokenizer, args.data_dir, args.seq_len, args.num_sequences)
    print(f"[data] sequences={len(sequences)} seq_len={args.seq_len}")

    model = AutoModel.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=dtype,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device).eval()
    layers = parse_layers(args.layers, len(model.layers))
    captured = {layer: {"x": [], "a": []} for layer in layers}
    hooks = Capture(model, layers)
    try:
        with torch.no_grad():
            for seq_idx, sequence in enumerate(sequences):
                hooks.clear()
                model(input_ids=sequence.unsqueeze(0).to(device), use_cache=False, return_dict=True)
                for layer in layers:
                    if layer not in hooks.x or layer not in hooks.a:
                        raise RuntimeError(f"missing hook output for layer {layer}")
                    captured[layer]["x"].append(hooks.x[layer].squeeze(0))
                    captured[layer]["a"].append(hooks.a[layer].squeeze(0))
                print(f"[forward] {seq_idx + 1}/{len(sequences)}")
    finally:
        hooks.close()

    rows = []
    for layer in layers:
        print(f"[analyze] layer={layer}")
        x_values = torch.stack(captured[layer]["x"])
        a_values = torch.stack(captured[layer]["a"])
        rows.append(analyze_layer(layer, x_values, a_values, args))

    write_csv(rows, os.path.join(output_dir, "layer_metrics.csv"))
    summary = {
        "config": vars(args),
        "device": str(device),
        "effective_dtype": str(dtype),
        "documents": metadata,
        "aggregate": aggregate(rows),
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if not args.no_plots:
        plot(rows, output_dir)
    print(f"[done] {output_dir}")


if __name__ == "__main__":
    main()
