import argparse
import csv
import importlib.util
import json
import math
import os

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BAND_EDGES = (0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00)
BAND_NAMES = ("0_1", "1_2", "2_5", "5_10", "10_20", "20_50", "50_100")


def load_base_module():
    path = os.path.join(os.path.dirname(__file__), "analyze.py")
    spec = importlib.util.spec_from_file_location("attention_residual_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="fdong/Qwen3-0.6B")
    parser.add_argument("--data-dir", default="/Users/bytedance/Desktop/dclm")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-sequences", type=int, default=8)
    parser.add_argument("--sample-pairs-per-sequence", type=int, default=32)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--analysis-device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def paired_positions(seq_len, pair_count):
    pair_count = min(pair_count, seq_len // 2)
    starts = torch.linspace(0, seq_len - 2, pair_count).round().long()
    positions = torch.stack((starts, starts + 1), dim=1).reshape(-1)
    return positions


def centered_sample(values, positions):
    values = values.float()
    mean = values.reshape(-1, values.shape[-1]).mean(0, keepdim=True)
    return values[:, positions] - mean


def top50_basis(matrix):
    flat = matrix.reshape(-1, matrix.shape[-1]).float()
    target = math.ceil(0.50 * flat.shape[-1])
    rank = min(target, flat.shape[0] - 1, flat.shape[1])
    _, _, basis = torch.pca_lowrank(flat, q=rank, center=False, niter=2)
    return basis.contiguous()


def explicit_band_ranges(dim, available_rank):
    ranges = []
    for name, lo, hi in zip(BAND_NAMES[:-1], BAND_EDGES[:-2], BAND_EDGES[1:-1]):
        start = min(math.ceil(lo * dim), available_rank)
        end = min(math.ceil(hi * dim), available_rank)
        ranges.append((name, start, end))
    ranges.append((BAND_NAMES[-1], min(math.ceil(0.50 * dim), available_rank), None))
    return ranges


def component(values, basis, start, end, top50_basis):
    if end is None:
        coords = values @ top50_basis
        return values - coords @ top50_basis.T
    if end <= start:
        return torch.zeros_like(values)
    current = basis[:, start:end]
    return (values @ current) @ current.T


def sequence_metrics(values):
    # positions are stored as adjacent pairs: [p0,p0+1,p1,p1+1,...]
    values = values.float()
    nseq = values.shape[0]
    left = values[:, 0::2]
    right = values[:, 1::2]
    adjacent = BASE.cosine_mean(left, right)
    within = BASE.cosine_mean(values[:, ::2], values[:, 1::2])
    between = BASE.cosine_mean(values[:-1].reshape(-1, values.shape[-1]), values[1:].reshape(-1, values.shape[-1]))
    centroids = F.normalize(values[:, 0::2].mean(1), dim=-1)
    queries = F.normalize(values[:, 1::2], dim=-1)
    logits = torch.einsum("nsd,md->nsm", queries, centroids)
    labels = torch.arange(nseq)[:, None].expand(nseq, queries.shape[1])
    accuracy = float((logits.argmax(-1) == labels).float().mean())
    return {
        "adjacent_cosine": adjacent,
        "between_cosine": between,
        "sequence_gap": within - between,
        "sequence_accuracy": accuracy,
    }


def energy(values):
    return float(values.square().sum())


def analyze_layer(layer, x_full, a_full, positions, analysis_device):
    h_full = x_full.float() + a_full.float()
    x = centered_sample(x_full, positions).to(analysis_device)
    a = centered_sample(a_full, positions).to(analysis_device)
    h = centered_sample(h_full, positions).to(analysis_device)
    dim = h.shape[-1]

    h_basis = top50_basis(h)
    a_basis = top50_basis(a)
    available = min(h_basis.shape[1], a_basis.shape[1])
    top50_end = min(math.ceil(0.50 * dim), available)
    h_top50 = h_basis[:, :top50_end]
    a_top50 = a_basis[:, :top50_end]
    ranges = explicit_band_ranges(dim, available)

    ex_total = energy(x)
    ea_total = energy(a)
    eh_total = energy(h)
    rows = []
    h_components = {}
    a_in_h_components = {}
    a_own_components = {}

    for name, start, end in ranges:
        xb = component(x, h_basis, start, end, h_top50)
        ab = component(a, h_basis, start, end, h_top50)
        hb = xb + ab
        ex = energy(xb)
        ea = energy(ab)
        eh = energy(hb)
        cross = float(2.0 * (xb * ab).sum())
        denom_sources = max(ex + ea, 1e-12)
        row = {
            "layer": layer,
            "band": name,
            "x_source_energy_fraction": ex / max(ex_total, 1e-12),
            "a_source_energy_fraction": ea / max(ea_total, 1e-12),
            "h_energy_fraction": eh / max(eh_total, 1e-12),
            "a_norm_share_in_band": ea / denom_sources,
            "x_norm_share_in_band": ex / denom_sources,
            "cross_term_over_h_energy": cross / max(eh, 1e-12),
            "a_shapley_share_in_h_band": (ea + 0.5 * cross) / max(eh, 1e-12),
            "x_shapley_share_in_h_band": (ex + 0.5 * cross) / max(eh, 1e-12),
            "a_to_x_norm_ratio": math.sqrt(ea / max(ex, 1e-12)),
        }
        for prefix, values in (("a", ab), ("h", hb)):
            row.update({f"{prefix}_{key}": value for key, value in sequence_metrics(values).items()})
        rows.append(row)
        h_components[name] = hb
        a_in_h_components[name] = ab
        a_own_components[name] = component(a, a_basis, start, end, a_top50)

    transfer = []
    for source_name, source in a_own_components.items():
        source_energy = energy(source)
        source_metrics = sequence_metrics(source)
        for target_name, start, end in ranges:
            projected = component(source, h_basis, start, end, h_top50)
            transfer.append({
                "layer": layer,
                "a_band": source_name,
                "h_band": target_name,
                "energy_fraction": energy(projected) / max(source_energy, 1e-12),
                "a_band_adjacent_cosine": source_metrics["adjacent_cosine"],
                "a_band_sequence_gap": source_metrics["sequence_gap"],
                "a_band_sequence_accuracy": source_metrics["sequence_accuracy"],
            })
    return rows, transfer


def write_csv(rows, path):
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, keys):
    grouped = {}
    for row in rows:
        group = tuple(row[key] for key in keys)
        grouped.setdefault(group, []).append(row)
    result = []
    for group, items in grouped.items():
        out = dict(zip(keys, group))
        for key in items[0]:
            if key in keys or key == "layer":
                continue
            out[key] = sum(float(item[key]) for item in items) / len(items)
        result.append(out)
    return result


def plot_band_rows(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate_rows = aggregate([row for row in rows if int(row["layer"]) > 0], ["band"])
    labels = [row["band"].replace("_", "-") + "%" for row in aggregate_rows]
    x = list(range(len(labels)))

    def grouped(keys, ylabel, filename, ylim=None):
        fig, ax = plt.subplots(figsize=(10, 5))
        width = 0.8 / len(keys)
        for index, (label, key) in enumerate(keys):
            offset = (index - (len(keys) - 1) / 2) * width
            ax.bar([value + offset for value in x], [row[key] for row in aggregate_rows], width, label=label)
        ax.set_xticks(x, labels)
        ax.set_xlabel("H spectral band")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=180)
        plt.close(fig)

    grouped(
        [("Fraction of X energy", "x_source_energy_fraction"),
         ("Fraction of A energy", "a_source_energy_fraction")],
        "Source energy fraction in H band", "x_a_source_energy_by_h_band.png", (0, 1),
    )
    grouped(
        [("A norm share", "a_norm_share_in_band"),
         ("A Shapley energy share", "a_shapley_share_in_h_band")],
        "A contribution inside H band", "a_contribution_inside_h_band.png", (0, 1),
    )
    grouped(
        [("A component", "a_adjacent_cosine"),
         ("Final H component", "h_adjacent_cosine")],
        "Adjacent-token cosine", "band_continuity.png",
    )


def plot_transfer(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate_rows = aggregate([row for row in rows if int(row["layer"]) > 0], ["a_band", "h_band"])
    index = {(row["a_band"], row["h_band"]): row for row in aggregate_rows}
    matrix = torch.tensor([
        [index[(source, target)]["energy_fraction"] for target in BAND_NAMES]
        for source in BAND_NAMES
    ])
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    labels = [name.replace("_", "-") + "%" for name in BAND_NAMES]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("H spectral band")
    ax.set_ylabel("A own spectral band")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if matrix[i, j] < 0.45 else "black")
    fig.colorbar(image, ax=ax, label="Energy fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "a_band_to_h_band_transfer.png"), dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = BASE.choose_device(args.device)
    analysis_device = torch.device(args.analysis_device)
    dtype = BASE.choose_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    sequences, metadata = BASE.build_sequences(tokenizer, args.data_dir, args.seq_len, args.num_sequences)
    model = AutoModel.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=dtype,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device).eval()
    layers = BASE.parse_layers(args.layers, len(model.layers))
    captured = {layer: {"x": [], "a": []} for layer in layers}
    hooks = BASE.Capture(model, layers)
    try:
        with torch.no_grad():
            for index, sequence in enumerate(sequences):
                hooks.clear()
                model(input_ids=sequence.unsqueeze(0).to(device), use_cache=False, return_dict=True)
                for layer in layers:
                    captured[layer]["x"].append(hooks.x[layer].squeeze(0))
                    captured[layer]["a"].append(hooks.a[layer].squeeze(0))
                print(f"[forward] {index + 1}/{len(sequences)}")
    finally:
        hooks.close()

    positions = paired_positions(args.seq_len, args.sample_pairs_per_sequence)
    band_rows = []
    transfer_rows = []
    for layer in layers:
        print(f"[analyze] layer={layer}")
        rows, transfers = analyze_layer(
            layer,
            torch.stack(captured[layer]["x"]),
            torch.stack(captured[layer]["a"]),
            positions,
            analysis_device,
        )
        band_rows.extend(rows)
        transfer_rows.extend(transfers)

    write_csv(band_rows, os.path.join(args.output_dir, "band_attribution_by_layer.csv"))
    write_csv(transfer_rows, os.path.join(args.output_dir, "a_to_h_transfer_by_layer.csv"))
    write_csv(aggregate(band_rows, ["band"]), os.path.join(args.output_dir, "band_attribution_summary.csv"))
    write_csv(
        aggregate([row for row in band_rows if int(row["layer"]) > 0], ["band"]),
        os.path.join(args.output_dir, "band_attribution_summary_layers1_27.csv"),
    )
    write_csv(aggregate(transfer_rows, ["a_band", "h_band"]), os.path.join(args.output_dir, "a_to_h_transfer_summary.csv"))
    write_csv(
        aggregate([row for row in transfer_rows if int(row["layer"]) > 0], ["a_band", "h_band"]),
        os.path.join(args.output_dir, "a_to_h_transfer_summary_layers1_27.csv"),
    )
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "device": str(device), "dtype": str(dtype), "documents": metadata}, handle, indent=2)
    plot_band_rows(band_rows, args.output_dir)
    plot_transfer(transfer_rows, args.output_dir)
    print(f"[done] {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
