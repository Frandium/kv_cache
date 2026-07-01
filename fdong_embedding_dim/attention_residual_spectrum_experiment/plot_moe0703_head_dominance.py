import csv
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig2")

import matplotlib.pyplot as plt


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONTINUITY_CSV = os.path.join(
    os.path.dirname(__file__), "results_seq1024", "layer_metrics.csv"
)
ATTRIBUTION_CSV = os.path.join(
    os.path.dirname(__file__),
    "results_band_attribution_seq1024",
    "band_attribution_summary_layers1_27.csv",
)
OUTPUT = os.path.join(ROOT, "main_swapmoe", "assets", "attention_residual_head_dominance.png")


def mean(values):
    return sum(values) / len(values)


with open(CONTINUITY_CSV, newline="", encoding="utf-8") as handle:
    continuity_rows = [row for row in csv.DictReader(handle) if int(row["layer"]) > 0]

with open(ATTRIBUTION_CSV, newline="", encoding="utf-8") as handle:
    attribution_rows = {row["band"]: row for row in csv.DictReader(handle)}

top1 = attribution_rows["0_1"]
continuity = [
    mean([float(row["x_centered_adjacent_cosine"]) for row in continuity_rows]),
    mean([float(row["a_centered_adjacent_cosine"]) for row in continuity_rows]),
]
source_fraction = [
    float(top1["x_source_energy_fraction"]),
    float(top1["a_source_energy_fraction"]),
]
final_attribution = [
    float(top1["x_shapley_share_in_h_band"]),
    float(top1["a_shapley_share_in_h_band"]),
]

x_color = "#315C78"
a_color = "#D65A3A"
colors = [x_color, a_color]

fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
fig.suptitle(
    "Why attention continuity does not dominate standard MoE routing",
    fontsize=16,
    fontweight="bold",
    y=1.02,
)

panels = [
    (
        continuity,
        "1. Cross-token continuity",
        "Centered adjacent-token cosine",
        (0, 0.62),
        lambda value: f"{value:.3f}",
    ),
    (
        source_fraction,
        "2. Energy landing in H top 1%",
        "Fraction of each source's energy",
        (0, 1.02),
        lambda value: f"{100 * value:.2f}%",
    ),
    (
        final_attribution,
        "3. Contribution inside H top 1%",
        "Shapley energy attribution",
        (0, 1.02),
        lambda value: f"{100 * value:.2f}%",
    ),
]

for axis, (values, title, ylabel, ylim, formatter) in zip(axes, panels):
    bars = axis.bar(["Residual X", "Attention A"], values, color=colors, width=0.62)
    axis.set_title(title, fontsize=12, fontweight="bold", pad=12)
    axis.set_ylabel(ylabel)
    axis.set_ylim(*ylim)
    axis.grid(axis="y", alpha=0.22)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (ylim[1] * 0.025),
            formatter(value),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

axes[0].annotate(
    "A is more continuous",
    xy=(1, continuity[1]),
    xytext=(0.25, 0.585),
    arrowprops={"arrowstyle": "->", "color": a_color, "lw": 1.5},
    color=a_color,
    fontsize=10,
    fontweight="bold",
)
axes[2].annotate(
    "A is almost masked",
    xy=(1, final_attribution[1]),
    xytext=(0.25, 0.30),
    arrowprops={"arrowstyle": "->", "color": a_color, "lw": 1.5},
    color=a_color,
    fontsize=10,
    fontweight="bold",
)

fig.text(
    0.5,
    -0.01,
    "Qwen3-0.6B, DCLM 1024-token sequences, layers 1-27",
    ha="center",
    fontsize=10,
    color="#444444",
)
fig.tight_layout()
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
plt.close(fig)
print(OUTPUT)
