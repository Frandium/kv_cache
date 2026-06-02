# Report Card: Geometric Inhibition Stabilizes Slot-Aligned Routing

## Source Files

Anchor: `05_01_geometric_inhibition_anchor.md`

Summary: `summary.md`

Detailed: `detailed.md`

Figures / Tables: `figures/`, `tables/`

## 0. Executive Summary

本轮目标：

用一个组会可讲的简化实验判断：slot-stable initialization 加 geometric inhibition 是否能稳定 top-1 MoE routing。

最小机制审计：

只比较 dot/cosine router 的 random、slot-init、slot-init+geo 三种状态；去掉 expert warmup，所有 router 使用 `bias=False`。

清晰的假设：

如果 slot prototype 中有可用几何信息，那么 slot-stable init 应该提高 step-0 route-slot NMI；如果 geometric inhibition 有额外作用，那么在 init 之后 final route-slot NMI 和 seed stability 应继续提升。

关键发现：

1. Q1: slot-stable init 成功。C1 vs C0 step-0 NMI 为 0.242 vs 0.009；C4 vs C3 为 0.242 vs 0.008。
2. Q2: geometric inhibition 有额外贡献。C2/C5 final NMI 都达到 1.000，且 seed std 为 0。
3. Q3: 不是只提高 confidence。C2 vs C1 confidence +0.028，同时 NMI +0.554；C5 vs C4 confidence +0.029，同时 NMI +0.434。
4. Q5: 没有 accuracy tradeoff；所有条件 target accuracy 都是 1.000。

当前结论：

在 uniform multi-B synthetic 中，给定外部 slot assignment $a(s,i)=s$，geometric inhibition 可以在 slot-stable init 之外稳定 slot-aligned routing。

结论边界：

不能 claim label-free specialization、Zipfian robustness、real-data transfer，也不能说 expert utility 已完整解决。

下一步：

组会中将其作为 clean mechanism result；研究上下一步应测 Zipfian 或 less-oracle prototype。

## 1. Research Process Update

| item | content |
|---|---|
| Previous mainline | Ordinary top-1 selected-gate routing can lock in early assignments; full 05 showed warmup/inhibition can align route and utility under external slot assignment. |
| New probe | Simplified geometric inhibition experiment for meeting-facing routing-stability evidence. |
| New evidence | Slot init improves step-0 NMI; geometric inhibition drives final NMI to 1.000 for dot and cosine; confidence-only rival is weakened. |
| Knowledge update | Prototype geometry is useful, but init alone is not stable enough; explicit margin/separation makes routing reliable. |
| Next decision | Present as routing stabilization, then test frequency imbalance or less-oracle assignment. |

## 2. Terms Used Here

- `slot-stable initialization`: initialize router rows from centered slot prototypes.
- `geometric inhibition`: token-level margin plus router-center separation.
- `route-slot NMI`: alignment between slot label and selected expert.
- `selected gate confidence`: softmax probability of the selected top-1 expert.
- `confidence-only rival`: route looks more certain only because gates sharpen, not because slot assignment improves.

## 3. Key Figures

### Figure 1: Route-slot NMI trajectory

![Route-slot NMI trajectory](figures/route_slot_nmi_trajectory.png)

What to see:

C2 and C5 go to final route-slot NMI 1.000, while C1/C4 remain unstable and seed-dependent.

Supports:

Geometric inhibition stabilizes routing beyond slot-stable initialization.

Cannot prove:

It does not prove label-free specialization or full expert utility.

### Figure 2: Selected gate confidence trajectory

![Selected gate confidence trajectory](figures/selected_gate_confidence_trajectory.png)

What to see:

C2/C5 confidence increases, but the important point is that NMI also increases.

Supports:

The result is not merely confidence sharpening.

Cannot prove:

It does not prove confidence is the causal mechanism behind task performance.

### Figure 3: Route heatmap

![Route-slot heatmap](figures/route_slot_heatmap_step0_final.png)

What to see:

Slot-init improves step-0 structure; geometric inhibition makes final heatmaps cleanly slot-aligned.

Supports:

Prototype geometry contains usable slot information and geometric inhibition prevents route mixing.

Cannot prove:

It does not show how to get slot labels in real language data.

## 4. Current Claim

Given external slot assignment $a(s,i)=s$, geometric inhibition stabilizes slot-aligned top-1 routing in uniform multi-B synthetic data, without target accuracy tradeoff.

## 5. Claim Boundary

Can claim:

This is clean evidence for routing-geometry stabilization under supervised prototype assignment.

Cannot claim:

No claim yet about label-free discovery, Zipfian robustness, real-data transfer, or complete utility-aligned specialization.

## 6. Next Step

For group meeting, present this as the minimal positive mechanism result. For the research line, the next decisive test is whether this survives frequency imbalance or a less-oracle prototype construction.
