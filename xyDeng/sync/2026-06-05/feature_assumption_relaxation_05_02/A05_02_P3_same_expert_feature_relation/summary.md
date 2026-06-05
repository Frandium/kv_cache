# Result Summary: A05_02_P3_same_expert_feature_relation Same-Expert Feature Relation

Anchor: `Projects/from-attention-to-search/main/problem_anchors/05_02_feature_assumption_relaxation_anchor.md`

Story: P3 tested whether same-expert features share family structure more than load-matched random grouping.

## 0. Closure Summary

目的：
放松 minimal baseline 的第三个假设，不只看 feature 是否稳定 route，而是看 same-expert features 是否有 family structure。

假设 / 问题：
P3a tests input-family-only grouping. P3b tests input-family plus shared family target. 主指标是 family purity over load-matched random baseline, 即同负载随机基线上的 family purity 增量 $\Delta_{\mathrm{family}}$。

结论：
P3a 和 P3b 都不能判为 positive。所有条件 target accuracy 都是 1.0，说明任务学会；但 family purity 增量小且 seed 不稳定。P3b 没有比 P3a 更强，说明 shared family-level target utility 没有稳定诱导 ordinary top-1 router 形成 family grouping。

关键证据：

| Case | Condition | Target acc. | Active experts | Max load | Family purity actual | Random mean | Delta |
|---|---|---:|---:|---:|---:|---:|---:|
| P3a | B0 | 1.000 | 3.00 | 0.563 | 0.479 | 0.424 | 0.055 |
| P3a | I1 | 1.000 | 3.00 | 0.646 | 0.458 | 0.411 | 0.047 |
| P3b | B0 | 1.000 | 2.67 | 0.729 | 0.458 | 0.372 | 0.087 |
| P3b | I1 | 1.000 | 2.00 | 0.792 | 0.417 | 0.327 | 0.090 |

Seed-level boundary:
The positive-looking P3b deltas come mainly from seed `20260522`; seed `20260521` collapses to no family structure and seed `20260523` is near zero or negative. This is not seed-stable evidence.

结论边界：
This result supports the weaker interpretation that same-expert assignment is closer to arbitrary or early-locked buckets than reliable family grouping. It cannot prove no router can group families, only that this ordinary top-1 setup did not do so reliably.

下一步：
Do not move from same-feature route consistency to family-level specialization claims. The next decision should test whether adding an explicit non-collapse guard or utility-binding intervention changes P3b; otherwise ordinary top-1 should be treated as insufficient for structured same-expert grouping.

## 1. Validity Check

Target accuracy passed in all four condition groups. Therefore the negative P3 result is not a target-learning failure.

The main validity weakness is partial concentration or collapse in some seeds, especially P3b. This weakens positive grouping claims.

## 2. Primary Result

![Family purity vs load-matched random](figures/family_purity_vs_load_matched_random.png)

Interpretation:
Actual family purity is only slightly above the load-matched random baseline on average, and the seed-level table shows this is not stable.

## 3. Required Figures

![P3a expert x feature heatmap](figures/P3a_expert_feature_heatmap.png)

![P3b expert x feature heatmap](figures/P3b_expert_feature_heatmap.png)

![P3a expert x family composition heatmap](figures/P3a_expert_family_composition_heatmap.png)

![P3b expert x family composition heatmap](figures/P3b_expert_family_composition_heatmap.png)

![Expert load histogram](figures/expert_load_histogram.png)

![Target accuracy table](figures/target_accuracy_table.png)

## 4. Files

- Full detailed record: `detailed.md`
- Condition table: `tables/summary_by_condition.csv`
- Family relation table: `tables/family_relation_by_condition.csv`
- Seed-level family relation: `tables/family_relation_by_seed.csv`
- Route matrix: `tables/route_matrix_by_seed.csv`
- Expert load table: `tables/expert_load_by_seed.csv`
