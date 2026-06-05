# Report Card: 05_02 Feature Assumption Relaxation

## Source Files

Anchor:
`05_02_feature_assumption_relaxation_anchor.md`

Summary:
`A05_02_P1_compositional_token_feature_routing/summary.md`
`A05_02_P3_same_expert_feature_relation/summary.md`

Detailed:
`A05_02_P1_compositional_token_feature_routing/detailed.md`
`A05_02_P3_same_expert_feature_relation/detailed.md`

Figures / Tables:
`A05_02_P1_compositional_token_feature_routing/figures/`
`A05_02_P1_compositional_token_feature_routing/tables/`
`A05_02_P3_same_expert_feature_relation/figures/`
`A05_02_P3_same_expert_feature_relation/tables/`

## 0. Executive Summary

本轮目标：

从 advisor's minimal baseline setting 出发，放松两个假设：P1 放松 single-feature token；P3 检查 same-expert features 是否有 family structure。

最小机制审计：

先看 target 是否学会，再看 routing 是否 non-collapsed，最后才解释 route-axis NMI 或 family purity。

清晰的假设：

P1 若 positive，route axis 应随 target rule 变成 $S1$、$S2$、$(S1,S2)$。P3 若 positive，same-expert features 应比同负载随机分组更共享 family。

关键发现：

P1/P3 的 target accuracy 都是 1.0，但 routing 证据不支持 robust specialization。P1 的 C1/C2 多数 collapse；P3 的 family purity gain 小且 seed 不稳定。

当前结论：

ordinary top-1 routing 在这些 relaxed assumptions 下更接近 collapse / arbitrary bucket / early-locking bucket，而不是稳定的 feature-level specialization。

结论边界：

不能 claim expert utility specialization、Zipfian robustness、real-data generalization 或新 router design。

下一步：

决定是否用 explicit non-collapse / utility-binding intervention 重测 P3b；否则 P2 继续 parked。

## 1. Research Process Update

| item | content |
|---|---|
| Previous mainline | same-feature route consistency is weak because collapse can fake consistency |
| New probe | P1 compositional token routing and P3 same-expert feature relation |
| New evidence | target learning succeeds; route structure is collapsed or weak and seed-unstable |
| Knowledge update | ordinary top-1 is not enough for robust structured feature grouping under 05_02 |
| Next decision | test explicit guard/intervention on P3b or stop this ordinary top-1 line |

## 2. Terms Used Here

- `route consistency`: 同一 feature 的样本是否集中进入同一 expert。
- `non-collapse`: routing 使用多个 experts，且不是一个 expert 吸收绝大多数样本。
- `NMI`: 归一化互信息，用来衡量 route 和候选 feature axis 的对齐程度。
- `family purity`: 同一 expert 内 features 共享 family 的比例。
- `load-matched random baseline`: 保持 expert 负载不变后随机打乱 family label 得到的比较基线。

## 3. Key Figures

### Figure 1: P1 B0 target-rule x route-axis NMI

![P1 B0 target-rule x route-axis NMI](A05_02_P1_compositional_token_feature_routing/figures/B0_target_rule_route_axis_nmi.png)

What to see:

C3-B0 的 route 更偏向 $(S1,S2)$，但 C1/C2 因 collapse 不能作为稳定正证据。

Supports:

组合目标可以留下 route-axis clue。

Cannot prove:

不能证明 ordinary top-1 会稳定按 target-relevant factor route。

### Figure 2: P1 expert load histogram

![P1 expert load histogram](A05_02_P1_compositional_token_feature_routing/figures/expert_load_histogram.png)

What to see:

C1/C2 多数 seed 使用单 expert 或高度集中。

Supports:

高 target accuracy 不能替代 non-collapse guard。

Cannot prove:

不能单独说明 collapse 的具体训练动力学原因。

### Figure 3: P3 family purity vs load-matched random

![P3 family purity vs load-matched random](A05_02_P3_same_expert_feature_relation/figures/family_purity_vs_load_matched_random.png)

What to see:

actual family purity 只比同负载随机基线略高，且逐 seed 不稳定。

Supports:

P3a/P3b 都不是稳定 positive。

Cannot prove:

不能排除带显式 non-collapse 或 utility-binding intervention 的 router 会形成 family grouping。

### Figure 4: P3 expert load histogram

![P3 expert load histogram](A05_02_P3_same_expert_feature_relation/figures/expert_load_histogram.png)

What to see:

P3b 的部分 positive-looking family purity 与更强 concentration 同时出现。

Supports:

P3b 不能被解释为可靠的 shared-target utility grouping。

Cannot prove:

不能证明所有 shared-target settings 都会失败。

## 4. Current Claim

The current safe claim is: under P1/P3 relaxed assumptions, ordinary top-1 can learn the target but does not robustly produce non-collapsed, structured feature routing.

## 5. Claim Boundary

Can claim:

P1 is not broadly positive; P3a/P3b are not stable positive evidence for family grouping.

Cannot claim:

No possible router can group features; no real-data conclusion; no utility specialization.

## 6. Next Step

Make one decision: either test a non-collapse / utility-binding intervention on P3b, or stop ordinary top-1 assumption relaxation here and keep P2 parked.
