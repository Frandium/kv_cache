---
parent_node: 05_01_geometric_inhibition_anchor
status: active-after-A05_02_P1-A05_02_P3
source_of_truth_status: project-local-anchor
---

# 05_02 Feature Assumption Relaxation

## 汇报前 Summary

结论：

05_02 的主线不是直接证明 specialization，也不是把 true inhibition 作为中心故事。它的目标是从 advisor's minimal baseline setting 出发，一次放松一个数据假设，观察 route consistency 是否仍然 non-collapsed 且有意义。

A05_02_P1_compositional_token_feature_routing 和 A05_02_P3_same_expert_feature_relation 的当前结论是 negative / bounded：

- P1：target 全部学会，但 C1/C2 多数 seed route collapse；只有 C3-B0 在 non-collapsed-ish 条件下显示 route 更偏向组合轴 $(S1,S2)$。因此 P1 不能整体 claim target-rule-dependent routing。
- P3：P3a 和 P3b target 全部学会，但 family purity over load-matched random baseline 小且 seed 不稳定。P3b 没有比 P3a 更强，因此 shared family target utility 没有稳定诱导 ordinary top-1 family grouping。

minimal baseline setting 的三个关键假设是：

1. token 是 single-feature；
2. feature distribution 是 uniform；
3. 多个 features 进入同一 expert 后，不分析它们之间的关系。

当前执行顺序：

1. P1：compositional token feature routing，已完成；整体不支持，保留 C3-B0 作为弱线索。
2. P3：same-expert feature relation，已完成；不支持稳定 family-structured grouping。
3. P2：Zipfian frequency，先 parked；等 feature definition 和 same-expert relation 清楚后再做。

抑制（inhibition）只保留为一个 condition / candidate mechanism。它可以和 baseline 比较，但不能替代 05_02 的主问题。

结论边界：

即使 P1/P3 positive，也只能说明 assumption relaxation 下 routing 仍可保持 non-collapsed 和可解释结构；不能 claim expert utility specialization、Zipf robustness、real-data generalization 或新 router design。

下一步：

不要把 same-feature route consistency 升级成 specialization claim。下一步若继续主线，应先决定是否引入显式 non-collapse / utility-binding guard 来重测 P3b；否则 ordinary top-1 same-expert assignment 暂时按 arbitrary bucket / early-locking bucket 处理。P2 继续 parked。

## 1. Problem Definition

### Parent problem

ordinary top-1 MoE router 在 next-token prediction 训练下可能形成稳定 route consistency，但这种 consistency 可能来自 collapse、surface shortcut 或 early lock-in，不一定是有意义的 feature-level routing。

### Sharper subproblem

advisor's minimal baseline setting 已经给出一个最小观察面：同一 feature 可以稳定 route 到同一 expert。05_02 不再重复这个基线，而是检查当基线假设被放松时，这种 route consistency 是否仍然：

1. 不坍缩；
2. 与目标相关 feature 对齐；
3. 在 many-to-one feature assignment 中有结构。

### Decision question

当 minimal baseline setting 的假设被逐步放松时，router 产生的是：

- target-relevant, non-collapsed route consistency；
- fixed surface-axis routing；
- $B_i$ identity shortcut；
- arbitrary early-locking bucket；
- collapse；
- 或 structured same-expert grouping。

### Inside boundary

- synthetic controlled data only；
- P1 compositional token feature routing；
- P3 family-structured same-expert relation；
- baseline 和 inhibition candidate 可作为条件比较；
- 每个解释都先通过 target learning 与 collapse guard。

### Outside boundary

- 不做 utility binding；
- 不做 Zipfian / frequency full run；
- 不做 real corpus；
- 不设计新 router；
- 不把 route consistency 直接等同于 specialization。

### Operational definitions

route consistency：同一 feature label $f$ 的样本是否集中进入同一 expert。

$$
\mathrm{Consistency}(F)=\frac{1}{|F|}\sum_f \max_e P(route=e\mid f)
$$

non-collapse：routing 至少使用多个 experts，并且不能由单一 expert 吸收绝大多数样本。

target-relevant routing：route 的主要信息轴随 target rule 变化；P1 中 C1 对齐 $S1$，C2 对齐 $S2$，C3 对齐 $(S1,S2)$。

same-expert relation：同一 expert 内 features 是否比 load-matched random grouping 更共享 family、representation geometry 或 target relation。

## 2. Physical Priors

### Prior 1: compositional tokens expose route-axis ambiguity

一个 token context 可以同时包含多个 candidate features。若 router 真跟随 target-relevant factor，改变 target rule 应改变 route axis；若 router 只跟随 surface axis 或 identity shortcut，route axis 不会随 target rule 改变。

### Prior 2: high consistency can be a false positive

collapse 或 deterministic lock-in 都能让 consistency 很高。因此 consistency 只有在 target learning 成功且 routing non-collapsed 后才可解释。

### Prior 3: $F>E$ forces many-to-one assignment

feature 数量大于 expert 数量时，多个 features 进入同一 expert 是必然现象。关键问题不是是否 many-to-one，而是 same-expert features 是否比随机同负载分组更有 family / geometry / target structure。

### Prior 4: frequency is downstream

Zipfian frequency 会改变 sample count、coverage 和 load pressure。若 feature definition 与 same-expert relation 未定，frequency 结果无法解释。因此 P2 parked。

## 3. Hypotheses And Rival Explanations

### H1: target-rule-dependent routing in P1

在 compositional token 中，dominant route axis 应随 target rule 改变：

- C1: $Y=Y(S1)$ 对齐 $S1$；
- C2: $Y=Y(S2)$ 对齐 $S2$；
- C3: $Y=Y(S1,S2)$ 对齐 $(S1,S2)$。

Rival explanation：route 固定跟随同一 surface axis 或 nuisance $B_i$。

### H2: family-structured grouping in P3

在 $E=4, G=4, K=4, F=16$ 的 family-structured feature dataset 中，same-expert features 应比 load-matched random grouping 更 family-structured。

Rival explanation：同一 expert 内 features 只是 arbitrary bucket 或 collapse 产物。

### H3: inhibition is only a condition

inhibition candidate 可以作为 `I*` 条件比较，但 05_02 的 claim 不依赖它必须成功。若 `I*` 提高 consistency 但增加 collapse，它是负面机制证据，不是 assumption relaxation 的成功。

## 4. Mathematical Modeling

For P1:

$$
x=[r_{\mathrm{start}}, S1, S2, B_i, Y, r_{\mathrm{end}}]
$$

Feature labels:

- $F_{S1}=S1$
- $F_{S2}=S2$
- $F_B=B_i$
- $F_{pair}=(S1,S2)$

Primary observable is normalized mutual information, 即归一化互信息（NMI）：

$$
\mathrm{NMI}(route;F)
$$

For P3:

$$
f=(g,k),\quad g\in\{1,\dots,G\},\quad k\in\{1,\dots,K\}
$$

Same-expert relation is compared against a load-matched random baseline:

$$
\Delta_{\mathrm{family}}=
\mathrm{Purity}_{actual}-\mathrm{Purity}_{random}
$$

## 5. Computational Realization

### P1

Construct compositional token contexts with $|S1|=4$, $|S2|=4$, $|B_i|=64$ or $256$, and optional nuisance variants if the data pipeline already supports them. Route logging uses one fixed position across all conditions: the $B_i$ position, which is also the pre-target position immediately before $Y$.

### P3

Construct family-structured features with $E=4$, $G=4$, $K=4$, $F=16$, uniform distribution. Each feature is $f=(g,k)$. Route logging uses one fixed position across all conditions: the last `F_gk` token, the pre-target position immediately before the target.

P3 has two subconditions:

- P3a: input-family-only grouping. The target is feature-level $Y_{gk}$.
- P3b: input-family + target-family grouping. The target is family-level $Y_g$.

### Conditions

At minimum compare:

- `B0`: matched no-inhibition baseline；
- `I*`: approved inhibition candidate, if available.

Inhibition is not the storyline; it is only one condition to test whether the relaxation remains non-collapsed and meaningful.

## 6. Minimal Falsifiable Tests

### P1 supported if

Under non-collapse and successful target learning, dominant route axis changes with target rule:

- C1 aligns with $S1$；
- C2 aligns with $S2$；
- C3 aligns with $(S1,S2)$。

### P1 weakened if

route always follows the same surface axis or $B_i$ despite $B_i$ being nuisance.

### P1 invalid if

target learning fails or routing collapses.

### P3 supported if

same-expert features are more family-structured than load-matched random grouping under non-collapse.

### P3 weakened if

same-expert grouping is no more structured than random.

### P3 invalid if

routing collapses, target learning fails, or too few features remain per active expert.

## 7. Current Evidence

A05_01_feature_key_routing_stability_center_only shows same-feature consistency can be high, but this is weak evidence because collapse can also produce high consistency.

A05_01_feature_axis_audit_existing_logs shows that after collapse filtering, the best existing non-collapsed route structure is closer to slot than to $B_i$.

A05_01_true_token_conditioned_inhibition_replay shows token-conditioned inhibition can sharpen consistency but may amplify collapse. This is why inhibition remains a candidate condition, not the main storyline.

A05_02_P1_compositional_token_feature_routing:

- Evidence files:
  - `Projects/from-attention-to-search/main/experiments/A05/A05_02_P1_compositional_token_feature_routing/summary.md`
  - `Projects/from-attention-to-search/main/experiments/A05/A05_02_P1_compositional_token_feature_routing/detailed.md`
- Observation:
  - all six condition groups reach target accuracy 1.0;
  - C1/C2 mostly collapse or use only two experts;
  - C3-B0 has mean active experts 3.0, max load 0.686, and route-axis NMI highest for $(S1,S2)$: 0.528.
- Interpretation:
  - C3 supports a narrow clue that compositional target structure can affect route axis;
  - C1/C2 prevent a broad P1 positive claim.

A05_02_P3_same_expert_feature_relation:

- Evidence files:
  - `Projects/from-attention-to-search/main/experiments/A05/A05_02_P3_same_expert_feature_relation/summary.md`
  - `Projects/from-attention-to-search/main/experiments/A05/A05_02_P3_same_expert_feature_relation/detailed.md`
- Observation:
  - all four condition groups reach target accuracy 1.0;
  - family purity deltas over load-matched random are small: P3a-B0 0.055, P3a-I1 0.047, P3b-B0 0.087, P3b-I1 0.090;
  - positive-looking P3b result is seed-unstable and concentrated in seed `20260522`.
- Interpretation:
  - input family alone does not reliably induce grouping;
  - shared target utility does not reliably strengthen grouping under ordinary top-1;
  - same-expert assignment remains closer to arbitrary bucket / early-locking bucket than structured family grouping.

## 8. Claim Boundary And Next Decision

Safest current claim：

05_02 now weakens the broad ordinary top-1 assumption-relaxation claim. Under these P1/P3 tests, target learning is easy, but meaningful non-collapsed routing is not robust. Same-feature route consistency should not be promoted to target-relevant routing or same-expert structured grouping without additional guards.

What cannot be claimed：

1. expert utility specialization；
2. real-data generalization；
3. Zipfian robustness；
4. successful new router design；
5. that inhibition is the explanation unless its condition passes the same guards.

Next smallest decision：

Decide whether the mainline should next test an explicit non-collapse / utility-binding intervention on P3b, or stop ordinary top-1 assumption relaxation here and treat it as negative evidence. Keep P2 parked until this decision is made.
