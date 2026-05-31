---
parent_node: problem.moe_expert_specialization
status: daily_sandbox_promoted_evidence
source_draft: 00_handwrite_anchor.md
---

# 2x2 Top-1 Router Lock-In Anchor

## Report Summary（中文，汇报用）

目的：用最小 2 feature / 2 expert / top-1 NTP-style proxy 审查 router 为什么不能稳定形成 route-pattern specialization。

物理建模 / 假设：top-1 routing 对早期 router 几何非常敏感；如果不同 token 在 step 0/10 被分到同一个 expert，selected gate 会放大这个选择，并让未选 expert starvation。

最小实验：固定 $h_A=e_1,h_B=e_2$，2 experts，linear top-1 router，next-token CE；审查 early trajectory、success dynamics、router bias。

当前结论：失败 seed 在 step 0 已 collapse，不是训练中 drift；成功 seed 在 step 0 split 后会稳定保持并形成 counterfactual CE advantage；去掉 router bias 将 random-init split 从 3/10 提高到 6/10，但不能保证 specialization。

结论边界：只 claim 当前最小 top-1 NTP-style proxy 的 route-pattern dynamics；不 claim full LM、attention 场景或 utility-aligned specialization。

下一步：测试一个 anti-lock-in 机制，优先 structured router initialization，其次 exploration noise 或 load-balancing。

## Problem

Broader question:

```text
Why does ordinary top-1 MoE gating fail to produce stable, interpretable
feature-level expert specialization?
```

Minimal sandbox question:

```text
In a 2-feature / 2-expert top-1 NTP-style setting, does route collapse come
from early top-1 lock-in, and can better initial router geometry preserve
route-pattern separation?
```

## Hypothesis / Conjecture

```text
Top-1 route-pattern specialization is path dependent.
Initial same-expert assignment causes lock-in and expert starvation.
Initial split assignment is amplified into stable routing and expert advantage.
```

## Physical Priors

1. Top-1 routing converts a small logit difference into a discrete training
   path.
2. The unselected expert receives no token update, so early imbalance can
   become starvation.
3. Router bias can create a shared expert preference across tokens.

For $h_A=e_1,h_B=e_2$:

$$
z(h)=W_rh+b,\qquad p(e|h)=\operatorname{softmax}(z(h)).
$$

Define:

$$
\delta_A=z_{A,E1}-z_{A,E0},\qquad
\delta_B=z_{B,E1}-z_{B,E0}.
$$

If $\delta_A$ and $\delta_B$ have the same sign, A/B select the same expert. If
they have opposite signs, A/B split across experts.

## Minimal Decisive Test

Use only the NTP-style top-1 proxy:

```text
features: h_A=e1, h_B=e2
targets: A_next, B_next
router: linear top-1 selected-gate router
experts: two linear experts
objective: next-token CE
checkpoints: 0, 10, 50, 100, 200, 300
```

Primary evidence:

```text
NMI trajectory
selected gate trajectory
expert usage / update norm
counterfactual CE per forced expert
initial router delta signs
```

## Current Evidence

Early lock-in audit:

```text
Failed seeds are already collapsed at step 0.
Selected gate rises quickly by step 10 and reaches about 0.985 by step 300.
Unselected expert has zero usage for these tokens.
Counterfactual CE shows the selected expert learns both A/B targets.
```

Success dynamics audit:

```text
Successful seeds start with A/B split at step 0.
The split stays stable through step 300.
Selected logit margin rises from near 0 to about 2.
Selected expert becomes much better than unselected expert for its token.
```

No-router-bias audit:

```text
with bias:  random-router split 3/10 -> final split 3/10
no bias:    random-router split 6/10 -> final split 6/10
router prior: known-good split 10/10 -> final split 10/10
```

Main evidence files:

```text
03_summary.md
04_detailed.md
figures/ntp_style_lockin_audit.png
figures/ntp_style_success_dynamics_audit.png
figures/ntp_style_no_router_bias/summary_by_seed.png
tables/ntp_style_lockin_audit.csv
tables/ntp_style_success_dynamics_audit.csv
tables/ntp_style_router_bias_comparison.csv
```

## Claim Boundary And Next Decision

Safest current claim:

```text
In this minimal top-1 NTP-style proxy, route collapse is mainly explained by
early top-1 lock-in and expert starvation. Initial split can be amplified into
stable route-pattern separation. Router bias worsens collapse by adding shared
expert preference, but removing bias only partially improves split rate.
```

What cannot be claimed:

```text
This proves utility-aligned specialization.
This directly transfers to full LM or attention settings.
This fully explains ABC synthetic failure.
```

Next smallest decision:

```text
Test one anti-lock-in mechanism:
structured router initialization / exploration noise / load-balancing.
```
