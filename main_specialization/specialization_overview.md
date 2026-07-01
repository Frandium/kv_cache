# Feature-Level Expert Specialization Overview

## Current Mainline

The current research question is:

```text
Can top-1 MoE gating form a stable, interpretable, and useful feature-level
expert partition after the correct route-relevant hidden states are selected
and early-training overwrite is controlled?
```

This line is not yet a deployable method. It is a mechanism audit for when an
MoE expert partition can be trusted as feature-level specialization.

## Current Judgment

The evidence after A05, A06, and A07 supports a narrower claim:

```text
A feature-level partition is reachable in controlled route-position geometry,
but ordinary top-1 routing does not naturally discover or preserve it.
```

The current bottleneck is not proving again that hidden states contain feature
geometry. The current bottleneck is:

```text
select the route-relevant hidden-state population
-> initialize a step-0 proxy partition
-> understand whether the partition margin is feature-residual or common-band
-> preserve it through step 5/10 real-text training
-> only then test functional expert utility
```

## Mechanism Picture

Use the minimal decomposition:

$$
h_f = c + r_f + n
$$

where $c$ is a common component, $r_f$ is the route-relevant feature residual,
and $n$ is non-route or nuisance structure. A linear top-1 gate scores:

$$
z_{f,e}=w_e^\top c+w_e^\top r_f+w_e^\top n
$$

This explains why uniform feature frequency does not guarantee uniform expert
use. The gate can follow $c$, all-position clustering can follow $n$, and early
training can overwrite an initially useful partition.

Functional specialization also needs a utility check. For common feature $C$
and rare feature $R$, the current A07 metric is:

$$
\mathcal{C}_{C,R}=\sum_e q_{C,e}q_{R,e}[-\cos(g_{C,e},g_{R,e})]_+
$$

This metric is useful only if reducing it also improves rare loss or expert
utility under matched guards.

## What A05 Says

A05 explains why ordinary routing can fail.

- Toy dot-product evidence supports common-logit causality.
- Real DCLM weakens the strong claim that step-0 routing is purely dominated by
  common logits.
- Real DCLM still shows that centering reduces max load and that the common
  channel can amplify quickly by step 10.

Current interpretation:

```text
The failure mechanism is wrong-object routing plus early-training feedback,
not simply lack of feature signal.
```

## What A06 Says

A06 explains where the usable feature geometry lives.

- Route-position feature centers are recoverable in controlled settings.
- All-position clustering is unreliable because it mixes route states with
  non-route, role, neutral, or nuisance states.
- In the controlled no-position bridge, center initialization preserves by a
  positive margin buffer, not by active router-center tracking.
- Margin shrink and forced-crossing audits show that positive margin is a real
  geometric safety region in the controlled bridge.
- Real DCLM proxy routing can exist at step 0, but ordinary training can erase
  the partition by step 5/10.

Current interpretation:

```text
The next step is to distinguish feature-residual margin from high-gain
common-band margin, then test early-training preservation under real DCLM.
```

## What A07 Says

A07 defines what would make a partition useful.

- Load balance and route purity are guards, not the main claim.
- The stronger claim requires reduced feature interference or improved expert
  utility.
- A07_04 measures common/rare gradient conflict in a neural proxy, but centered
  splitting is weak and does not yet beat the dense rare-loss boundary.

Current interpretation:

```text
Functional expert specialization remains unclaimed until preservation and
utility both pass.
```

## Current Claim Boundary

Can claim:

```text
Controlled route-position feature partition is reachable.
Ordinary top-1 routing does not naturally discover or preserve it.
All-position clustering is the wrong population for route-relevant features.
```

Cannot claim:

```text
Real semantic experts exist.
MoE already beats dense/shared training through specialization.
Balanced load or high route NMI proves useful experts.
KV-cache retrieval follows from current evidence.
```

## Next Decision

First run a mechanism audit:

```text
Does the positive route margin come from feature residual directions, or from a
high-gain common spectral band?
```

Then write and approve one real-text early-preservation protocol:

```text
Can a step-0 DCLM proxy partition survive the step-5/10 training window without
unacceptable language-model loss regression?
```

If yes, return to A07 utility and common/rare conflict. If no, decompose the
failure into gate update, hidden-state drift, expert feedback, and optimizer
geometry before proposing another method.

## Primary Pointers

```text
sync/S000_current_specialization/anchors/expert_specialization/00_feature_level_expert_specialization_mainline_anchor.md
Projects/from-attention-to-search/main/problem_anchors/05_failure_mechanism/05_04_03_real_text_common_logit_initialization_anchor.md
Projects/from-attention-to-search/main/problem_anchors/06_geometry_proxy_preservation/06_17_all_position_route_relevant_feature_discovery_anchor.md
Projects/from-attention-to-search/main/problem_anchors/07_features_overlap/07_04_common_rare_gradient_conflict_splitting_anchor.md
daily_research_reports/0624/meetings/meeting_brief.md
```
