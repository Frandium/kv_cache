# Report Card: Round-2 Dot-Product Common-Logit Audit

## Source Files

Anchor: `05_04_02_dotproduct_common_logit_causality_anchor.md`

Summary: `summary.md`

Detailed: `detailed.md`

Figures / Tables: `figures/`, `tables/`

## 0. Executive Summary

Goal:

Test whether common-logit dominance explains why random-init sparse top-1 MoE routing collapses in the uniform no-position slot task.

Minimal mechanism audit:

Use a dot-product router so logit components add linearly, then audit timing, prediction, slot-init basin threshold, common-logit cancellation, and common-source ablations.

Clear hypothesis:

If the common component is causal, it should be early and predictive, and subtracting it from router logits should improve slot-level routing specialization while preserving target accuracy.

Key finding:

Common dominance is visible at step 0, grows before step 10, predicts the final dominant expert, and common-logit cancellation raises final slot NMI from 0.080 to 0.896 or 0.963 while keeping accuracy at 1.000.

Current conclusion:

In this toy dot-product sparse top-1 setting, collapse is best interpreted as a common-logit-driven basin problem rather than missing slot information alone.

Claim boundary:

This does not identify the full common source, prove expert computation is slot-specialized, or transfer the result to real language models.

Next step:

Design a label-free anti-common or anti-lock-in router that keeps sparse top-1 routing while making slot specialization reachable from random initialization.

## 1. Research Process Update

| item | content |
|---|---|
| Previous mainline | Round 1 showed no-position random-init routing still collapses even though the routed hidden state contains slot information. |
| New probe | Round 2 tests whether common-logit dominance is early, predictive, and causally intervention-relevant under a dot-product router. |
| New evidence | P0 decomposition is numerically exact; P1 shows early common dominance and prediction; P2 shows a slot-init basin threshold; P3 shows common cancellation improves route-slot NMI with accuracy preserved; P4 points to `B_CONST` / B-position identity as likely sources. |
| Knowledge update | The current best explanation is common-logit-driven routing basin lock-in in this toy setting. |
| Next decision | Build the smallest label-free router intervention that suppresses common lock-in without soft routing or top-k mixtures. |

## 2. Terms Used Here

- `Common logit`: the router-score contribution shared across slots, measured through the common hidden component under dot-product decomposition.
- `Slot NMI`: normalized mutual information between slot identity and selected expert; higher means routing is more slot-specialized.
- `Top-1 collapse`: most routed tokens select the same expert, usually shown by high `max_load` and low route-slot NMI.
- `Common cancellation`: subtracting the estimated common-logit term from router scores at the audited routed position.

## 3. Key Figures

### Figure 1: Common vs Slot Margin Timing

![Common vs slot margin timing](figures/round2_dotproduct_common_logit_4gpu_20260609/r2_p1_common_slot_margin_trajectory.png)

What to see:

The common margin is already larger than the slot margin at step 0 and grows sharply before step 10.

Supports:

Common dominance is not only a final-collapse aftereffect; it is present early enough to plausibly shape lock-in.

Cannot prove:

This timing plot alone does not prove causality; that requires the cancellation intervention.

### Figure 2: Common Cancellation Improves Slot Specialization

![Common cancellation](figures/round2_dotproduct_common_logit_4gpu_20260609/r2_p3_baseline_vs_cancel.png)

What to see:

Baseline random init has final slot NMI 0.080; cancellation from steps 0-10 reaches 0.896, and cancellation through final reaches 0.963, with accuracy 1.000.

Supports:

The common component is intervention-relevant for slot-level routing specialization, not merely load balance.

Cannot prove:

It does not show that expert computations are causally slot-specialized or that this intervention is deployable in real models.

### Figure 3: Common-Source Audit

![Common source audit](figures/round2_dotproduct_common_logit_4gpu_20260609/r2_p4_source_comparison.png)

What to see:

Changing the audit/source condition reduces common margin most strongly when routing at slot or varying `B_CONST`, while filler variation leaves common dominance comparatively high.

Supports:

`B_CONST` / routed B-position identity is the leading source candidate.

Cannot prove:

This is not full source identification; token identity, residual mean, and optimizer/top-1 feedback remain distinguishable rival sources.

## 4. Current Claim

Random-init dot-product sparse top-1 routing collapses in this minimal no-position slot task because common-logit advantage is early, predictive of the final dominant expert, and intervention-relevant. Slot initialization succeeds because it moves routing across a basin threshold where slot-aligned margins can dominate.

## 5. Claim Boundary

Can claim:

- Common-logit dominance is early and predictive in this Round-2 toy setup.
- Common-logit cancellation improves route-slot NMI while preserving accuracy.
- Slot-init success shows a basin threshold among the tested alpha values.
- Fixed `B_CONST` / routed B-position identity is the leading common-source candidate.

Cannot claim:

- The common source is fully identified.
- Load balance alone is feature specialization.
- Expert computation is causally slot-specialized.
- The result transfers to real language models, longer contexts, cosine routers, or broader MoE settings.

## 6. Next Step

Design and test a minimal label-free anti-common or anti-lock-in router under the same sparse top-1 constraint. The next experiment should change the router mechanism, not the dataset, so the causal target remains the common-logit basin.
