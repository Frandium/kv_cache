# Protocol: A06_18 Revision Audit

Approval status: approved by user request on 2026-06-25 for a local two-GPU
full controlled revision run.

## 0. Approval Snapshot

**Purpose:** Test whether a revised label-free representation selector can
beat raw all-position clustering on held-out route-position feature recovery.

**Primary anchor:**
`../../../problem_anchors/06_geometry_proxy_preservation/06_18_label_free_route_relevant_state_selection_anchor.md`

**Decision question:** Can a label-free representation of all-position hidden
states produce clusters whose hidden-space centers recover held-out
route-position features better than raw all-position k-means?

**Primary metric:** held-out route-position `feature_NMI`.

**Decision rule:** pass only if a revised selector beats raw all-position and
approaches route-only / slot-offset-3 positive controls. Fail if it only
improves reconstruction, load, or visualization while matching all-position
feature merge.

**Claim boundary:** controlled selector validity only. This run does not claim
real-DCLM proxy semantics, training preservation, utility, or SAE necessity.
Do not create A06_19 for this run.

## 1. Tested Hypothesis

All-position clustering fails because raw hidden space mixes route-relevant and
non-route populations. A useful label-free representation should separate the
route-relevant residual well enough that K=4 latent clusters become good
hidden-space gating centers.

## 2. Compared Conditions

| Condition | Role |
|---|---|
| Raw all-position k-means | failure baseline |
| Route-only k-means | positive control |
| Slot-offset-3 k-means | positive control |
| A06_18 split-stability selector | known failed selector family |
| PCA latent clustering `q=4,8,16,32` | linear low-rank representation test |
| Bottleneck AE latent clustering `q=4,8,16,32` | learned compressed representation test |
| Overcomplete SAE-code clustering `d_sae=4*d_model, 8*d_model`, L1/top-k | sparse code representation test |

## 3. Representation Rule

For PCA, bottleneck AE, and SAE variants:

1. fit the representation on calibration hidden states without feature labels;
2. cluster all-position states in representation space with K=4;
3. convert each latent cluster to a hidden-space gating center by averaging the
   original hidden states assigned to that cluster;
4. evaluate those centers on held-out route-position states.

Feature labels are used only for final evaluation and failure interpretation.

## 4. Data And Architecture

**Data:** A06_17 controlled `C4_all_position_scope`.

**Architecture:** one-layer Transformer plus top-1 MoE from A06_16, no
positional embedding, `d_model=128`, `n_heads=4`, `ffn_dim=256`, four features,
four experts, sequence length 32, route position slot offset 3.

**Seeds:** `20260623` to `20260630`.

**Execution:** local two-GPU full run, two shards.

## 5. Guards

Report max load, active experts, nuisance NMI, slot-start NMI, selected/cluster
role composition, position composition, and AE/SAE reconstruction error.

## 6. Success / Failure / Insufficient

**Success:** at least one revised selector beats all-position on mean
`feature_NMI`, is close to route-only / slot-offset-3, and is not explained by
load or nuisance guards.

**Failure:** representation variants match all-position, improve only
reconstruction/load, or select clusters dominated by non-route/target/neutral
roles without improving held-out route readout.

**Insufficient:** route-only or slot-offset-3 controls fail, held-out route
states are missing, or implementation cannot produce reconstruction/guard
tables.

## 7. Next Decision

If a representation variant passes, later test the same initialization in
slot early training and then write a very short A06_19. If no variant passes,
analyze whether the bottleneck is representation fitting, latent clustering, or
the absence of a route-readout constraint.
