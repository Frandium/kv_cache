# A06_02_02 Freeze Split and Common-Stability Protocol

## 0. Protocol Summary

This protocol follows A06_02_01. The previous run showed that the early route-concentration spike in real DCLM top-1 MoE is not explained by router-weight update alone. The new goal is to separate:

1. hidden-state drift under fixed gate;
2. gate-hidden co-adaptation;
3. expert-parameter update;
4. expert-output feedback;
5. layerwise source of hidden-state drift;
6. batch/checkpoint stability of the common direction.

## 1. Decision Question

In small random-initialized real-text sparse top-1 MoE, what is the causal source of the step-10 late-layer common-logit spike?

Candidate sources:

```text
A. hidden-state drift is sufficient under fixed gate;
B. gate update is necessary for W-H interaction;
C. expert parameter updates create the hidden drift;
D. expert-output feedback into downstream layers creates the hidden drift;
E. earlier hidden-producing layers, rather than the audited layer itself, create the spike;
F. the observed common direction is not stable enough, so the current common-channel interpretation is insufficient.
```

## 2. Fixed Experimental Contract

Keep the same contract as A06_02_01 unless explicitly changed.

```text
model_family: Qwen-style decoder-only MoE
initialization: random
num_hidden_layers: 6
hidden_size: 512
num_attention_heads: 8
num_key_value_heads: 4
num_experts: 8
expert_hidden_dim: 2048
router_type: linear
router_bias: false
top_k: 1
use_shared_expert: false
lambda_lb: 0.0
norm_topk_prob: false
gating_reference: exact active router linear input
dataset: DCLM packed binary stream
input_length: 256
train_sequences: 32768
audit_sequences: 8192
seeds: 0, 1, 2
max_train_steps: 300
learning_rate: 3e-4
checkpoints: 0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 300
```

The audit batch must be identical across variants, seeds, and checkpoints.

## 3. Required Saved Objects

For every variant, seed, layer, checkpoint, and audit batch:

```text
H_t^l: exact pre-gate router input at layer l
W_t^l: gate weight matrix at layer l
Z_t^l = H_t^l (W_t^l)^T
route_ids_t^l: top-1 selected expert IDs
c_t^l = mean_i H_t^l[i]
R_t^l = H_t^l - c_t^l
loss_t
router_grad_norm_t^l
expert_grad_norm_t^l
other_grad_norm_t
```

Sanity check:

```text
max_abs(H_t^l (W_t^l)^T - logged_router_logits_t^l) <= 1e-6
```

If this fails, the run is invalid.

## 4. Phase A: Common Direction Stability Diagnostic

### Question

Is the common direction stable across batches and checkpoints, or is the step-10 spike produced by a rotating/new common direction?

### Implementation

For the fixed audit set, split 8192 sequences into 8 equal audit shards.

For each seed, layer, checkpoint, and shard:

```text
c_{s,l,t,b} = mean_i H_{s,l,t,b}[i]
```

Also compute a low-rank common subspace from each shard using top-k PCA directions of H, with k in {1, 4, 8}.

### Metrics

```text
cos_to_global_c = cos(c_{s,l,t,b}, c_{s,l,t,global})
mean_pairwise_cos_c across shards
min_pairwise_cos_c across shards
cos_checkpoint = cos(c_{s,l,t,global}, c_{s,l,0,global})
cos_to_step10 = cos(c_{s,l,t,global}, c_{s,l,10,global})
subspace_overlap_topk across shards
common_winner = argmax_e (W_{s,l,t,e}^T c_{s,l,t,b})
common_winner_persistence = mode_fraction(common_winner across shards)
```

### Outcomes

Supported common-stability path:

```text
mean_pairwise_cos_c is high, subspace overlap is high, and common_winner_persistence is high in layers 3--5 around steps 5--20.
```

Weakened common-stability path:

```text
common directions rotate strongly across shards/checkpoints, but routes still concentrate.
```

Interpretation if weakened:

```text
The current mean-common-vector model is incomplete. Inspect higher-rank common subspace, position buckets, or residual anisotropy.
```

## 5. Phase B: Core Freeze Variants

### Variant B0: normal

```text
all parameters train
```

Purpose: reproduce the A06_02_01 baseline spike.

### Variant B1: freeze_gate_all

```text
all MoE gate weights fixed at initialization W_0
all non-gate parameters train normally
```

Question:

```text
Can hidden-state drift alone produce step-10 route concentration under a fixed random gate?
```

Interpretation:

```text
If spike persists: hidden drift is sufficient under fixed gate.
If spike is suppressed: gate update is necessary for the fast W-H interaction.
```

### Variant B2: freeze_experts_all

```text
all expert MLP parameters fixed at initialization
all gates, attention, layernorms, embeddings, and LM head train normally
```

Question:

```text
Are expert parameter updates necessary for the step-10 spike?
```

Interpretation:

```text
If spike is suppressed: expert parameter update contributes to hidden drift.
If spike persists: upstream non-expert hidden drift plus gate interaction is sufficient.
```

### Variant B3: freeze_gate_and_experts

```text
all gate weights fixed at W_0
all expert MLP parameters fixed at initialization
attention, embeddings, layernorms, and LM head train normally
```

Question:

```text
Can non-expert hidden-producing paths alone create the spike?
```

Interpretation:

```text
If spike persists: non-expert hidden drift is sufficient.
If spike is suppressed: gate/expert adaptation is required.
```

## 6. Phase C: Expert-Output Feedback Split

Expert-output feedback has two different meanings: expert parameter update and expert output affecting downstream hidden states. Phase B2 tests expert parameter update. Phase C tests output-feedback paths.

### Variant C1: freeze_experts_all

Reuse B2 as the expert-parameter-update ablation.

### Variant C2: stopgrad_moe_output

```text
forward pass uses the normal MoE output
but the MoE output is detached before residual addition for downstream gradient flow
```

Question:

```text
Does downstream gradient through selected expert outputs contribute to the spike?
```

Caveat:

```text
This blocks gradient feedback, not forward hidden-state feedback.
```

### Variant C3: zero_moe_output_forward_diagnostic

```text
replace each MoE output with zero before residual addition during training and audit
keep router decisions logged
```

Question:

```text
Is forward expert output into downstream hidden states necessary for downstream-layer route concentration?
```

Caveat:

```text
This is a diagnostic-only destructive intervention. LM loss may become worse. Do not interpret as a deployable method.
```

### Variant C4: previous_layer_expert_output_block

For target layers 3, 4, and 5:

```text
block or zero only the expert output of previous MoE layers before the target layer
keep attention and non-expert paths active
```

Question:

```text
Does the late-layer spike come from selected expert outputs in earlier layers?
```

Preferred minimal targets:

```text
target layer 3: block expert outputs in layers 0--2
target layer 4: block expert outputs in layers 0--3
target layer 5: block expert outputs in layers 0--4
```

## 7. Phase D: Layerwise Hidden-Producer Localization

### Question

Which upstream layer range produces the hidden-state drift that interacts with the gate in layers 3--5?

### Variants

Use a prefix-freeze ladder:

```text
D0 normal
D1 freeze_prefix_before_layer3: freeze embeddings + layers 0--2
D2 freeze_prefix_before_layer4: freeze embeddings + layers 0--3
D3 freeze_prefix_before_layer5: freeze embeddings + layers 0--4
```

Each variant keeps the target and later layers trainable.

D3 is the same logical condition as the previous final-layer hidden-producer freeze and should reproduce the previous suppression of the layer-5 step-10 spike.

### Metrics

For target layers 3, 4, and 5:

```text
common_margin_t
residual_margin_t
dominance_ratio_t
raw_max_load_t
centered_max_load_t
delta_max_load_t
cos(c_t, c_0)
common_winner_persistence_t
```

### Outcomes

```text
If freezing prefix before layer k suppresses the spike at layer k and later layers, then upstream hidden drift before k is necessary.
If the spike persists despite prefix freeze, then the target layer's own gate/expert path or later co-adaptation is sufficient.
```

## 8. Phase E: Replay Within Each Variant

For each variant, compute cross-checkpoint replay at least for steps 0 and 10:

```text
Z_{a,b}^l = H_b^l (W_a^l)^T
```

Required pairs:

```text
W0_H0
W10_H10
W10_H0
W0_H10
W10_centeredH10
```

Primary scalar:

```text
M = common_margin
```

Amplification decomposition:

```text
A_actual = M(W10,H10) - M(W0,H0)
A_gate = M(W10,H0) - M(W0,H0)
A_hidden = M(W0,H10) - M(W0,H0)
A_interaction = A_actual - A_gate - A_hidden
```

Run the same decomposition for:

```text
raw_max_load
centered_max_load
delta_max_load
log(dominance_ratio)
```

## 9. Primary Metrics

Primary decision metrics:

```text
common_margin
residual_margin
dominance_ratio
raw_max_load
centered_max_load
delta_max_load
A_gate, A_hidden, A_interaction
suppression_fraction of step-10 spike
```

Common-stability metrics:

```text
cos(c_t, c_0)
cos(c_t, c_10)
mean_pairwise_cos_c across audit shards
top-k common-subspace overlap
common_winner_persistence
```

Execution sanity metrics:

```text
gate reconstruction max error
router_grad_norm
expert_grad_norm
other_grad_norm
LM loss
```

## 10. Suppression Fraction

For each variant V and metric M:

```text
Spike_normal = M_normal(step10) - M_normal(step0)
Spike_V = M_V(step10) - M_V(step0)
Suppression_V = 1 - Spike_V / Spike_normal
```

Use `common_margin` as the primary M.

Interpretation:

```text
Suppression_V > 0.7: strong suppression
0.3 <= Suppression_V <= 0.7: partial suppression
Suppression_V < 0.3: weak suppression
```

These are working thresholds, not final claims.

## 11. Falsifiable Outcomes

### Outcome 1: fixed-gate hidden drift sufficient

Supported if:

```text
freeze_gate_all still shows a large step-10 spike, especially in layers 3--5.
```

Meaning:

```text
Hidden-state drift can produce concentration even when gate weights do not update.
```

### Outcome 2: gate update necessary for W-H interaction

Supported if:

```text
freeze_gate_all suppresses the spike, while normal has large A_interaction.
```

Meaning:

```text
Hidden drift alone is not enough; changed W and changed H must align.
```

### Outcome 3: expert parameter updates necessary

Supported if:

```text
freeze_experts_all strongly suppresses the spike.
```

Meaning:

```text
Selected expert learning contributes to the hidden-state drift or feedback loop.
```

### Outcome 4: non-expert hidden producer sufficient

Supported if:

```text
freeze_gate_and_experts still shows strong spike.
```

Meaning:

```text
Attention, embeddings, layernorms, or LM-head-driven representation drift can produce the common-channel amplification without gate/expert adaptation.
```

### Outcome 5: expert-output forward feedback necessary

Supported if:

```text
zero_moe_output_forward_diagnostic or previous_layer_expert_output_block suppresses downstream-layer spikes.
```

Meaning:

```text
Selected expert outputs into the residual stream are part of the causal feedback loop.
```

### Outcome 6: common direction unstable

Supported if:

```text
cross-batch common cosines and subspace overlaps are low, but route concentration remains high.
```

Meaning:

```text
The scalar mean-common-vector model is insufficient; use higher-rank common subspace or residual-anisotropy analysis.
```

## 12. Insufficient Evidence Conditions

The experiment is insufficient if any condition holds:

```text
1. H_t is not the exact active gate input.
2. audit batch differs across variants.
3. only max_load is reported without common-logit decomposition.
4. shared expert or load-balance loss is enabled.
5. reconstruction error is non-negligible.
6. only one seed or one layer is reported.
7. destructive variants are interpreted without LM-loss sanity.
8. freeze variants change optimizer schedule or checkpoint spacing relative to normal.
```

## 13. Minimal Execution Order

Execute in this order:

```text
Step 0: add Phase A common-stability diagnostics to existing A06_02_01 artifacts if possible.
Step 1: run B0 normal, B1 freeze_gate_all, B2 freeze_experts_all, B3 freeze_gate_and_experts.
Step 2: run D1/D2/D3 prefix-freeze ladder.
Step 3: run C2/C3/C4 only if B2/D variants suggest expert-output feedback is plausible.
Step 4: produce one summary table that maps each gap to a supported/weakened/insufficient status.
```

If compute is limited, the minimum decisive set is:

```text
normal
freeze_gate_all
freeze_experts_all
freeze_gate_and_experts
freeze_prefix_before_layer3
freeze_prefix_before_layer4
freeze_prefix_before_layer5
common-stability diagnostic on existing artifacts
```

## 14. Deliverables

```text
summary.md
  - one-page decision report
  - table: hypothesis -> result -> status -> next decision

detailed.md
  - full variant setup
  - trajectory tables
  - replay decomposition tables
  - common-stability tables
  - layerwise suppression tables

results/
  - trajectory_metrics.csv
  - replay_metrics.csv
  - stability_metrics.csv
  - freeze_suppression_summary.csv
  - gradient_metrics.csv

figures/
  - step trajectory curves
  - layerwise spike heatmaps
  - W_a H_b replay heatmaps
  - common-stability cosine heatmaps
  - freeze suppression bar plots

config/
  - run_config.json
  - source_manifest.json
```

## 15. Minimal Final Report Table

| Gap | Variant / diagnostic | Primary metric | Supports | Weakens |
|---|---|---|---|---|
| fixed gate under hidden drift | freeze_gate_all | step-10 common_margin suppression | hidden drift sufficient if spike persists | gate update necessary if suppressed |
| expert parameter update | freeze_experts_all | suppression fraction | expert learning matters if suppressed | non-expert path sufficient if persists |
| non-expert hidden drift | freeze_gate_and_experts | spike persistence | non-expert producer sufficient | gate/expert adaptation required |
| layer source | prefix-freeze ladder | layerwise suppression | upstream source localized | source not in frozen prefix |
| common stability | shard/checkpoint diagnostic | cos/subspace/winner persistence | coherent common-channel interpretation | mean-common model insufficient |
| expert-output feedback | output block / stopgrad | downstream spike suppression | expert output feedback matters | feedback not necessary |
