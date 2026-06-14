# A06_02_02 Freeze Split and Common-Stability Summary

## Purpose

This experiment audits the causal source of the step-10 late-layer common-logit spike observed in A06_02_01.

The decision question is:

```text
In small random-initialized real-text sparse top-1 MoE, what creates the step-10 late-layer common-logit spike?
```

The candidate sources are hidden-state drift under a fixed gate, gate-hidden interaction, expert parameter update, non-expert hidden-producing paths, layer-local hidden producers, and instability of the common direction itself.

## Setup

```text
model: random-initialized Qwen-style decoder-only MoE
layers: 6
hidden_size: 512
attention_heads: 8
kv_heads: 4
experts: 8
expert_hidden_dim: 2048
router: linear, bias-free
top_k: 1
shared_expert: disabled
load_balance_loss: lambda_lb=0.0
dataset: DCLM packed binary stream
input_length: 256
train_sequences: 32768
audit_sequences: 8192
seeds: 0, 1, 2
max_train_steps: 300
checkpoints: 0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 300
```

Full run:

```text
job_id: pt-tfd7w34x
state: SUCCEEDED
run_dir: runs/freeze_split_common_stability/a06_02_02_freeze_split_common_stability_4gpu_20260614_full01
```

## Conclusion

The main result is that early route concentration and common-logit amplification are related but not identical.

For layer 5, normal training increases `common_margin` by `1.2760` from step 0 to step 10, and increases `raw_max_load` by `0.7334`. Freezing the final-layer hidden producer nearly removes both effects (`common_margin_delta=0.0278`, `raw_max_load_delta=0.0660`). This localizes the fast layer-5 spike to the hidden-producing path before the layer-5 router input.

However, freezing the gate, experts, or both does not remove raw top-1 load concentration. For example, `freeze_gate_all` still has `raw_max_load_delta=0.7314`, nearly the same as normal. Therefore a fixed random gate plus hidden-state drift can still concentrate routes. The full normal common-margin spike is larger than this fixed-gate effect because hidden drift and gate-hidden interaction combine.

The common direction is not a batch artifact. In layers 3--5 at step 10, split-level common vectors have `pairwise_cos_mean >= 0.9999`, primary-secondary cosine is `1.0000`, and common-winner agreement is `1.0000`. But the common direction is checkpoint-specific, not static across training: layer-5 cosine to step 0 is only `0.2574` at step 10 and `0.1417` at step 300.

## Key Evidence

Layer-5 step0 to step10 changes, mean over 3 seeds:

```text
variant                     common_margin_delta   raw_max_load_delta
normal                      1.2760                0.7334
freeze_gate_all             0.5354                0.7314
freeze_experts_all          0.1919                0.7183
freeze_gate_and_experts     0.2872                0.7114
freeze_prefix_before_layer3 0.5264                0.5674
freeze_prefix_before_layer4 0.1914                0.3775
freeze_prefix_before_layer5 0.0278                0.0660
```

Cross-checkpoint replay computes $Z_{a,b}=H_bW_a^\top$. For layer 5:

```text
normal:
  A_actual       1.2760
  A_gate        -0.0137
  A_hidden       0.4858
  A_interaction  0.8038

freeze_gate_all:
  A_actual       0.5354
  A_gate         0.0000
  A_hidden       0.5354
  A_interaction  0.0000
```

This means the direct gate-only term does not explain the spike. The normal spike is mainly hidden drift plus gate-hidden interaction.

Common-stability audit, layers 3--5:

```text
step10 pairwise_cos_mean: 0.9999--1.0000
step10 primary_secondary_cos: 1.0000
step10 common-winner agreement: 1.0000
step10 layer5 cos_to_step0: 0.2574
step300 layer5 cos_to_step0: 0.1417
```

This supports a checkpoint-specific, batch-stable common direction. It weakens the stronger idea that the same common vector stays fixed throughout early training.

## Hypothesis Answers

```text
H1 hidden-state drift under fixed random gate is sufficient:
  Supported for raw route concentration and as a partial common-margin source.
  Not sufficient for the full normal common-margin spike.

H2 gate-hidden interaction is required for full normal common-logit amplification:
  Supported.
  Layer-5 normal interaction is 0.8038, while freeze_gate_all interaction is 0.0000.

H3 expert parameter updates are necessary:
  Supported for full common-margin amplification.
  Not supported for raw top-1 load concentration.

H4 non-expert hidden-producing paths alone can create concentration:
  Supported for raw load concentration.
  Partial for common-margin amplification.

H5 the layer-5 hidden producer is the local source of the fast layer-5 spike:
  Supported.
  freeze_prefix_before_layer5 suppresses layer-5 common-margin delta by 97.82% and raw-load delta by 91.00%.

H6 the common direction is stable enough for a mean-vector interpretation:
  Supported within the same checkpoint and across disjoint audit batches.
  Weakened as a static-across-training direction because step10 and step300 common vectors rotate far from step0.
```

## Validity Audit

```text
trajectory_rows: 1386 / 1386 expected
checkpoint_rows: 231 / 231 expected
replay_rows: 15246 / 15246 expected
fraction_rows: 126 / 126 expected
max_router_logit_reconstruction_error: 0.0
```

Freeze semantics were checked by gradient norms:

```text
freeze_gate_all: router_grad_norm = 0.0000
freeze_experts_all: expert_grad_norm = 0.0000
freeze_gate_and_experts: router_grad_norm = 0.0000, expert_grad_norm = 0.0000
```

The posthoc common-stability audit used a key-compatible fallback source because `/data/250010109/MoE_Router` was no longer visible on later ACP workers. The fallback disabled shared experts and loaded checkpoints with `strict=True`; provenance is recorded in:

```text
results/common_stability_audit_provenance.json
```

## Claim Boundary

This result applies to a small random-initialized 6-layer, 8-expert, top-1 Qwen-style MoE trained on DCLM packed text for 300 steps with shared expert disabled and load-balance loss disabled.

It supports a causal split for early training in this setting. It does not prove behavior in pretrained MoEs, large-scale MoEs, shared-expert MoEs, top-2 routing, or mitigation methods. It also does not test Phase C expert-output forward-feedback interventions; those were explicitly held out.

## Next Decision

The next decision should target the remaining rival explanation: whether the raw top-1 load concentration that survives gate/expert freezing is caused by residual anisotropy, position structure, or forward expert-output feedback. The current result already closes the narrower question that the full layer-5 common-margin spike requires the layer-5 hidden-producing path and gate-hidden interaction.
