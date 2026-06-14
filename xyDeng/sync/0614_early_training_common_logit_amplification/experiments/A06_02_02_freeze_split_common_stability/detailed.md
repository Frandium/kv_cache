# A06_02_02 Freeze Split and Common-Stability Detailed Record

## 0. Quick Recap

Question:
What creates the step-10 late-layer common-logit spike in the real DCLM random-initialized top-1 MoE run?

Answer:
The full layer-5 common-margin spike requires the hidden-producing path before the layer-5 router input and is mainly hidden drift plus gate-hidden interaction. A fixed random gate can still produce strong raw top-1 route concentration, so raw load concentration and common-logit amplification must be separated.

Most important numbers:

```text
normal layer5 step0->10:
  common_margin_delta = 1.2760
  raw_max_load_delta  = 0.7334

freeze_prefix_before_layer5 layer5 step0->10:
  common_margin_delta = 0.0278
  raw_max_load_delta  = 0.0660

normal layer5 replay:
  A_gate        = -0.0137
  A_hidden      =  0.4858
  A_interaction =  0.8038
```

Common-stability result:
At a fixed checkpoint, the common direction is extremely stable across disjoint audit batches. It is not a static direction across training checkpoints.

## 1. Source Files

Anchor:

```text
Projects/from-attention-to-search/main/problem_anchors/06_02_early_training_common_logit_amplification_anchor.md
```

Protocol:

```text
Projects/from-attention-to-search/main/experiments/A06_02_02_freeze_split_common_stability/protocol_for_approval.md
```

Scripts:

```text
Projects/from-attention-to-search/main/experiments/A06_02_02_freeze_split_common_stability/scripts/run_freeze_split_common_stability.py
Projects/from-attention-to-search/main/experiments/A06_02_02_freeze_split_common_stability/scripts/audit_common_batch_stability_pca.py
Projects/from-attention-to-search/main/experiments/A06_02_02_freeze_split_common_stability/scripts/summarize_freeze_split_results.py
Projects/from-attention-to-search/main/experiments/A06_02_02_freeze_split_common_stability/scripts/submit_freeze_split_common_stability_4gpu_acp.sh
Projects/from-attention-to-search/main/experiments/A06_02_02_freeze_split_common_stability/scripts/submit_common_pca_acp.sh
```

Run directory:

```text
Projects/from-attention-to-search/main/experiments/A06_02_02_freeze_split_common_stability/runs/freeze_split_common_stability/a06_02_02_freeze_split_common_stability_4gpu_20260614_full01
```

## 2. Experimental Setup

```text
model_family: Qwen-style decoder-only MoE
initialization: random
num_hidden_layers: 6
hidden_size: 512
num_attention_heads: 8
num_key_value_heads: 4
head_dim: 64
num_experts: 8
expert_hidden_dim: 2048
vocab_size: 151936
initializer_range: 0.02
router_type: linear
router_bias: false
top_k: 1
use_shared_expert: false
lambda_lb: 0.0
norm_topk_prob: false
gating_reference: exact active router linear input
```

Data:

```text
dataset: DCLM packed binary stream
span_length: 257 tokens
input_length: 256 tokens
target: shifted next-token target, 256 tokens
padding: none
audit_sequences: 8192
audit_tokens_per_layer: 2097152
train_sequences: 32768
```

Training:

```text
max_train_steps: 300
learning_rate: 3e-4
train_batch_size_per_rank: 2
audit_batch_size: 8
seeds: 0, 1, 2
checkpoints: 0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 300
```

Variants:

```text
normal:
  all parameters train

freeze_gate_all:
  all MoE gate weights fixed at initialization W0
  all non-gate parameters train

freeze_experts_all:
  all expert MLP parameters fixed
  gates, attention, layernorms, embeddings, final norm, and LM head train

freeze_gate_and_experts:
  all gate weights and all expert MLP parameters fixed
  attention, layernorms, embeddings, final norm, and LM head train

freeze_prefix_before_layer3:
  embeddings and layers 0-2 frozen
  layer 3 and later train

freeze_prefix_before_layer4:
  embeddings and layers 0-3 frozen
  layer 4 and later train

freeze_prefix_before_layer5:
  embeddings and layers 0-4 frozen
  layer-5 input_layernorm, self_attn, and post_attention_layernorm frozen
  layer-5 MoE/gate path remains trainable
```

Phase C expert-output feedback interventions were intentionally not run.

## 3. Execution

Full ACP run:

```bash
A06_02_02_ALLOW_REAL_SUBMIT=1 \
RUN_NAME=a06_02_02_freeze_split_common_stability_4gpu_20260614_full01 \
Projects/from-attention-to-search/main/experiments/A06_02_02_freeze_split_common_stability/scripts/submit_freeze_split_common_stability_4gpu_acp.sh
```

Full run record:

```text
job_id: pt-tfd7w34x
state: SUCCEEDED
start_time: 2026-06-14T10:16:40Z
completed_at: 2026-06-14T12:09:38Z
runtime_log: logs/acp/a06_02_02_freeze_split_common_stability_4gpu_20260614_full01_runtime_20260614_101641.log
```

Posthoc common-stability/PCA audit:

```bash
A06_02_02_ALLOW_REAL_SUBMIT=1 \
Projects/from-attention-to-search/main/experiments/A06_02_02_freeze_split_common_stability/scripts/submit_common_pca_acp.sh
```

Common-stability run record:

```text
job_id: pt-dzy0m0jz
state: SUCCEEDED
start_time: 2026-06-14T12:21:57Z
completed_at: 2026-06-14T12:26:07Z
runtime_log: logs/acp/a06_02_02_freeze_split_common_stability_4gpu_20260614_full01_common_pca_4gpu_env_runtime_20260614_122158.log
```

Execution caveat:
The later ACP workers did not expose `/data/250010109/MoE_Router`, which was the source path used by the full run. The PCA audit therefore used a key-compatible fallback source under `/data/250010109/past_records/qwen_exp`, replaced candidate-code `shared_expert` modules with zero-output modules, and loaded checkpoints with `strict=True`. This provenance is recorded in:

```text
runs/.../results/common_stability_audit_provenance.json
```

The fallback is acceptable for the posthoc stability diagnostic because strict checkpoint loading passed after removing shared experts. It remains a source-code boundary for the PCA audit, because its source-file checksums differ from the full-run source manifest.

## 4. Completeness and Sanity

Full run row counts:

```text
trajectory_rows: 1386 / 1386 expected
checkpoint_rows: 231 / 231 expected
replay_rows: 15246 / 15246 expected
fraction_rows: 126 / 126 expected
```

Replay reconstruction:

```text
max_abs(H W^T - logged_router_logits): 0.0
```

Freeze-gradient sanity at the first backward pass:

```text
variant                     router_grad_norm   expert_grad_norm   other_grad_norm
normal                      0.127689           0.596040           5.278476
freeze_gate_all             0.000000           0.596040           5.278476
freeze_experts_all          0.127689           0.000000           5.278476
freeze_gate_and_experts     0.000000           0.000000           5.278476
freeze_prefix_before_layer3 0.087858           0.297060           3.011226
freeze_prefix_before_layer4 0.049354           0.221063           2.574854
freeze_prefix_before_layer5 0.032620           0.145181           1.793366
```

This confirms that the intended frozen parameter groups did not receive gradients.

## 5. Main Result

Primary metric:

```text
common_margin_delta = common_margin(step10) - common_margin(step0)
```

Companion metric:

```text
raw_max_load_delta = max_expert_load(step10) - max_expert_load(step0)
```

Layer-5 results, mean over 3 seeds:

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

Interpretation:

1. `freeze_gate_all` preserves almost all raw load concentration. Gate update is not necessary for raw top-1 route concentration.
2. `freeze_gate_all` only produces part of the normal common-margin spike. Hidden drift under fixed gate is a partial source, not the full explanation.
3. `freeze_experts_all` suppresses common-margin amplification but not raw load concentration. Expert updates contribute strongly to common-logit amplification, but raw top-1 concentration can survive without them.
4. `freeze_gate_and_experts` still has strong raw concentration, so non-expert hidden-producing paths can drive top-1 concentration.
5. `freeze_prefix_before_layer5` suppresses the fast layer-5 spike. The tested layer-5 effect needs the hidden-producing path before the layer-5 router input.

## 6. Replay Decomposition

Replay definition:

```text
Z_{a,b} = H_b W_a^T

A_actual      = common_margin(W10, H10) - common_margin(W0, H0)
A_gate        = common_margin(W10, H0)  - common_margin(W0, H0)
A_hidden      = common_margin(W0, H10)  - common_margin(W0, H0)
A_interaction = A_actual - A_gate - A_hidden
```

Layers 3--5, mean over 3 seeds:

```text
variant                     layer  A_actual  A_gate   A_hidden  A_interaction  raw_load_delta
normal                      3      0.6225    0.0614   0.3912    0.1699         0.7143
normal                      4      0.7974   -0.0009  -0.0560    0.8542         0.7253
normal                      5      1.2760   -0.0137   0.4858    0.8038         0.7334
freeze_gate_all             5      0.5354    0.0000   0.5354    0.0000         0.7314
freeze_experts_all          5      0.1919   -0.0114   0.2891   -0.0858         0.7183
freeze_gate_and_experts     5      0.2872    0.0000   0.2872    0.0000         0.7114
freeze_prefix_before_layer5 5      0.0278    0.0278   0.0000    0.0000         0.0660
```

The normal layer-5 direct gate-only term is slightly negative. The spike is not a standalone router-weight update effect. It is hidden-state drift plus a large gate-hidden interaction.

## 7. Common Direction Stability

Phase A split:

```text
primary audit set: 8 splits x 1024 sequences = 8192 sequences
secondary audit set: 8 splits x 1024 sequences = 8192 sequences
steps: 0, 10, 300
seeds: 0, 1, 2
pca_k: 1, 4, 8
```

Late-layer primary-set stability, mean over seeds:

```text
step  layer  pairwise_cos_mean  pairwise_cos_min  winner_agreement  primary_secondary_cos  cos_to_step0
0     3      0.9869             0.9691            1.0000            0.9973                 1.0000
0     4      0.9858             0.9682            1.0000            0.9972                 1.0000
0     5      0.9858             0.9699            1.0000            0.9972                 1.0000
10    3      0.9999             0.9998            1.0000            1.0000                 0.3039
10    4      1.0000             0.9999            1.0000            1.0000                 0.2899
10    5      1.0000             0.9999            1.0000            1.0000                 0.2574
300   3      0.9955             0.9786            1.0000            0.9998                 0.1185
300   4      0.9955             0.9785            0.9167            0.9998                 0.1314
300   5      0.9957             0.9794            0.8571            0.9998                 0.1417
```

Interpretation:

1. Within a checkpoint, the common direction is highly stable across disjoint batches.
2. At step 10, the spike is not a fixed-audit-batch artifact.
3. Across checkpoints, the common direction rotates substantially. The correct claim is checkpoint-specific batch-stable common direction, not one static common vector through training.

PCA subspace overlap also supports stability. At step 10, layers 3--5 have top-1 subspace overlap between `0.9940` and `0.9979`; top-4 overlap remains between `0.8539` and `0.9191`.

## 8. Hypothesis Decisions

```text
H1 hidden-state drift under fixed random gate is sufficient:
  Supported for raw route concentration and as a partial common-margin source.
  It cannot reproduce the full normal common-margin spike.

H2 gate-hidden interaction is required for full normal common-logit amplification:
  Supported.
  Layer-5 normal A_interaction is 0.8038; freeze_gate_all interaction is 0.0000.

H3 expert parameter updates are necessary:
  Supported for the full common-margin spike.
  Not supported for raw top-1 route concentration.

H4 non-expert hidden-producing paths alone can create concentration:
  Supported for raw top-1 concentration.
  Partial for common-margin amplification.

H5 layer-local hidden producer is necessary for the fast layer-5 spike:
  Supported.
  freeze_prefix_before_layer5 suppresses layer-5 common-margin delta by 97.82% and raw-load delta by 91.00%.

H6 common direction is stable enough:
  Supported within the same checkpoint and across disjoint batches.
  Weakened as a static-across-training direction.
```

## 9. Artifact Map

Primary result files:

```text
runs/.../results/trajectory_metrics.csv
runs/.../results/gradient_metrics.csv
runs/.../results/checkpoint_manifest.csv
runs/.../results/replay_metrics.csv
runs/.../results/replay_fraction_summary.csv
runs/.../results/common_batch_stability_summary.csv
runs/.../results/common_batch_stability_set_compare.csv
runs/.../results/common_checkpoint_stability.csv
runs/.../results/common_subspace_stability.csv
runs/.../results/common_stability_audit_provenance.json
```

Derived compact tables:

```text
runs/.../tables/gradient_freeze_sanity.csv
runs/.../tables/trajectory_l5_step_summary.csv
runs/.../tables/replay_decomposition_mean_l3_l5.csv
runs/.../tables/replay_decomposition_by_seed_l3_l5.csv
runs/.../tables/common_stability_late_layers.csv
```

Figures:

```text
runs/.../figures/trajectory_common_margin_normal.png
runs/.../figures/trajectory_max_load_normal.png
runs/.../figures/trajectory_common_margin_freeze_prefix_before_layer5.png
runs/.../figures/trajectory_max_load_freeze_prefix_before_layer5.png
runs/.../figures/replay_heatmap_common_margin_normal.png
runs/.../figures/replay_heatmap_raw_max_load_normal.png
```

## 10. Claim Boundary

This run covers only:

```text
small random-initialized Qwen-style MoE
6 layers, hidden size 512, 8 experts
top-1 routing
no shared expert
no load-balance loss
DCLM packed text
0--300 training steps
```

It does not cover pretrained large MoEs, shared-expert MoEs, top-2 routing, final expert utility specialization, or mitigation methods.

Phase C expert-output forward-feedback was not run. Therefore this experiment does not decide whether forward expert output is the remaining cause of the raw load concentration that survives gate/expert freezing.

The common-stability diagnostic used a key-compatible fallback source because the original `MoE_Router` path was unavailable on later ACP workers. The fallback strict-loaded checkpoints after disabling shared experts, but source checksums differ from the original full-run source. Treat common-stability as a strong posthoc diagnostic, not as an additional training intervention.

## 11. Next Decision

The next experiment should target the remaining gap:

```text
Why does raw top-1 load concentration survive when gate weights and expert parameters are frozen?
```

Minimal next tests should distinguish residual anisotropy, position structure, and forward expert-output feedback. The already-closed part is the narrower common-margin claim: the fast layer-5 common-logit spike requires the layer-5 hidden-producing path and gate-hidden interaction.
