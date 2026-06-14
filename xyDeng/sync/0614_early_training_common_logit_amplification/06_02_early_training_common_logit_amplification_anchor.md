# 06_02 Early-Training Common-Logit Amplification In Real Top-1 MoE

## 0. Tiny Summary

父问题：为什么 sparse top-1 MoE router 在早期进入 route concentration / expert underuse？
当前子问题：真实 DCLM top-1 MoE 中，为什么 step 0 common dominance 不强，但 step 10 common-logit 和负载集中快速放大？
核心假设：common component 先作为 load-bias source，随后被 gate update、hidden-state drift、expert feedback 或它们的交互放大。
最小测试：在固定 audit batch 上做 cross-checkpoint replay：$W_a H_b$，再用 freeze split 验证 replay 指向的 causal path。
当前边界：只解释 small random-initialized Qwen-style top-1 MoE on DCLM 的 0--300 step 早期训练；Phase C expert-output forward feedback 尚未测试。
下一步决策：是否继续拆分 raw load concentration 的残余机制，即 residual anisotropy、position structure、forward expert-output feedback 三者谁在起主要作用。

## 1. Problem Definition

Parent problem:
Why do ordinary sparse top-1 MoE routers underuse experts or enter route concentration before useful specialization can form?

Sharper subproblem:
In real DCLM language-model training, the common component is not the dominant step-0 logit-margin source, but common-logit concentration spikes by step 10.

Decision question:
Is the early common-logit amplification mainly carried by gate-weight update, hidden-state producer drift, expert-output feedback, or their interaction?

Not in scope:
This anchor does not test pretrained large MoEs, final expert utility specialization, deployable mitigation, or mentor-style expert-spectrum initialization.

## 2. Physical Prior

P1:
The router-input common component is stable enough across early checkpoints to receive coherent optimizer pressure.
Meaning:
If $c_t$ stays aligned, repeated gradients can accumulate along a shared direction instead of averaging out.
Could be wrong if:
Common directions rotate strongly across checkpoints or audit batches while concentration still appears.

P2:
Top-1 routing creates selected-path feedback.
Meaning:
The selected expert receives traffic and task gradient, while under-selected experts receive less corrective signal.
Could be wrong if:
Freezing gate or expert updates does not change the step-10 concentration trajectory.

P3:
Expert outputs can reshape later router inputs.
Meaning:
Selected experts are not passive receivers; their outputs can alter downstream hidden states and reinforce early route choices.
Could be wrong if:
Expert-freeze or expert-feedback blocking preserves the same amplification pattern.

## 3. Hypothesis

H1:
Gate update explains most early common-logit amplification.
Supported if:
$W_{10}H_0$ reproduces most of the $W_{10}H_{10}$ common-margin increase, and `freeze_gate` suppresses the step-10 spike.
Weakened if:
$W_{10}H_0$ stays near $W_0H_0$, or `freeze_gate` does not suppress concentration.

H2:
Hidden-state drift explains most amplification.
Supported if:
$W_0H_{10}$ reproduces most of the spike, and hidden-producer freezing suppresses concentration.
Weakened if:
$W_0H_{10}$ remains near baseline while actual $W_{10}H_{10}$ spikes.

H3:
Gate and hidden changes interact, possibly through expert feedback.
Supported if:
$W_{10}H_0$ and $W_0H_{10}$ each explain only a minority, but $W_{10}H_{10}$ is large and expert-freeze suppresses it.
Weakened if:
Either gate-only or hidden-only replay already explains the spike without freeze evidence.

## 4. Minimal Mathematical Model

Objects:
For checkpoint $t$, $H_t$ is the exact active gate input on the fixed audit batch, $W_t$ is the bias-free gate matrix, $h_{t,i}=c_t+r_{t,i}$, and $z_{a,b,i,e}=w_{a,e}^{\top}h_{b,i}$.

Core decomposition:
$$
z_{a,b,i,e}=w_{a,e}^{\top}c_b+w_{a,e}^{\top}r_{b,i}
$$

Mechanism relation:
If $W_t$ or $H_t$ moves so that the common-score spread grows faster than residual-score spread, top-1 routing can concentrate even when step-0 common dominance was weak.

Observable metrics:
`common_margin`, `residual_margin`, `dominance_ratio`, `raw_max_load`, `centered_max_load`, `delta_max_load`, `routing_entropy`, `effective_experts`, and replay fractions for common-margin amplification.

Falsifier:
If replay cannot attribute the step-10 spike to $W$, $H$, or their interaction, and freeze variants do not change the spike, this common-logit amplification model is insufficient.

## 5. Minimal Tests

Experiment:
Normal 300-step trajectory.
Question:
Does the step-10 amplification reproduce with exact gate-input audits?
Intervention:
None; train the same random-init DCLM top-1 MoE and audit fixed checkpoints.
Primary metric:
Trajectory of `common_margin` and `raw_max_load`.
Supports:
Step-10 common-margin and max-load spike recurs.
Weakens:
Trajectory remains flat or no common-channel spike appears.

Experiment:
Cross-checkpoint replay.
Question:
Does the spike live in gate weights, hidden states, or interaction?
Intervention:
Compute $Z_{a,b}=H_bW_a^{\top}$ for saved checkpoints.
Primary metric:
Fractions of step-10 `common_margin` amplification explained by $W_{10}H_0$, $W_0H_{10}$, and interaction.
Supports:
One path or interaction explains most amplification.
Weakens:
Fractions are unstable or fail to reproduce actual checkpoint metrics.

Experiment:
Freeze split validation.
Question:
Is the replay-indicated path necessary during training?
Intervention:
Run `freeze_gate`, `freeze_experts`, and final-layer-only `freeze_hidden_producer`.
Primary metric:
Suppression of step-10 `common_margin` and `raw_max_load`.
Supports:
The matched freeze condition suppresses the spike.
Weakens:
Freeze variants preserve the same spike.

## 6. Falsifiable Outcomes

Supported:
Replay attributes most step-10 amplification to a specific path or interaction, and the matching freeze intervention suppresses the spike.

Weakened:
Replay shows no stable attribution, or freeze interventions do not alter step-10 concentration.

Insufficient evidence:
The run misses exact gate inputs, changes the audit batch across checkpoints, enables shared experts or load-balance loss, reports only load without common-logit decomposition, or lacks seed/layer coverage.

## 7. Current Evidence

Observation:
The A06_02_01 full run reproduced early route concentration in the normal condition. In layer 5, normal training moved from `common_margin=0.1482`, `raw_max_load=0.2582` at step 0 to `common_margin=1.4242`, `raw_max_load=0.9916` at step 10.

Observation:
Cross-checkpoint replay in A06_02_01 weakened a router-weight-only explanation. In layer 5, the step-10 actual common-margin increment was `1.2760`, while gate-only replay contributed `-0.0137`, hidden-only replay contributed `0.4858`, and the remaining interaction was `0.8038`.

Observation:
A06_02_02 freeze split shows that raw route concentration and common-logit amplification are related but distinct. In layer 5, `freeze_gate_all` keeps almost the full raw load spike (`raw_max_load_delta=0.7314` vs `0.7334` normal), but only part of the common-margin spike (`common_margin_delta=0.5354` vs `1.2760` normal). Therefore fixed-gate hidden drift can concentrate top-1 routes, while the full common-margin spike requires more than fixed-gate hidden drift.

Observation:
Expert updates strongly contribute to the common-margin channel but are not necessary for raw load concentration. In layer 5, `freeze_experts_all` reduces `common_margin_delta` from `1.2760` to `0.1919`, while `raw_max_load_delta` remains `0.7183`.

Observation:
The prefix ladder localizes the fast layer-5 spike to the layer-5 hidden-producing path. In layer 5, `freeze_prefix_before_layer5` reduces `common_margin_delta` to `0.0278` and `raw_max_load_delta` to `0.0660`, suppressing the normal step-10 effect by `97.82%` and `91.00%` respectively.

Observation:
The common direction is batch-stable at a fixed checkpoint but not static across training. In layers 3--5 at step 10, split-level common vectors have `pairwise_cos_mean >= 0.9999`, primary-secondary cosine is `1.0000`, and common-winner agreement is `1.0000`. But layer-5 cosine to step 0 is only `0.2574` at step 10 and `0.1417` at step 300.

Execution note:
The A06_02_02 full ACP job succeeded. The posthoc common-stability audit used a key-compatible fallback source because the original `/data/250010109/MoE_Router` source path was unavailable on later ACP workers. The fallback disabled shared experts and strict-loaded checkpoints; source checksums still differ, so this is a posthoc diagnostic boundary.

Evidence location:
`Projects/from-attention-to-search/main/experiments/A06_02_02_freeze_split_common_stability/summary.md`

## 8. Claim Boundary And Next Decision

Can claim:
For this small random-initialized Qwen-style DCLM top-1 MoE, early raw top-1 route concentration can arise under a fixed random gate through hidden-state drift, but the full late-layer common-margin spike requires the hidden-producing path before the audited router input and gate-hidden interaction. Expert parameter updates amplify the common-margin channel, but raw load concentration can survive without them.

Cannot claim:
We have not yet tested Phase C forward expert-output feedback interventions. Therefore we cannot claim whether the raw load concentration that survives gate/expert freezing is caused by residual anisotropy, position structure, or forward expert-output feedback. This also does not explain pretrained large MoEs, shared-expert MoEs, top-2 routing, or final expert specialization.

Next decision:
Decide whether to run a focused residual/position/expert-output split for the remaining raw-load-concentration mechanism. The common-margin claim is already narrowed: the fast layer-5 common-logit spike requires layer-5 hidden-producer drift and gate-hidden interaction.
