# Protocol For Approval: A06_02_01 Cross-Checkpoint Replay

Primary anchor: `Projects/from-attention-to-search/main/problem_anchors/06_02_early_training_common_logit_amplification_anchor.md`

Anchor decision question: Is early common-logit amplification mainly carried by gate-weight update, hidden-state producer drift, expert-output feedback, or their interaction?

Experiment role: staged causal audit. Phase A/B localize the spike by replaying $W_aH_b$ with streamed or recomputed gate inputs; Phase C validates the replay-indicated path, including a required final-layer-only hidden-producer freeze.

Stage mapping:
- Phase A normal trajectory -> reproduce the step-10 amplification.
- Phase B cross-checkpoint replay -> attribute amplification to $W$, $H$, or interaction.
- Phase C freeze split -> validate necessity of the replay-indicated path; `freeze_hidden_producer` is scoped to the final MoE layer only.

Goal: decompose the step-10 common-logit and route-concentration spike in the same small random-initialized DCLM top-1 MoE setting as the previous real-text audit.

Decision question: Does gate update, hidden-state drift, expert-feedback, or an interaction explain most of the early amplification?

Tested hypothesis: common component is a load-bias source at step 0 and is amplified by early selected-path training dynamics.

Rival explanation: the spike is a measurement artifact, audit-batch instability, high-frequency token effect, residual anisotropy, or an unlocalized optimizer effect not attributable to $W$ or $H$.

Data: DCLM packed stream, 257-token spans, 256 valid input tokens, fixed audit batch of 8192 sequences, training split matching the previous real-text run.

Model: random-initialized small Qwen-style decoder-only MoE, 6 layers, hidden size 512, 8 attention heads, 4 KV heads, expert hidden size 2048, vocabulary 151936.

Routing / algorithm: bias-free linear sparse top-1 gate; exact active gate input as $H_t$; `use_shared_expert=false`; `lambda_lb=0.0`; `norm_topk_prob=false`; no oracle gating; no multihead routing.

Loss / objective: next-token LM objective on DCLM packed text.

Conditions: train normal model for 300 steps and audit checkpoints `0,1,2,3,5,10,20,50,100,200,300`. Use 3 seeds and all 6 layers for Phase A/B unless the approved preflight check reveals resource limits. Do not save all full `H_t` tensors; stream or recompute gate inputs and write aggregated replay metrics plus small debug shards only. Phase C includes `freeze_gate`, `freeze_experts`, and `freeze_last_layer_hidden_producer`.

Final-layer hidden-producer freeze: audit and validate this variant only at the final MoE layer. Freeze all modules that can change the final layer's gate input before its MoE gate: token embeddings, layers before the final layer, and the final layer's pre-gate attention / layernorm path. Keep the final layer gate and expert path trainable. Final norm / LM head may remain trainable because they do not change the final layer gate input; record the exact module list in `run_config.json`.

Primary metric: `common_margin` amplification from $W_0H_0$ to $W_{10}H_{10}$, decomposed into gate-only $W_{10}H_0$, hidden-only $W_0H_{10}$, and interaction.

Secondary metrics: `raw_max_load`, `centered_max_load`, `delta_max_load`, `residual_margin`, `dominance_ratio` with epsilon/log reporting, routing entropy, effective experts, active-expert ratio, dominant-expert persistence, `cos(c_t,c_0)`, gradient norms.

Known good case: replay $W_tH_t$ must reproduce logged checkpoint logits with max absolute error at or below `1e-6` in fp32-equivalent computation.

Known bad case: a deliberately mismatched hidden tensor or non-gate hidden state must fail the reconstruction check.

Known confusing case: $W_{10}H_0$ and $W_0H_{10}$ each explain little, while $W_{10}H_{10}$ spikes; this indicates interaction, not protocol failure.

Success: Phase A reproduces the step-10 spike, and Phase B attributes most common-margin amplification to gate, hidden, or interaction with stable seed/layer evidence.

Failure: Phase A does not reproduce the spike, or replay attribution is unstable and cannot reproduce actual checkpoint metrics.

Insufficient evidence: audit batch changes, exact gate input is not captured, shared expert or load-balance loss is enabled, reconstruction error is non-negligible, or only max-load metrics are reported.

What this cannot claim: pretrained large-MoE behavior, final expert specialization, deployable mitigation, or the necessity of a causal path until matching freeze evidence is run.

User approval checklist:
- Approve Phase A/B plus required final-layer-only `freeze_hidden_producer` validation.
- Approve checkpoint list `0,1,2,3,5,10,20,50,100,200,300`.
- Approve resource policy: stream/recompute hidden states instead of saving all full `H_t` tensors.
- Approve final-layer-only boundary for hidden-producer freeze claims.
- Decide after Phase B which Phase C freeze variants should be run.
