# 05_04_03 Real-Text Common-Logit Initialization Anchor

## 0. Tiny Summary

父问题：为什么 sparse top-1 MoE router 在早期进入 route concentration / expert underuse？
当前子问题：DCLM real text 上，随机初始化线性 top-1 router 是否在 step 0 就被 common-logit offset 主导？
实验结论：step 0 的 common-logit dominance 被削弱；common component 不是最大 logit-margin 来源，但 centering 明显降低 max load。
关键转向：训练 10 步后 common 通道快速放大，route concentration 更像早期训练反馈问题，而不是纯随机初始化几何必然。
当前边界：只裁定该小型 Qwen-style top-1 MoE、DCLM packed text、随机初始化和 300-step 早期训练。
下一步决策：做 early-training causal split，分别冻结 gate、hidden-state producer 和 expert-output feedback。

## 1. Problem Definition

Parent problem:
Why do ordinary sparse top-1 MoE routers underuse experts or enter route concentration before useful specialization can form?

Sharper subproblem:
In real DCLM language-modeling data, does a random-initialized Qwen-style linear MoE router already contain a common-logit winner before meaningful training?

Decision question:
On DCLM packed-token text, does random-initialized sparse top-1 linear routing concentrate because $w_e^T c$ dominates $w_e^T r_i$ in the actual router-gate input?

Not in scope:
This anchor does not test pretrained MoE behavior, final specialization, expert utility, or a deployable anti-common initialization.

## 2. Physical Prior

P1:
Router-input hidden states in real text contain a shared direction or low-dimensional common component.
Meaning:
Token distribution, position, residual stream scale, and layer normalization can create structure shared by many routed tokens.
Could be wrong if:
The estimated common component is unstable across layers, seeds, batches, or position buckets.

P2:
A random bias-free linear gate can convert the common component into an expert-specific offset.
Meaning:
For a gate row $w_e$, the term $w_e^T c$ is shared across tokens but differs across experts.
Could be wrong if:
The spread of $w_e^T c$ is small relative to token residual margins.

P3:
Increasing expert count increases the chance of an extreme common-projection winner.
Meaning:
With more random rows, the maximum common projection can become more extreme and reduce active-expert ratio.
Could be wrong if:
Residual token differences or anisotropic zero-mean covariance dominate the routing distribution.

## 3. Hypothesis

H1:
At step 0, common-logit margin exceeds residual margin in multiple layers and seeds.
Supported if:
`dominance_ratio > 1` and common-winner agreement is above random routing in most audited layers.
Weakened if:
Residual margins dominate and common-winner agreement is close to $1/E$.
Status:
Weakened. Final run has mean `dominance_ratio=0.5251`; only 1 of 18 layer/seed cases has `dominance_ratio > 1`.

H2:
Removing the common component from the actual gate input reduces route concentration.
Supported if:
Centered routing lowers `max_load` and increases active experts without changing the router weights.
Weakened if:
Raw and centered routing have similar load, entropy, and active-expert curves.
Status:
Supported in the narrower load-bias sense. Step-0 `raw_max_load=0.2781`, `centered_max_load=0.1561`, and `delta_max_load=0.1220`.

H3:
Expert-count scaling amplifies common-winner dominance.
Supported if:
Virtual larger $E$ increases common margin and lowers raw active-expert ratio more than centered routing.
Weakened if:
Raw and centered scaling curves remain similar or do not worsen with $E$.
Status:
Weakened for virtual random gates. `dominance_ratio` stays below 1 and does not rise monotonically with $E$.

## 4. Minimal Mathematical Model

Objects:
`h_i` is the exact tensor passed into the active linear gate for routed token `i`; `c = mean_i h_i`; `r_i = h_i - c`; `w_e` is the bias-free gate row for expert `e`.

Core decomposition:
$$
h_i = c + r_i,\qquad z_{i,e}=w_e^T h_i=w_e^T c+w_e^T r_i
$$

Mechanism relation:
If the top expert gap in $w_e^T c$ is larger than typical residual top gaps, top-1 routing can select a shared common winner for many tokens.

Observable metrics:
`common_margin`, `residual_margin`, `dominance_ratio`, `common_winner_agreement`, `max_load`, `routing_entropy`, `active_experts_ratio`, and raw-vs-centered deltas.

Falsifier:
If common margin is not large, common winner does not predict routes, and centering does not change concentration, common-logit bias is not the main initialization explanation.

## 5. Minimal Tests

Experiment:
Step-0 common-logit audit.
Question:
Does the exact gate input at random initialization show common-logit dominance?
Intervention:
None; compare raw routing with centered routing on the same gate weights.
Primary metric:
`dominance_ratio` with `common_winner_agreement` and `delta_max_load`.
Supports:
Common margin dominates and centering reduces concentration.
Weakens:
Residual routing explains routes and centering has little effect.

Experiment:
Virtual expert-count scaling.
Question:
Does increasing random expert count amplify common-winner dominance on the same hidden states?
Intervention:
Sample virtual bias-free gates for $E = 4..1024$ using protocol and matched real-gate scales.
Primary metric:
Trend of `common_margin`, `dominance_ratio`, and `active_experts_ratio`.
Supports:
Raw concentration worsens with $E$ and centered routing weakens the trend.
Weakens:
Raw and centered curves stay similar.

Experiment:
Early-training and actual-E validation.
Question:
Does the initialization bias persist or amplify during the first 300 optimizer steps, and does actual $E$ worsen underuse?
Intervention:
Train random-init top-1 MoE with no shared expert and no load-balance loss, auditing fixed DCLM data at exact checkpoints.
Primary metric:
Checkpoint trajectory of common dominance and active-expert ratio across actual $E$.
Supports:
Bias is visible at step 0 and persists or worsens before meaningful LM-loss improvement.
Weakens:
Concentration emerges only late or does not scale with actual $E$.

## 6. Falsifiable Outcomes

Supported part:
Centering the exact gate input reduces route concentration, so the common component is a real step-0 load-bias source.

Weakened:
Residual margins dominate step-0 common margins, and virtual expert-count scaling does not amplify common-margin dominance.

New supported direction:
Early training rapidly amplifies common-logit concentration. For actual $E=8$, step 10 has `dominance_ratio=21.7874` and `raw_max_load=0.8507`.

Insufficient evidence:
The run does not capture the exact gate input, uses shared experts or oracle gating, omits seed/layer coverage, or reports only load without common-logit decomposition.

## 7. Current Evidence

Observation:
The final DCLM real-text run succeeded on 4 ACP GPUs: `pt-mrx0wq1v`, run `real_text_common_logit_audit_v5_4gpu_20260611_r3`.

Interpretation:
The exact gate-input reconstruction error is `0.0`, so the decomposition audited the intended router object.

Phase 1 observation:
At step 0, mean `common_margin=0.1237`, `residual_margin=0.2364`, and `dominance_ratio=0.5251`. This weakens common-logit dominance as the main random-init explanation.

Phase 1 load observation:
At step 0, `raw_max_load=0.2781` and `centered_max_load=0.1561`. This supports a narrower common-component load-bias claim.

Phase 2 observation:
Virtual $E=4..1024$ does not produce monotone common-margin amplification; `dominance_ratio` remains around `0.41..0.45` under protocol scale.

Phase 3 observation:
For actual $E=8$, step 10 has `common_margin=0.7427`, `residual_margin=0.0707`, `dominance_ratio=21.7874`, and `raw_max_load=0.8507`.

Phase 4 observation:
By step 300, actual larger $E$ shows more underuse risk: raw active-expert ratio is `0.9792` at $E=16$ and `0.9149` at $E=32$.

Evidence location:
Real-text evidence is recorded in `Projects/from-attention-to-search/main/experiments/A05/A05_04_03_real_text_common_logit_audit/summary.md` and `detailed.md`.

Daily report card:
`daily_research_reports/0611/real_text_common_logit_audit/05_report_card.md`

## 8. Claim Boundary And Next Decision

Can claim if supported:
For this small random-initialized Qwen-style top-1 MoE on DCLM packed text, the common component is a real route-concentration bias source, but it is not the dominant step-0 logit-margin source.

Current interpretation:
The main mechanism should be reframed from pure initialization common-logit dominance to early-training common-logit amplification.

Cannot claim:
The result proves final feature specialization, solves MoE collapse, identifies the cause of step-10 amplification, or applies to pretrained large MoEs.

Next decision:
Run a causal early-training split: freeze gate weights, freeze hidden-state producers, and freeze expert outputs in separate short runs while keeping the same DCLM audit and exact gate-input decomposition.
