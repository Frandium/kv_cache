# A05_04_03 Real-Text Common-Logit Audit Summary

## Purpose

This experiment tests whether the toy common-logit mechanism transfers to real DCLM text in a random-initialized Qwen-style sparse top-1 MoE router.

Decision question:
On DCLM packed-token text, does random-initialized sparse top-1 linear routing concentrate because $w_e^T c$ dominates $w_e^T r_i$ in the actual router-gate input?

## Conclusion

Result: mixed, with the original step-0 dominance hypothesis weakened.

At step 0, the common component does not dominate the residual component on average: `common_margin=0.1237`, `residual_margin=0.2364`, and `dominance_ratio=0.5251`. Only 1 of 18 layer/seed cases has `dominance_ratio > 1`. This weakens the claim that random initialization is already controlled by a common-logit winner.

However, removing the mean component from the exact gate input still reduces route concentration: `raw_max_load=0.2781` versus `centered_max_load=0.1561`, with `delta_max_load=0.1220`. This supports a narrower claim: the common component is not the largest logit-margin source at step 0, but it still materially biases top-1 load.

Training changes the picture quickly. For actual $E=8$, by step 10 the common margin rises to `0.7427`, residual margin falls to `0.0707`, `dominance_ratio` jumps to `21.7874`, and `raw_max_load` reaches `0.8507`. The likely next target is therefore early optimizer feedback and representation drift, not pure initialization geometry alone.

## Key Evidence

Final full run:
`runs/real_text_common_logit_audit_v5/real_text_common_logit_audit_v5_4gpu_20260611_r3`

ACP job:
`pt-mrx0wq1v`, 4 GPUs, status `SUCCEEDED`, completed at `2026-06-11T17:39:10`.

Completeness:
`phase1_rows=18`, `phase2_rows=20736`, `phase3_rows=108`, `phase4_rows=432`.

### Phase 1: Step-0 Real Gate Audit

Primary object:
`h_i` is the exact tensor passed into the active linear gate. The runner verifies `router_logits == h @ W.T`; max reconstruction error is `0.0`.

| metric | mean |
| --- | ---: |
| common_margin | 0.1237 |
| residual_margin | 0.2364 |
| dominance_ratio | 0.5251 |
| raw_max_load | 0.2781 |
| centered_max_load | 0.1561 |
| delta_max_load | 0.1220 |
| raw_active_experts_ratio | 1.0000 |

Interpretation:
Residual token differences decide the step-0 margin more than the shared mean component. But the shared mean still pushes many tokens toward the same expert, because centering reduces max load by about 12.2 percentage points.

![Phase 1 common vs residual margin](runs/real_text_common_logit_audit_v5/real_text_common_logit_audit_v5_4gpu_20260611_r3/figures/phase1_common_vs_residual_margin.png)

![Phase 1 raw vs centered load](runs/real_text_common_logit_audit_v5/real_text_common_logit_audit_v5_4gpu_20260611_r3/figures/phase1_raw_vs_centered_load.png)

### Phase 2: Virtual Expert-Count Scaling

Virtual random gates were sampled for $E=4,8,16,32,64,128,256,512,1024$ with both protocol scale and matched real-gate scale.

Under protocol scale, `dominance_ratio` stays around `0.41..0.45` rather than rising with $E$. `raw_max_load` is always higher than `centered_max_load`, but absolute `raw_max_load` decreases as $E$ increases, as expected from more bins.

Interpretation:
This does not support the stronger claim that simply increasing expert count amplifies common-margin dominance at random initialization. It does support the narrower raw-vs-centered load effect.

![Phase 2 virtual E common margin](runs/real_text_common_logit_audit_v5/real_text_common_logit_audit_v5_4gpu_20260611_r3/figures/phase2_E_vs_common_margin_protocol.png)

### Phase 3: Early Training Trajectory, Actual $E=8$

| step | lm_loss | common_margin | residual_margin | dominance_ratio | raw_max_load | centered_max_load |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | n/a | 0.1237 | 0.2364 | 0.5251 | 0.2781 | 0.1561 |
| 1 | 12.0581 | 0.1432 | 0.2275 | 0.6300 | 0.3362 | 0.1606 |
| 10 | 10.8455 | 0.7427 | 0.0707 | 21.7874 | 0.8507 | 0.2504 |
| 50 | 7.6703 | 0.2405 | 0.1039 | 3.7652 | 0.8113 | 0.3191 |
| 100 | 7.6394 | 0.5418 | 0.4278 | 1.7050 | 0.6876 | 0.3974 |
| 300 | 7.3875 | 1.1931 | 0.5500 | 1.6448 | 0.5870 | 0.3255 |

Interpretation:
The common-logit channel becomes dominant after a few optimizer steps. The mechanism is therefore more plausibly an early-training amplification or feedback effect than a pure step-0 geometric inevitability.

![Phase 3 max load over steps](runs/real_text_common_logit_audit_v5/real_text_common_logit_audit_v5_4gpu_20260611_r3/figures/phase3_max_load_over_steps.png)

### Phase 4: Actual Expert-Count Sweep

At step 0, increasing actual $E$ from 4 to 32 does not worsen active-expert ratio; all experts are used. By step 300, raw active-expert ratio falls for larger $E$: `0.9792` at $E=16$ and `0.9149` at $E=32$.

| num_experts | step300 lm_loss | common_margin | residual_margin | dominance_ratio | raw_max_load | centered_max_load | raw_active_experts_ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 7.3185 | 1.4342 | 0.5159 | 2.1871 | 0.6814 | 0.4244 | 1.0000 |
| 8 | 7.3875 | 1.1931 | 0.5500 | 1.6448 | 0.5870 | 0.3255 | 1.0000 |
| 16 | 7.0431 | 0.6843 | 0.6819 | 1.2467 | 0.4714 | 0.2951 | 0.9792 |
| 32 | 7.6378 | 0.7006 | 0.5666 | 1.1910 | 0.3951 | 0.2508 | 0.9149 |

Interpretation:
Actual larger $E$ increases underuse risk during early training, but not through the originally predicted monotone increase in step-0 common-margin dominance.

![Phase 4 actual E active experts](runs/real_text_common_logit_audit_v5/real_text_common_logit_audit_v5_4gpu_20260611_r3/figures/phase4_E_train_vs_active_experts_ratio.png)

## Claim Boundary

Can claim:
For this small random-initialized Qwen-style top-1 MoE on DCLM packed text, the common component is a real load-bias source, but it is not the dominant step-0 logit-margin source. Early training rapidly amplifies common-logit concentration.

Cannot claim:
This does not prove final specialization failure, does not explain pretrained MoEs, and does not yet identify whether the step-10 amplification is caused by gate update, hidden-state drift, expert-output feedback, or optimizer geometry.

## Next Decision

Refine the next experiment toward early-training causality:
separate gate-weight update, hidden-state representation drift, and expert-output feedback by freezing one component at a time while keeping the same DCLM audit and exact gate-input decomposition.
