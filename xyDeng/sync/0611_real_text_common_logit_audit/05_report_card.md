# 05 Report Card: Real-Text Common-Logit Audit

## Source Files

- Anchor: `Projects/from-attention-to-search/main/problem_anchors/05_04_03_real_text_common_logit_initialization_anchor.md`
- Summary: `Projects/from-attention-to-search/main/experiments/A05_04_02_real_text_common_logit_audit/summary.md`
- Detailed: `Projects/from-attention-to-search/main/experiments/A05_04_02_real_text_common_logit_audit/detailed.md`
- Final run: `Projects/from-attention-to-search/main/experiments/A05_04_02_real_text_common_logit_audit/runs/real_text_common_logit_audit_v5/real_text_common_logit_audit_v5_4gpu_20260611_r3`
- ACP job: `pt-mrx0wq1v`

## Executive Summary

The real-text audit weakens the original step-0 common-logit dominance hypothesis.

At random initialization, residual token differences are larger than the shared common projection: `common_margin=0.1237`, `residual_margin=0.2364`, and `dominance_ratio=0.5251`. Only 1 of 18 layer/seed cases has `dominance_ratio > 1`.

But the common component still matters for load: centering the exact gate input reduces `max_load` from `0.2781` to `0.1561`. During training, common-logit concentration becomes much stronger: by step 10, `dominance_ratio=21.7874` and `raw_max_load=0.8507`.

Current decision:
Treat common-logit bias as a real load-bias factor, but move the main causal question from pure random-init geometry to early-training amplification.

## Research Process Update

The experiment implemented the protocol with:

- no shared expert;
- exact active gate input as $h_i$;
- non-oracle `hidden_states` gating reference;
- sparse top-1 routing;
- `lambda_lb=0.0`;
- `norm_topk_prob=false`, so non-selected router logits still receive normal softmax-denominator gradients;
- 257-token DCLM spans, 256 valid input tokens;
- exact audits at steps `0,1,10,50,100,300`.

The final 4-GPU ACP run succeeded and passed row-count and gate-reconstruction checks.

## Terms Used Here

Common component:
The mean vector $c$ of the exact router inputs $h_i$ over audited tokens.

Residual component:
The token-specific difference $r_i=h_i-c$.

Common margin:
The top-expert gap from $w_e^T c$.

Residual margin:
The top-expert gap from $w_e^T r_i$.

Dominance ratio:
`common_margin / residual_margin`. Values above 1 support common-logit dominance.

Centered routing:
Routing after replacing $h_i$ with $h_i-c$. It tests how much the mean component contributes to concentration without changing router weights.

## Key Figures

Step 0 does not show common-margin dominance.

![Phase 1 common vs residual margin](figures/phase1_common_vs_residual_margin.png)

Step 0 still shows a real load effect from the common component.

![Phase 1 raw vs centered load](figures/phase1_raw_vs_centered_load.png)

Virtual larger $E$ does not produce monotone common-margin amplification.

![Phase 2 virtual E common margin](figures/phase2_E_vs_common_margin_protocol.png)

Training rapidly amplifies route concentration.

![Phase 3 max load over steps](figures/phase3_max_load_over_steps.png)

Actual larger $E$ shows more underuse risk by step 300.

![Phase 4 active experts](figures/phase4_E_train_vs_active_experts_ratio.png)

## Current Claim

For this small random-initialized Qwen-style top-1 MoE on DCLM packed text, the common component is a real source of route-concentration bias, but it is not the dominant step-0 logit-margin source. Early training rapidly amplifies common-logit concentration.

## Claim Boundary

This does not prove final expert specialization failure, pretrained MoE behavior, or a deployable mitigation. It also does not yet identify whether the step-10 amplification is caused by gate updates, hidden-state drift, expert-output feedback, or optimizer geometry.

## Next Step

Run a causal early-training split:
freeze gate weights, freeze hidden-state producers, and freeze expert outputs in separate short runs while keeping the same DCLM audit and exact gate-input decomposition.
