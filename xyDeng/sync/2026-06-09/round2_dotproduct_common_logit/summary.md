# A05_04_02 Round-2 Dot-Product Common-Logit Summary

## Conclusion

The full Round-2 run is completed. The protocol-required final documents exist under the formal result directory as `round2_summary.md` and `round2_detailed.md`; this file is the project-standard reading entry.

Common-logit dominance is already visible at step 0, grows before step 10, predicts the final dominant expert, and common-logit cancellation improves slot-level specialization while preserving accuracy. Within this toy dot-product top-1 setting, this supports common-logit dominance as a causal contributor to random-init collapse.

## Purpose

Round 1 showed that no-position sparse top-1 MoE can still collapse even when routed hidden states contain slot information. Round 2 tests whether a shared common logit component explains the collapse under a dot-product router.

## How

- Router: dot-product only, $score_e(h)=w_e^T h$.
- Routing: sparse top-1 only.
- Dataset: unchanged Round-1 moving block, `prefix filler + SLOT_s B_CONST Y_s + suffix filler`.
- Positional encoding: disabled.
- Loss and routing metrics: computed at the `B_CONST` position unless a P4 diagnostic explicitly changes the audit position.
- Sections executed: R2-P0, R2-P1, R2-P2, R2-P3, R2-P4.

## Metrics

The primary decision metric is slot-level route specialization, measured by route-slot NMI and purity/heatmap structure, while preserving target accuracy. Load balance is reported but is not treated as feature specialization.

Supporting metrics include dot-product reconstruction error, common margin, slot margin, early common-argmax prediction of the final dominant expert, and source-ablation common/slot margins.

## Results

- R2-P0 decomposition sanity: mean reconstruction error $1.073e^{-7}$, max error $5.849e^{-7}$; common-dominant cells 2238/2560.
- R2-P1 timing: step-0 common margin 0.2993 vs slot margin 0.0523; step-10 common margin 1.7087 vs slot margin 0.2993.
- R2-P1 prediction: common argmax predicts the final dominant expert at 0.844 at step 0 and 0.969 at step 10.
- R2-P2 slot-init basin: first alpha with mean final NMI >= 0.90 is 0.2.
- R2-P3 cancellation: baseline final slot NMI 0.080; cancel 0-10 final slot NMI 0.896; cancel 0-final final slot NMI 0.963; all preserve accuracy 1.000.
- R2-P4 source audit: fixed `B_CONST` / B-token identity is a likely major contributor; routing at the B position also contributes; filler/template variation is not the main supported source.

## What Result A Means

If common cancellation improves route-slot NMI while preserving accuracy, the common-logit component is not merely a measurement artifact or load-balance issue. It is a plausible causal contributor to random-init collapse in this controlled setting.

## What Result B Means

If source ablation changes common margin without producing robust slot specialization, it helps localize likely common-component sources but does not by itself prove the full causal source. P4 supports `B_CONST` / routed-position identity as likely contributors, not a complete source identification.

## Claim Boundary

The claim applies only to the toy uniform no-position synthetic task, dot-product router, sparse top-1 routing, and B-position loss/metrics. It does not show transfer to real language models and does not prove expert computation is causally slot-specialized.

## Artifact Map

- Formal result directory: `results/round2_dotproduct_common_logit_4gpu_20260609/`
- Protocol final summary: `results/round2_dotproduct_common_logit_4gpu_20260609/round2_summary.md`
- Protocol final detailed record: `results/round2_dotproduct_common_logit_4gpu_20260609/round2_detailed.md`
- Figures: `figures/round2_dotproduct_common_logit_4gpu_20260609/`
- Run status: `results/round2_dotproduct_common_logit_4gpu_20260609/run_summary.json`
- ACP job id: `pt-fplz4uf0`

## Next Decision

Design a minimal anti-common or anti-lock-in router that preserves sparse top-1 routing while making slot specialization reachable from random initialization.
