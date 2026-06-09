# Round-2 Detailed Record

## Source Files

- Protocol: `daily_research_reports/0609/round2_dotproduct_common_logit_protocol.md`
- Anchor: `Projects/from-attention-to-search/main/problem_anchors/05_04_02_dotproduct_common_logit_causality_anchor.md`
- Runner: `Projects/from-attention-to-search/main/experiments/A05_04_02_round2_dotproduct_common_logit/scripts/run_round2_dotproduct_common_logit.py`
- Round-1 dot-product baseline: `Projects/from-attention-to-search/main/experiments/A05_04_01_round1_slot_specialization_protocol_dot_product/results/round1_slot_specialization_dot_product_4gpu_20260609`

## Setup

Model: same tiny sparse top-1 MoE as Round 1, with positional embedding weights zeroed and frozen. Router metric is dot product only: `score_e(h) = w_e^T h`. Routing remains sparse top-1; no soft routing, no top-k, and no probability-weighted multi-expert mixture is introduced.

Dataset: same Round-1 moving-block data for the main run: `prefix filler + SLOT_s B_CONST Y_s + suffix filler`; the three-token block remains contiguous; loss is computed at the B position; routing metrics are computed at the B position except the explicitly marked P4 route-at-slot diagnostic.

Implementation adaptation: common cancellation estimates the current common vector from the current batch or evaluation set at the routed audit position, then subtracts `w_e^T c_t` only at that position in the router logits. Non-audited token positions keep their original dot-product top-1 router scores.

## Artifact Map

- Result directory in source repo: `main/experiments/A05_04_02_round2_dotproduct_common_logit/results/round2_dotproduct_common_logit_4gpu_20260609`
- Synced figure directory: `figures/round2_dotproduct_common_logit_4gpu_20260609`
- ACP job id: `pt-fplz4uf0`
- ACP runtime log: `Projects/from-attention-to-search/main/experiments/A05_04_02_round2_dotproduct_common_logit/logs/acp/round2_dotproduct_common_logit_4gpu_20260609_runtime_20260609_155611.log`
- ACP state: `SUCCEEDED`
- Raw metrics: `r2_all_trained_metrics_by_step.csv`, `r2_all_trained_route_confusion.csv`, `r2_all_trained_margin_decomposition.csv`
- P0: `r2_p0_decomposition_sanity.csv`, `r2_p0_summary.md`, figure `r2_p0_reconstruction_error.png`
- P1: `r2_p1_timing_by_seed.csv`, `r2_p1_common_predicts_final_table.md`, `r2_p1_summary.md`, figure `r2_p1_common_slot_margin_trajectory.png`
- P2: `r2_p2_alpha_sweep_metrics.csv`, `r2_p2_basin_threshold_summary.md`, figures `r2_p2_alpha_vs_final_slot_nmi.png`, `r2_p2_alpha_vs_margin_gap.png`
- P3: `r2_p3_common_cancel_metrics.csv`, `r2_p3_route_heatmaps.csv`, `r2_p3_summary.md`, figures `r2_p3_baseline_vs_cancel.png`, `r2_p3_route_heatmaps.png`
- P4: `r2_p4_source_ablation_metrics.csv`, `r2_p4_summary.md`, figure `r2_p4_source_comparison.png`

## R2-P0 Decomposition Sanity

Mean reconstruction error: 1.073e-07. Max reconstruction error: 5.849e-07. Common-dominant cells: 2238/2560.

![P0 reconstruction error](figures/round2_dotproduct_common_logit_4gpu_20260609/r2_p0_reconstruction_error.png)

## R2-P1 Timing Audit

Step 0: common margin 0.2993, slot margin 0.0523, common-predicts-final rate 0.844. Step 10: common margin 1.7087, slot margin 0.2993, common-predicts-final rate 0.969.

![P1 margin trajectory](figures/round2_dotproduct_common_logit_4gpu_20260609/r2_p1_common_slot_margin_trajectory.png)

## R2-P2 Slot-Init Basin Audit

- alpha=0: final slot NMI 0.080, max_load 0.969, initial slot-minus-common margin -0.2421
- alpha=0.05: final slot NMI 0.699, max_load 0.587, initial slot-minus-common margin 0.0056
- alpha=0.1: final slot NMI 0.856, max_load 0.466, initial slot-minus-common margin 0.1676
- alpha=0.2: final slot NMI 0.930, max_load 0.372, initial slot-minus-common margin 0.2743
- alpha=0.4: final slot NMI 0.930, max_load 0.372, initial slot-minus-common margin 0.3357
- alpha=0.6: final slot NMI 0.930, max_load 0.372, initial slot-minus-common margin 0.3398
- alpha=0.8: final slot NMI 0.930, max_load 0.372, initial slot-minus-common margin 0.3365
- alpha=1: final slot NMI 0.930, max_load 0.372, initial slot-minus-common margin 0.3352

![P2 alpha NMI](figures/round2_dotproduct_common_logit_4gpu_20260609/r2_p2_alpha_vs_final_slot_nmi.png)

![P2 alpha margin gap](figures/round2_dotproduct_common_logit_4gpu_20260609/r2_p2_alpha_vs_margin_gap.png)

## R2-P3 Common-Logit Cancellation

- R2P3_cancel_0_10: final slot NMI 0.896, max_load 0.434, accuracy 1.000
- R2P3_cancel_0_final: final slot NMI 0.963, max_load 0.309, accuracy 1.000
- baseline_random_init_dot_product: final slot NMI 0.080, max_load 0.969, accuracy 1.000

![P3 baseline vs cancellation](figures/round2_dotproduct_common_logit_4gpu_20260609/r2_p3_baseline_vs_cancel.png)

![P3 route heatmaps](figures/round2_dotproduct_common_logit_4gpu_20260609/r2_p3_route_heatmaps.png)

## R2-P4 Common Source Audit

- R2P4_original: common 2.3158, slot 0.5559, NMI 0.080, accuracy 1.000
- R2P4_route_at_slot: common 0.4681, slot 0.3750, NMI 0.504, accuracy 1.000
- R2P4_varied_b: common 0.9528, slot 0.4170, NMI 0.070, accuracy 1.000
- R2P4_varied_filler: common 1.9539, slot 0.5715, NMI 0.080, accuracy 1.000

![P4 source comparison](figures/round2_dotproduct_common_logit_4gpu_20260609/r2_p4_source_comparison.png)

## Final Answers

1. Dot-product decomposition exactness: mean reconstruction error 1.073e-07, max 5.849e-07.
2. Common-logit timing: step-0 common margin 0.2993; step-10 common margin 1.7087.
3. Early common ranking prediction: step-0 prediction rate 0.844; step-10 prediction rate 0.969.
4. Slot-init basin: threshold alpha 0.2; best alpha 0.2.
5. Common-cancellation causal test: supported by NMI-plus-accuracy improvement in this run.
6. Likely common source: fixed `B_CONST` / B-token identity is a likely major contributor; routing at the B position itself also contributes, because the route-at-slot diagnostic reduces the common margin; filler/template variation is not the main supported source. This is still a toy source audit, not a full source identification.
7. Next scientific decision: Design a minimal anti-common or anti-lock-in router that preserves sparse top-1 while making slot-specialization reachable from random initialization.

## Claim Boundary

Do not claim common source is identified unless P4 shows a source ablation with preserved task validity. Do not claim common-logit is causal unless P3 improves slot NMI or purity while preserving accuracy. Do not treat load balance as feature specialization. Do not claim expert computation is causally slot-specialized or transfer this result to real language models.
