---
experiment_id: A15_00_E01_actual_router_input_band_response
status: completed_strict_h1_fail_typed_result
completed: 2026-07-30
primary_anchor: A15_00_covariance_head_gate_alignment
companion_cn: summary_cn.md
---

# Summary: Actual-Router-Input Band Access And Training Allocation

Primary anchor: [A15_00](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/15_00_covariance_head_gate_alignment_anchor.md)  
Protocol: [approved Protocol](protocol.md)  
Detailed record: [detailed.md](detailed.md)

## Conclusion

**Q1 endpoint answer — pass.** At 30k, 40k, and 80k, both trained linear-Gate
lineages have much larger equal-energy gain on the covariance head of the
representation actually consumed by the Gate. At 40k/80k, the median
per-layer head:middle gain ratios are 5.41/6.36 for LB and 4.03/4.27 for
decommon; head:tail ratios are 19.98/25.36 and 14.61/17.15. The corresponding
log contrasts are far above the singular-value-preserving Haar q95 of about
0.04, with paired calibration-basis intervals above zero. The result is not an
input-energy-only artifact.

**Middle and tail are visible, but weaker.** Their $G$, realized response,
route-flip, and native-margin effects are nonzero. At 80k, median coarse
head/middle/tail route-flip fractions are 0.741/0.126/0.018 for LB and
0.645/0.089/0.013 for decommon. E01 therefore supports “weaker current access
and use,” not “the linear Gate cannot see middle/tail.”

**Strict persistent-training hypothesis — fail, with a typed result.** Both
saved net Gate displacements are themselves head-oriented and exceed their
matched nulls. However, after holding the representation basis fixed, they do
not consistently strengthen the already head-biased Gate. The head:middle
contrast decreases precisely in both intervals and both lineages. Head:tail
strengthens in LB, decreases in decommon from 30k to 40k, and is unresolved for
decommon from 40k to 80k. Thus 30k--80k supports **head-aligned endpoints plus
head-oriented net displacements without persistent fixed-basis
strengthening**. It does not establish when the alignment arose, a persistent
per-step gradient tendency, or a covariance-caused training mechanism.

The distinction is important: $\mathbf B^{update}$ asks where the squared
norm of the net update lies if that update were a Gate by itself;
$\Delta_W\mathbf B$ asks whether adding it to the existing Gate increases the
endpoint contrast. A positive $\mathbf B^{update}$ can dilute an even more
head-biased existing Gate, and signed cross terms with $W$ also matter.

## Primary Evidence

### Endpoint equal-energy access

| Lineage | Step | $B_{H:M}$ (95% basis interval) | $G_H/G_M$ | $B_{H:T}$ (95% basis interval) | $G_H/G_T$ | Haar q95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LB | 30k | 1.683 [1.634, 1.766] | 5.38 | 2.976 [2.931, 2.998] | 19.60 | 0.039 / 0.038 |
| LB | 40k | 1.689 [1.605, 1.719] | 5.41 | 2.995 [2.911, 2.986] | 19.98 | 0.038 / 0.038 |
| LB | 80k | 1.850 [1.720, 1.837] | 6.36 | 3.233 [3.149, 3.225] | 25.36 | 0.042 / 0.039 |
| decommon | 30k | 1.463 [1.414, 1.505] | 4.32 | 2.811 [2.725, 2.828] | 16.63 | 0.039 / 0.034 |
| decommon | 40k | 1.394 [1.340, 1.445] | 4.03 | 2.682 [2.633, 2.692] | 14.61 | 0.037 / 0.040 |
| decommon | 80k | 1.452 [1.375, 1.491] | 4.27 | 2.842 [2.784, 2.840] | 17.15 | 0.039 / 0.041 |

The percentile basis intervals can be slightly bootstrap-biased relative to
the full-sample point, but every endpoint conclusion is separated from both
zero and its matched orientation null. Full layerwise values are retained in
[endpoint_contrasts.csv](tables/endpoint_contrasts.csv).

### Saved-interval decomposition

| Lineage | Interval | Contrast | $B^{update}$ | $\Delta_WB$ (95% interval) | $\Delta_UB$ (95% interval) | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| LB | 30k→40k | H:M | 0.990 | -0.036 [-0.035, -0.028] | 0.019 [-0.012, 0.024] | update head-oriented; endpoint contrast diluted |
| LB | 30k→40k | H:T | 2.630 | 0.080 [0.074, 0.081] | -0.098 [-0.106, -0.083] | fixed-basis strengthening |
| LB | 40k→80k | H:M | 0.974 | -0.054 [-0.055, -0.042] | 0.119 [0.081, 0.126] | $W$ dilutes; $U$ raises endpoint contrast |
| LB | 40k→80k | H:T | 2.814 | 0.162 [0.155, 0.164] | -0.108 [-0.124, -0.092] | fixed-basis strengthening |
| decommon | 30k→40k | H:M | 0.293 | -0.067 [-0.068, -0.060] | 0.007 [-0.020, 0.012] | fixed-basis dilution |
| decommon | 30k→40k | H:T | 1.511 | -0.015 [-0.019, -0.011] | -0.105 [-0.116, -0.087] | fixed-basis dilution plus drift |
| decommon | 40k→80k | H:M | 0.410 | -0.061 [-0.066, -0.050] | 0.088 [0.071, 0.113] | $W$ dilutes; $U$ raises endpoint contrast |
| decommon | 40k→80k | H:T | 1.570 | 0.008 [-0.001, 0.013] | -0.100 [-0.109, -0.062] | fixed-basis effect insufficient |

Every $B^{update}$ exceeds its matched Haar q95 (0.039--0.056) with a positive
basis interval. The strict failure comes from the separately registered
$\Delta_WB$ condition, not from an absence of head orientation in the net
displacements.

## Guards And Rivals

- All six checkpoints and coordinate signatures passed provenance checks.
- The direct Gate pre-input replayed native logits with relative Frobenius
  error at most $10^{-5}$ and top-1 agreement 1.0. Replacing it with expert
  input $h$ produced only about 0.49--0.51 median top-1 agreement.
- All 12 layers at all six endpoints passed coarse half-split stability. The
  observed random-overlap q95 values are 0.086, 0.335, and 0.585 for dimensions
  64, 256, and 448; observed coarse overlaps are much larger.
- The Haar construction preserved nonzero Gate singular values with maximum
  relative error $1.9\times10^{-6}$.
- F1 is the strongest median fine band at every endpoint and exceeds the
  simultaneous Haar envelope; no hidden fine-band peak reverses the coarse
  result.
- Same-layer actual-input head contrasts are much stronger than next-layer or
  expert-input-basis controls. Basis orthogonality and band-energy
  reconstruction passed; within-group response cross terms are reported.

## Claim Boundary And Next Decision

E01 can claim actual-input equal-energy access, realized response, native
route dependence, and net allocation across the two saved intervals. It
cannot claim that head access is useful, that middle/tail lacks functional
information, that covariance caused the alignment, or that spectral routing
improves loss/FLOP.

**Next mainline decision:** do not start spectral joint training from Q1.
First apply the Q2 admission gate: on independent tokens, spectral features
must predict one-step joint-training compatibility beyond native linear score
and beat matched random-dimensional and wrong-layer controls.

If a separate small-model online run is approved, its sole purpose should be
to locate the origin and maintenance of head alignment. It should densely log
raw Gate gradients, optimizer-preconditioned updates, fixed-probe covariance
bases, signed $W$--$\Delta W$ band cross terms, margins, flips, and loads from
initialization. That run tests dynamics; it does not test functional utility.

