---
experiment_id: A15_02_01_E01_cross_update_compatibility_gate
status: completed_fail
verdict: fail
completed: 2026-07-30
protocol: protocol.md
summary: summary.md
---

# Detailed: A15_02_01_E01 Cross-Update Compatibility Gate

Primary anchor: [A15_02_01 compatibility gate](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_02_middle_tail_functional_resolution/subanchors/15_02_01_cross_update_compatibility_gate_anchor.md)  
Protocol: [approved Protocol](protocol.md)  
Summary: [summary.md](summary.md)

## 0. Quick Recap

**Purpose:** test whether fixed middle, covariance long-tail, or non-head
coordinates add held-out prediction of one-step same-expert compatibility
beyond native Router scores and registered nuisance controls.

**Hypothesis:** at least one band is positive, beats equal-dimensional random
and wrong-layer controls, and reproduces across LB and decommon before transfer
or matched training.

**Conclusion:** operational Pass and scientific Fail. Compatibility was
measurable, but the validation candidate set was empty; the registered stop
rule correctly blocked final functional evaluation, transfer, and training.

**Evidence:** all 3,072 fit/validation A/B pairs completed, every self-update
reduced its own loss, expert restore was exact, half-step rankings were stable,
and no fixed M/T/N band passed the full cross-lineage admission rule.

## 1. Decision and outcome

The experiment asked whether actual-Router-input middle, covariance long-tail,
or their union incrementally predicts bidirectional one-step same-expert
compatibility after controlling native Router scores, margin, expert identity,
load, token count, difficulty, norm, document, and batch, and whether any such
signal beats equal-dimensional random and wrong-layer bases.

The operational audit passed. The scientific verdict is **Fail** because the
Validation candidate set was empty. The registered stop rule was applied before
Final test, 40k replication, four-layer transfer, or E02 training.

## 2. Terminology / Metric-Purpose Ledger

| Metric or guard | Concrete computation / unit | Why it was measured | What it answers | What it cannot answer |
| --- | --- | --- | --- | --- |
| Actual-input replay | Offline linear Gate logits and native hooked logits; relative Frobenius error and top-1 agreement | Verify that the analyzed tensor is the deployed Gate input | Whether all downstream spectral coordinates refer to the correct object | Functional value |
| Residual-neighbor novelty | $1-|\mathrm{kNN}_{band}\cap\mathrm{kNN}_{native}|/32$ after Fit-only linear residualization | Separate “different partition” from “useful partition” | Whether a band changes held-out local neighborhoods beyond native scores | Co-training benefit |
| Bidirectional compatibility $C$ | Negative mean of A-to-B and B-to-A cross-loss changes, nat/token | Give “train together” a direct local loss definition | Whether two routed groups locally help or conflict when updating one expert | Long-run optimizer dynamics |
| Self-loss change | Loss of the updating group after its own step, nat/token | Validate update sign and local scale | Whether the probing step is a descent step | Cross-group usefulness |
| Half-step Spearman | Pair ranking correlation between $\eta$ and $\eta/2$ | Reject a step-size-only artifact | Whether compatibility ordering is locally stable | Real AdamW behavior |
| Gradient cosine | Cosine between exact expert gradients | Check expected first-order source of $C$ and target dynamic range | Whether local gradient alignment tracks compatibility | Band specificity |
| Incremental $\Delta R^2$ | Validation $R^2$ with native controls plus two band features minus native-only $R^2$ | Primary functional admission metric | Whether the band adds low-capacity held-out compatibility prediction | Matched-training improvement |
| Full-space random q95 | q95 of 256 equal-dimensional Haar-orientation $\Delta R^2$ values | Reject generic high-dimensional geometry | Whether the registered covariance rank span is better than arbitrary directions | Causality |
| Non-head random q95 | q95 of 256 equal-dimensional orientations inside the 704-dimensional non-head span | Reject “any non-head directions work” for M/T | Whether the exact M/T rank location is special within non-head | Applicability to N, which fills non-head |
| Wrong-layer delta | Same rank range from the preregistered source layer | Reject arbitrary layer geometry | Whether target-layer spectral coordinates are more predictive | Absence of cross-layer shared directions |
| Nuisance comparison | Full registered controls versus core native-score controls | Detect norm, loss, gradient magnitude, position, load, or band-energy shortcuts | Whether a positive result survives observed nuisance controls | All possible confounding |
| Document bootstrap | 2,000 document-block resamples of frozen Validation predictions | Quantify document sampling stability | Precision under new DCLM documents | New independent model seeds |

## 3. Frozen objects and provenance

### Models

- 12-layer H768, 8-expert, top-1 LB lineage, checkpoint 80k.
- 12-layer H768, 8-expert, top-1 decommon lineage, checkpoint 80k.
- Functional layers were preregistered as 1, 6, and 12.
- The 40k endpoints and four-layer checkpoint were preflighted but were not
  evaluated after the Validation stop rule fired.

### Data

- 512 new DCLM held-out documents, 1,024 tokens each.
- Split by complete document: 64 Operationalization, 192 Fit, 128 Validation,
  128 locked Final.
- Token tensor SHA-256:
  `da6942431c0fa4c17c2e0ec3e4611ec63d19ffc196002eed66ac71f55b991592`.
- The documents are disjoint from the Q1 evaluation set. No Q2 document was
  used to estimate a spectral basis.

### Spectral bases

- Q1's independently calibrated actual-Gate-input means and covariance bases
  were reused for the 12-layer endpoints.
- $H$: ranks 1--64; $M$: 65--320; $T$: 321--768; $N=M\cup T$: 65--768.
- Every true or control subspace contributed only two pair features: cosine and
  squared distance of group means of per-token unit projected coordinates.
- True band energy entered nuisance controls, preventing direction features
  from winning only through input magnitude.

## 4. Result-before-outcome amendments

Two operational amendments were frozen before any compatibility-versus-band
outcome was inspected.

### Native-load-weighted pair allocation

Route-only feasibility inspection showed that decommon nearly starved some
experts, so an equal number of pairs per expert was not mathematically feasible.
The group contract remained 32 tokens from one document, and A/B remained
document-, batch-, and token-independent. Up to 256 pairs per model-layer-split
were allocated in proportion to native route mass, capped by available no-reuse
pairs. Expert identity and load stayed in the controls. Every scientific cell
reached 256 pairs, above the registered minimum of 192.

This changes the estimand to the native-routed token population and prevents a
nearly unused expert from being treated as representative. It cannot answer
compatibility inside an expert that the native Router almost never uses.

### Float32 local-loss readout

The native checkpoint, winners, and all routing weights remained bfloat16 and
frozen. For the finite one-step loss probe only, checkpoint parameter values
were losslessly promoted to float32 and autocast was disabled. A pre-outcome
smoke showed that bfloat16 output quantization was coarser than the intended
local loss changes and could flip self-loss signs. Float32 therefore measures
the smooth local loss geometry around the same parameter values and routes. It
does not measure a bfloat16 deployment effect.

## 5. A/B construction and exact target

Each A or B group contains exactly 32 loss-bearing tokens from one document,
all natively routed to the same target expert. A and B use different documents,
logical batches, and tokens. Matching used only native controls.

For target expert parameters $\theta_{\ell,e}$:

$$
\Delta_{A\rightarrow B}
=L_B(\theta_{\ell,e}-\eta\nabla L_A)-L_B(\theta_{\ell,e}),
$$

$$
C_e(A,B)=-\frac12(\Delta_{A\rightarrow B}+\Delta_{B\rightarrow A}).
$$

$C>0$ means the two local updates help on average; $C<0$ means conflict. All
other parameters and all MoE routes stayed frozen, and the expert snapshot was
restored exactly between directions.

The Operationalization split selected one local step per model and layer from
a fixed grid. Six of six model-layer cells passed self-loss, relative-step-size,
half-step ordering, and exact-restore guards.

## 6. Measurement validity

All twelve Fit/Validation compatibility cells contained 256 pairs.

| Model | Layer | Fit half-step Spearman | Validation half-step Spearman | Self-loss pass | Exact restore |
| --- | ---: | ---: | ---: | ---: | ---: |
| LB | 1 | 0.952 | 0.957 | 1.000 | yes |
| LB | 6 | 0.992 | 0.991 | 1.000 | yes |
| LB | 12 | 0.995 | 0.997 | 1.000 | yes |
| decommon | 1 | 0.877 | 0.875 | 1.000 | yes |
| decommon | 6 | 0.967 | 0.975 | 1.000 | yes |
| decommon | 12 | 0.994 | 0.993 | 1.000 | yes |

Compatibility standard deviation ranged from $2.33\times10^{-6}$ to
$3.10\times10^{-5}$ nat/token. Across cells, compatibility correlated 0.71--0.96
with exact expert-gradient cosine. This verifies usable target variation and the
expected local mechanism; it does not support any spectral candidate.

The full guard table is [measurement_guards.csv](tables/measurement_guards.csv).

## 7. Q2-A static resolution

On locked Final documents, the true M/T/N residual-neighbor novelty across the
registered layers and lineages ranged from 0.732 to 0.902. A fixed
equal-dimensional full-space random reference ranged from 0.714 to 0.877.

Thus the bands do produce neighborhoods different from native logits. Random
directions also produce very different neighborhoods, so novelty alone is not a
functional or covariance-rank-specific certificate. Long-tail was modestly more
novel than the fixed random reference in most registered cells, but this result
had no admission authority.

All-layer and condition-level values are in
[static_novelty.csv](tables/static_novelty.csv).

## 8. Q2-B Validation functional gate

The baseline was a standardized low-capacity ridge on symmetric native
logit/margin/expert/load features plus registered NLL, norm, position, gradient
magnitude, outlier, document-aggregate, and band-energy controls. Each band
added exactly two features, independent of its dimension. Ridge regularization
was selected on Validation exactly as preregistered; the same search was applied
to every random orientation.

### Model-level results

Three-layer median incremental $\Delta R^2$ and direction controls:

| Band | Model | True median | Full random q95 | Non-head random q95 | Wrong-layer median | Eligible |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| M | LB | -0.0000735 | +0.0000725 | +0.0001000 | -0.0000913 | no |
| M | decommon | -0.0000430 | +0.0000536 | +0.0000011 | +0.0000177 | no |
| T | LB | +0.0002237 | -0.0000233 | +0.0000239 | -0.0000442 | yes in LB only |
| T | decommon | -0.0000520 | +0.0000244 | -0.0000088 | -0.0000703 | no |
| N | LB | -0.0000590 | -0.0000554 | n/a | -0.0000624 | no |
| N | decommon | -0.0000429 | -0.0000198 | n/a | -0.0000401 | no |

The selected-candidate set is empty because a candidate had to pass separately
in both lineages. M and N had non-positive model medians in both. T passed the
point gates only in LB and failed positivity and random controls in decommon.

Document-bootstrap intervals also included zero for all model-band summaries;
there is no hidden stable positive candidate behind the point gate. The exact
selection ledger is [validation_candidate_gate.csv](tables/validation_candidate_gate.csv),
and all layer cells are in
[validation_functional_cells.csv](tables/validation_functional_cells.csv).

![Static novelty versus functional gate](figures/static_vs_functional_gate.png)

## 9. Registered stopping decision

Protocol section 11.1 states that if no candidate qualifies on 12-layer 80k
Validation, the experiment terminates and must not inspect compatibility Final,
replicate at 40k, run the four-layer transfer gate, or start E02. That condition
was met.

Consequently:

- Final compatibility documents remained unopened for functional selection or
  evaluation; they were used only for the preregistered static Q2-A diagnostic.
- 40k compatibility was not run.
- Four-layer compatibility transfer was not run.
- No 8x5090 job was submitted. The conditional E02 authorization never became
  active because there is no legal $S^*$ treatment.

This is a completed Fail, not an incomplete run.

## 10. Figure contracts and visual audit

### `static_vs_functional_gate.png`

- Question: does a different static partition also provide functional
  compatibility prediction?
- Left metric: median residual-neighbor novelty over layers 1/6/12, percent;
  true covariance band versus one fixed equal-dimensional random reference.
- Right metric: model-level median Validation $\Delta R^2$ in units of
  $10^{-4}$; true band, 256-orientation random q95, and wrong-layer median.
- Data: locked static Final documents on the left; compatibility Validation on
  the right.
- Allowed conclusion: static novelty is high, while no candidate passes both
  lineages' functional gates.
- Limitation: the left random point is a fixed diagnostic reference, not its
  full distribution; the right panel is Validation candidate selection, not a
  Final estimate.
- Render audit: labels, zero line, legend, row order, and units were visually
  inspected at full resolution on 2026-07-30.

### `validation_candidate_gate.png`

- Question: how uncertain are the model-level candidate point estimates?
- Metric: model-level median Validation $\Delta R^2$ with 2,000 document-block
  bootstrap intervals, plus random q95 and wrong-layer points.
- Allowed conclusion: intervals do not support a stable positive candidate.
- Limitation: Validation is selection data; these intervals are evidence for
  the registered stop decision, not Final-test confidence intervals.
- Render audit: labels, error-bar direction, units, zero line, and legend were
  visually inspected at full resolution on 2026-07-30.

## 11. Artifact map

### Curated record

- [Protocol](protocol.md) and [Chinese protocol](protocol_cn.md)
- [English summary](summary.md) and [Chinese summary](summary_cn.md)
- [Static/function figure](figures/static_vs_functional_gate.png)
- [Candidate-gate figure](figures/validation_candidate_gate.png)
- [Static novelty table](tables/static_novelty.csv)
- [Functional cell table](tables/validation_functional_cells.csv)
- [Candidate selection table](tables/validation_candidate_gate.csv)
- [Measurement guard table](tables/measurement_guards.csv)

### Worker/raw evidence

Worker root:
`Projects/from-attention-to-search/XingyuD/MoE_Routing_Experiments/active/a15_02_functional_resolution/`

- `runs/a15_02_01_e01/preflight.json`
- `runs/a15_02_01_e01/data/data_manifest.json`
- `runs/a15_02_01_e01/features/{lb,decommon}_80000/`
- `runs/a15_02_01_e01/analysis/validation_candidate_gate.json`
- `runs/a15_02_01_e01/analysis/completion_audit.json`
- `core.py`, `e01.py`, `analyze.py`, `plot_results.py`, and unit tests

The completion audit records the exact SHA-256 hashes of the preflight, data,
candidate gate, and analysis code. Its operational status is `pass`, its
scientific verdict is `fail`, and it contains no failed guard.

## 12. Claim boundary and next decision

The result rejects only the preregistered low-capacity claim for fixed
covariance-rank M/T/N geometry at the registered checkpoints and layers. It does
not establish that non-head information has no nonlinear, semantic, or
long-horizon value, or that a function-aligned learned subspace would fail.

The next decision is whether to close fixed covariance bands as direct dispatch
coordinates and require a new anchor to define a function-aligned subspace from
expert gradients or cross-update residuals before spending matched-training
compute.
