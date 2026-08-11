# Summary: A15_08_E04 Strict-Conformance Repair

Primary Anchor: [A15_08 Target-Conditioned Layer Innovation](../../../../problem_anchors/15_spectral_representation_and_functional_routing/15_08_target_conditioned_layer_innovation/15_08_target_conditioned_layer_innovation_anchor.md)  
Frozen Protocol: [A15_08_E04_v1](protocol.md), SHA-256 `5072d76d9a44877e270b910424ea255ffee347dc17097640c88e20200d82a27d`  
Detailed record: [detailed.md](detailed.md)  
Result status / eligibility: `ELIGIBLE` after S0--S6; the [record-completeness audit](evidence/record_completeness_audit.json) confirms 59 complete artifacts across all 39 required families.  
Knowledge wording: `AI_DRAFT_AWAITING_HUMAN_CONFIRMATION`; eligibility, measurements, and the registered verdict are evidence-fixed.

An independent post-run [cold-read audit](evidence/independent_cold_read_audit.json)
then rechecked 47/47 provenance and computation items, including all 59
manifest hashes and a reconstruction from record arrays through 2,000 paired
bootstrap draws. It found no fatal discrepancy. This reader audit verifies the
frozen record; it is not retroactively inserted into the S0--S5 manifest.

## Result Snapshot

**Verdict:** `H2_PASS_H1_FAIL`. The machine field `h1=FAIL` means only that
the registered H1' rank-two sufficiency clause failed; it does not reject the
meeting's global hypothesis that representations may be low rank.

**What we established:** on 128 fresh complex confirmation episodes, the full
residualized layer-25 attention update improved terminal-code cross-entropy by
$G_{true}=0.767207$ nats/example, with paired episode-bootstrap 95% interval
$[0.751296,0.785146]$. The same-dimensional new-state comparison had the same
sign, and the gain exceeded the q95 of all 64 balanced target-independent
mismatch controls. Therefore the registered local H2 added-accessibility gate
passes.

The two frozen target-conditioned directions achieved
$G_2=0.255052$ nats/example, only 33.24% of the full point gain. Their
registered 80%-retention contrast was strictly negative,
$D_{80}=-0.358713$ with 95% interval
$[-0.369760,-0.348675]$. H1' therefore fails at rank two even though those two
directions beat both equal-rank null families.

**Why E04, rather than E03, carries the claim:** the final E03 cold read found
that E03 did not instantiate every registered namespaced ID and omitted exact
extraction-command receipts, complete simple episode/bootstrap arrays, and a
full registered-family manifest. E03 is therefore `INELIGIBLE_GUARD`; its
numbers are diagnostic only. E04 is a conformance-only fresh repair: it keeps
the E02-frozen scientific object, readouts, directions, rank, thresholds,
control families, and decision rules unchanged while repairing those guards.

**What we do next:** freeze one fresh-data rank-ladder design over
$r\in\{3,4,5,6,7\}$ on the same $R_U$ object. A Router experiment is not yet
admitted.

## Decision-Bearing Setup

**Question:** does the full residual update $R_U$ add fresh held-out
terminal-code accessibility beyond the old state and target-independent
same-budget corrections; if so, do the two frozen conditional-innovation
directions retain at least 80% of that gain?

**Treatment and controls:** the treatments are the frozen full-$R_U$ and
rank-two-$R_U$ corrections added to the old-state readout. Comparators are the
old-state readout, the same-dimensional $Z$ readout, 64 balanced-mismatch
corrections, 64 random rank-two corrections, and 64 TRAIN-label-permutation
rank-two corrections.

**Changed variable:** only the frozen feature supplied beyond $X$.

**Held fixed:** Qwen3-8B, final Answer token, normalized layer-25 attention
transition, E02 TRAIN/DEVELOPMENT-selected readouts, residualizers and
directions, rank two, 80% threshold, control budgets, and the paired-bootstrap
rule.

**Primary metric and unit:**
$G_{true}=CE(f_X(X))-CE(f_X(X)+g_{full}(R_U))$ in nats/example. Positive values
mean that the frozen full residual-update correction lowers fresh-confirmation
cross-entropy. The complete eight-target episode is the independent unit; all
intervals use 2,000 paired episode-bootstrap draws with seed 8281.

**Known limitation:** one synthetic two-hop target, one answer position, one
normalized attention transition, and frozen linear readout families.

## Verdict Basis

| Registered rule | Observed estimate and paired 95% interval (nats/example) | Guard / eligibility | Verdict | Evidence source |
| --- | ---: | --- | --- | --- |
| H2: lower bound of $G_{true}>0$ | $0.767207\ [0.751296,0.785146]$ | 128 fresh complex episodes | Pass | [metrics](tables/confirmation_metrics.csv), [complex arrays](evidence/complex_confirmation_arrays.npz) |
| H2: lower bound of $G_{state}>0$ | $0.754508\ [0.738226,0.773069]$ | same-dimensional state comparison | Pass | [metrics](tables/confirmation_metrics.csv) |
| H2: lower bound of $T_{cap}>0$ | $0.766839\ [0.750643,0.784307]$ | 64/64 balanced banks valid, distinct, zero-MI, and unreused | **H2 Pass** | [metrics](tables/confirmation_metrics.csv), [bank audit](evidence/materialized_confirmation_bank_audit.json) |
| H1': lower bound of $D_{80}>0$ | $-0.358713\ [-0.369760,-0.348675]$ | evaluated only after H2 Pass | **H1' Fail** because the upper bound is non-positive | [metrics](tables/confirmation_metrics.csv), [complex arrays](evidence/complex_confirmation_arrays.npz) |
| H1': lower bound of $T_{random}>0$ | $0.034265\ [0.024293,0.040979]$ | 64 frozen equal-rank random controls | Pass, but cannot rescue $D_{80}$ | [metrics](tables/confirmation_metrics.csv) |
| H1': lower bound of $T_{perm}>0$ | $0.234832\ [0.224277,0.244831]$ | 64 frozen TRAIN-label-permutation controls | Pass, but cannot rescue $D_{80}$ | [metrics](tables/confirmation_metrics.csv) |

Here $G_{state}=CE(f_X(X))-CE(f_Z(Z))$; $G_2$ is the gain from the two frozen
$R_U$ directions; $T_{cap}$ subtracts the within-draw higher-method q95 of the
64 balanced-mismatch gains from $G_{true}$;
$D_{80}=G_2-0.8G_{true}$; and $T_{random}$ and $T_{perm}$ subtract the
within-draw higher-method q95 of their respective 64-control families from
$G_2$.

## Key Evidence

| Quantity | Estimate | Paired 95% interval | Direct reading |
| --- | ---: | ---: | --- |
| full residual-update gain $G_{true}$ | 0.767207 | [0.751296, 0.785146] | positive added-accessibility effect |
| same-dimensional state gain $G_{state}$ | 0.754508 | [0.738226, 0.773069] | positive H2 state guard |
| gain beyond mismatch q95 $T_{cap}$ | 0.766839 | [0.750643, 0.784307] | not reproduced by the registered target-independent correction family |
| two-direction gain $G_2$ | 0.255052 | [0.244938, 0.264427] | 33.24% of the full point gain |
| 80%-retention contrast $D_{80}$ | -0.358713 | [-0.369760, -0.348675] | rank-two sufficiency fails |
| gain beyond random q95 $T_{random}$ | 0.034265 | [0.024293, 0.040979] | top-two directions are non-random under this null |
| gain beyond permutation q95 $T_{perm}$ | 0.234832 | [0.224277, 0.244831] | target conditioning is non-null under this control |

The corresponding complex-condition mean cross-entropies were 2.066253 for
the old-state readout, 1.311745 for the new-state readout, 1.299046 for old
plus full $R_U$, and 1.811201 for old plus the top-two $R_U$ coordinates.

## Central Figure

![E04 fresh-confirmation gates](figures/figure_e04_fresh_confirmation_gates.png)

The figure uses [confirmation_metrics.csv](tables/confirmation_metrics.csv)
and [complex_confirmation_arrays.npz](evidence/complex_confirmation_arrays.npz).
Its unit is nats/example; each point is the equal-weight mean over 128 complete
episodes, and each bar is a percentile 95% interval from 2,000 paired draws.
It visualizes only the registered local H2 Pass and H1' Fail. It cannot prove
adjacent-block novelty, whole-representation low rank, native use, or Router
gain. The complete figure contract is
[recorded here](evidence/plot_contracts.json).

## Claim Boundary

**Can claim:** for the controlled two-hop target, frozen Qwen3-8B, final Answer
token, and the named normalized layer-25 attention transition, the registered
full-$R_U$ linear correction contains terminal-code accessibility beyond the
old-state readout and the registered target-independent correction family. The
two frozen conditional-innovation directions are informative but insufficient
to retain 80% of that gain.

**Cannot claim:** adjacent-block novelty, that a deeper layer retains all
shallower information, whole-representation low rank, Shannon-information
creation, factual knowledge storage, nonlinear uniqueness, native model use,
natural-language generality, expert utility, or Router gain.

## Next Decision

**Decision:** freeze one fresh-data rank-ladder design over
$r\in\{3,4,5,6,7\}$ on the same $R_U$ object, with equal-rank random and
TRAIN-label-permutation control budgets and the unchanged 80% held-out-
retention gate.

**Completion criterion:** return the smallest rank whose $D_{80}$,
$T_{random}$, and $T_{perm}$ lower 95% bounds are all positive; if none passes,
record "no sufficient linear compression through rank seven" without changing
the threshold.

**Resume action:** draft only that fresh-data rank-ladder Protocol from the
frozen E04 object and gates. Do not change the task, reopen spectral-band
selection, or add a Router experiment.
