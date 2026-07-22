# Protocol: A14_E07 Reachable-Space Accounting

## 0. Approval Snapshot

**Approval status:** Approved by the researcher on 2026-07-19 for implementation, smoke guards, and the full registered run.

**Purpose:** Audit an implementation of the exact dimension decomposition that separates role-map compression, within-layer direction reuse, cross-layer direction reuse, and finite-sample activation.

**Primary package:** [A14 probabilistic sparse-activation tree handoff](../../README.md)

**Anchor decision question:** Which quantities control the global representation dimension when low-dimensional leaves are propagated through finite-depth shared role maps?

**Anchor physical prior tested:** A tree does not create global low rank by itself. Slow dimension growth must come from role-map compression or reuse of directions within and across layers.

**Anchor core model terms tested:** role-image dimension, within-layer overlap, overlap with cumulative history, and activation of reachable directions by finite samples.

**Anchor falsifier:** Any nonzero discrepancy between the registered exact recurrence and an independent dimension measurement in a valid, numerically separated case.

**Experiment role:** mathematical operationalization and implementation audit.

**Primary metric:** maximum absolute dimension mismatch between the formula prediction and the independent linear-algebra measurement.

**Claim boundary:** This run can establish only that the synthetic implementation realizes the exact accounting and separates its mechanisms. It cannot establish that language, pretrained models, or MoE systems satisfy those mechanisms.

**Minimal setup:** binary, ternary, and quaternary finite trees in a 512-dimensional ambient space; exact coordinate subspaces provide the oracle, and five random orthogonal rotations test basis invariance.

**Basic configuration:** float64; rotation seeds 0--4; leaf dimensions 4 or 8; registered depths remain below ambient saturation; no noise, training, text, or pretrained model.

**Conditions to run:** binary local accounting; general b-ary sequential accounting; five multi-layer growth regimes; full and deficient activation.

**Pass:** primary metric is zero in every registered case; all known-good cases pass; both deliberately wrong accounting rules fail their negative controls; full activation reaches the reachable dimension and deficient activation has the registered strict gap.

**Fail:** any valid case has nonzero exact mismatch; a negative control does not reject its wrong rule; or full activation fails to attain the reachable dimension.

**Insufficient:** invalid registered geometry, ambiguous numerical-rank gap, incomplete run, ambient saturation, or leakage between prediction and measurement paths.

**Cannot claim:** that trees are inherently low rank; finite depth is sufficient; language exhibits the registered overlaps; Transformers implement shared linear role maps; or low rank implies multiple experts, sparse routing, or optimal Top-k activation.

**Approval decision:** Approved. No expansion to real text, pretrained models, or new MoE experiments is authorized by this protocol.

## 1. Terminology / Definitions

| Term | Plain meaning | Concrete object / computation | Unit | Why it matters | Cannot prove |
| --- | --- | --- | --- | --- | --- |
| Reachable space | All directions allowed by the leaf space and role maps | Sum role-image spaces layer by layer | dimensions | Mechanism-level upper bound on observed rank | Finite data activate every direction |
| Role compression | A role map retains fewer directions than it receives | Dimension of one role-image space | dimensions | Separates information loss from overlap | The retained directions are linguistic |
| Within-layer reuse | Different roles map into shared directions | Intersection or sequential reuse inside a layer | dimensions | Prevents role dimensions from simply adding | Reuse of historical directions |
| Cross-layer reuse | A new layer falls into directions already used by history | Intersection of the new layer with cumulative history | dimensions | Directly controls global new directions | Reuse is caused by linguistic hierarchy |
| Full activation | Observed child tuples generate every reachable output direction | Observed parent column space equals reachable space | boolean | Separates mechanism capacity from finite-data rank | More samples alone guarantee activation |
| Maximum absolute dimension mismatch | Worst prediction-versus-measurement discrepancy | Maximum integer absolute difference across registered records | dimensions | Primary correctness metric | An experiment proves the theorem |

## 2. Anchor Alignment

**Decision question:** Should the strict A14 mathematical claim be stated as exact reachable-space growth controlled by role compression and direction reuse, with a common invariant space only as a strong corollary?

**Physical prior tested:** Roles may discard input directions, multiple roles may reuse the same output directions, and a new layer may reuse historical directions. These effects must be recorded separately.

**Core terms tested:** each role-image dimension; each role's sequential new contribution; the new layer's contribution beyond history; and the observed activated dimension.

**Falsifier:** Independent measurements fail to reproduce the exact accounting or fail to distinguish reachable capacity from observed rank.

**Claim boundary:** A14's model-general global shared-linear propagation claim remains closed. This audit refines only the conditional mathematics and its implementation.

## 3. Tested Hypotheses

**H1 — Binary exact accounting.** With rank-reducing role maps, the next-layer dimension is exactly the two role-image dimensions minus their shared dimension; global dimension additionally subtracts overlap with cumulative history.

**H2 — General b-ary exact accounting.** For three or more roles, sequentially measured new dimensions reproduce the independent measurement; pairwise-only inclusion--exclusion fails the registered three-lines-in-a-plane counterexample.

**H3 — Mechanism separability.** Role compression, within-layer reuse, and cross-layer reuse can produce equal layer rank while leaving different mechanism ledgers or global trajectories.

**H4 — Reachability is not observation.** Full activation attains the reachable dimension, whereas correlated or deficient child tuples can produce strictly lower observed rank.

## 4. Rival Explanations

1. Prediction and measurement accidentally share one implementation path.
2. Numerical rank thresholds create the result.
3. Ambient saturation makes growth appear slow.
4. Small or correlated samples, rather than mechanism geometry, reduce observed rank.

The protocol addresses these with an independent coordinate oracle, a separate SVD measurement path, rotation and spectral-gap guards, non-saturating dimensions, and full-versus-deficient activation.

## 5. Data / Model / Algorithm / Objective

### 5.1 Data-generating process

- Ambient dimension: 512.
- Build exact coordinate subspaces with registered dimensions and intersections.
- Apply one global random orthogonal rotation for each seed.
- Use no noise; E01 already audits noisy effective-rank propagation.

### 5.2 Role maps

Construct partial isometries from the current reachable basis into registered role-exclusive, role-shared, history-reused, and globally new coordinate blocks. A role may keep all or only a registered subset of the current directions.

### 5.3 Independent accounting paths

- **Prediction:** read only registered image and overlap dimensions and apply the binary or sequential b-ary recurrence.
- **Measurement:** concatenate realized image bases and estimate sums/intersections with float64 SVD; do not consume predicted dimensions.
- **Oracle:** retain exact coordinate-direction sets before rotation and use integer set unions as a third guard.

### 5.4 Activation

- **Full:** construct independent child tuples for every role and input basis direction.
- **Deficient:** force child tuples to be collinear or fixed-correlated so only part of the reachable output is observed.

There is no loss, optimizer, or training objective.

## 6. Conditions, Seeds, And Scales

| Item | Anchor clause tested | Model term / rival | Why needed | Evidence role | Pass | Fail | Insufficient | Output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 Binary local registry | General rank-reducing recurrence | compression and both overlaps | Independently vary three mechanisms | primary | mismatch 0 | any nonzero | invalid geometry | `binary_local.csv` |
| B2 General b-ary registry | Sequential role addition | higher-order dependence | Reject pairwise-only shortcut | primary | mismatch 0 | any nonzero | construction/order guard fails | `bary_local.csv` |
| B3 Multi-layer regimes | Slow-growth conditions | compression and reuse | Show distinct global trajectories | primary | all layers equal oracle | any mismatch | saturation | `growth_curves.csv` |
| B4 Full activation | Sufficient activation | reachability versus observation | Test attainability | primary guard | 100% equality | any gap | insufficient columns | `activation.csv` |
| B5 Deficient activation | Finite-sample boundary | sample correlation | Prevent capacity/observation conflation | primary guard | all registered strict gaps | missing gap | reachable dimension <= 1 | `activation.csv` |
| B6 Coordinate rotations | Basis invariance | numerical threshold artifact | Check five rotated realizations | secondary guard | 5/5 agree | any disagreement | ambiguous spectral gap | `rotation_audit.csv` |

### 6.1 Binary local registry

Current dimension is 12. Each role-image dimension is 0, 6, or 12. For every feasible pair, within-layer overlap is zero, approximately half, or maximal, and overlap with cumulative history is zero, approximately half, or complete. Infeasible targets are rejected before the run and do not enter the primary metric.

### 6.2 General b-ary registry

- Branching factors: 3 and 4.
- Current dimension: 8.
- Conditions: identical full images, disjoint full images, nested compression, partial new contribution, and fully redundant later roles.
- Negative control: three distinct lines in a plane have zero pairwise intersections but total dimension two.

### 6.3 Multi-layer regimes

| Regime | Role compression | Within-layer reuse | Cross-layer reuse | Expected trajectory |
| --- | --- | --- | --- | --- |
| G1 Worst expansion | none | none | none | branching-factor exponential bound before saturation |
| G2 Compression only | yes | none | none | fixed layer dimension, linearly increasing global dimension |
| G3 Within-layer reuse only | none | complete | none | fixed layer dimension, linearly increasing global dimension |
| G4 Within- and cross-layer reuse | none | complete | complete | fixed layer and global dimensions |
| G5 Bounded new directions | partial | partial | high | at most linear layer/global growth |

Non-saturating scales: binary `(r=4,L=5)` and `(r=8,L=4)`; ternary `(r in {4,8},L=3)`; quaternary `(r=4,L=3)` and `(r=8,L=2)`.

### 6.4 Seeds

Orthogonal-rotation seeds: 0, 1, 2, 3, 4. Seeds change coordinates only, not any registered dimension.

## 7. Primary Metric

**Definition:** maximum over records of the absolute difference between predicted and independently SVD-measured layer or cumulative dimension.

**Unit:** dimensions.

**Decision role:** Exact accounting admits no statistical tolerance; the metric must be zero.

**False-positive cost:** Shared code could make a wrong implementation self-consistent. Prediction, measurement, and coordinate-oracle paths therefore remain separate.

**False-negative cost:** SVD threshold noise could reject a correct construction. Only records passing the registered singular-value-gap guard enter the primary metric.

## 8. Secondary Metrics

Binary mismatch count; b-ary mismatch count; coordinate-oracle/SVD disagreement count; full-activation equality rate; deficient-activation strict-gap rate; rejection rate of two deliberately wrong estimators; and layer/global trajectories for G1--G5.

## 9. Known Good / Known Bad / Known Confusing

**Known good:** all-zero role maps; identical injective role images; disjoint injective role images; complete reuse of historical directions.

**Known bad:** an overlap-only estimator that omits role-image dimensions; pairwise-only inclusion--exclusion for the three-lines-in-a-plane case. Both must fail their registered negative control.

**Known confusing:** fixed rank at every layer with mutually new layer spaces; large reachability with rank-one correlated observations; shape-limited observations with fewer columns than reachable dimensions.

## 10. Stage-Level Profiling Plan

| Stage | Local question | Input evidence | Pass / fail / unclear | Debug artifact | Handoff |
| --- | --- | --- | --- | --- | --- |
| S0 Registry | Are targets feasible and non-saturating? | condition registry | all valid / invalid target / incomplete | `registry_audit.json` | valid cases only |
| S1 Construction | Do realized subspaces match exact coordinate targets? | coordinate sets | all match / any mismatch / missing guard | `construction_audit.csv` | rotate only after pass |
| S2 Measurement | Does SVD recover the coordinate oracle? | rotated bases | 5/5 / any mismatch / weak gap | `spectrum_guard.csv` | primary records only after pass |
| S3 Recurrence | Does prediction equal independent measurement? | registry and SVD records | max 0 / nonzero / leakage | `recurrence_audit.csv` | mechanism summary |
| S4 Activation | Do observations attain or miss reachability as registered? | child tuples and parent matrices | expected equality/gap / opposite / shape limited | `activation.csv` | capacity/observation separation |
| S5 Reporting | Are tables and figures consistent? | passed stage records | consistent / inconsistent / missing | `figures/`, `summary.json` | result audit |

## 11. Algorithm Specification

**Input:** registered condition, ambient dimension, leaf dimension, branching factor, depth, rotation seed.

**Numerical rule:** float64 adaptive SVD tolerance equals `100 * max(shape) * eps * largest_singular_value`. A primary record additionally requires at least a `1e6` ratio between the smallest retained and largest discarded singular values when both exist.

**Steps:** validate feasibility; allocate exact coordinate blocks; construct registered-rank role maps; compute predictions; independently measure realized dimensions; rotate and repeat; construct full/deficient child tuples; run wrong-estimator negative controls; aggregate tables and two central figures.

**Outputs:** per-condition CSV files, `summary.json`, run manifest/log, configuration snapshot, and two central figures.

**Failure labels:** `invalid_registered_geometry`, `ambient_saturation`, `oracle_measurement_leakage`, `svd_gap_ambiguous`, `recurrence_mismatch`, `activation_guard_failure`, `negative_control_failure`, `incomplete_run`.

## 12. Success / Failure / Insufficient Evidence

**Success:** primary metric zero; coordinate oracle, rotated SVD measurement, and recurrence prediction agree; known-good/bad/confusing guards produce their registered outcomes; activation controls separate reachability from observation.

**Failure:** any valid and numerically separated exact mismatch; failed negative control; or failed full-activation attainability.

**Insufficient:** incomplete registry, ambient saturation, ambiguous numerical rank, or inability to establish independent computation paths.

## 13. Figure Contracts

### Figure 1: Predicted versus independently measured dimension

The x-axis is formula prediction and the y-axis is SVD measurement; color separates binary, b-ary, and cumulative records. Support requires every point on the diagonal and maximum mismatch zero. It can establish implementation agreement only, not a language claim.

### Figure 2: Growth under distinct mechanisms

The x-axis is tree layer and the y-axis is dimension; line style separates layer from cumulative dimension, and color separates G1--G5. The registered pattern distinguishes worst expansion, linear accumulation, fixed global space, and bounded new directions. It cannot identify which regime describes language.

Activation remains a table-level guard rather than a third central claim figure.

## 14. What This Cannot Claim

This experiment cannot decide whether short language sequences share a low-dimensional space, whether real parent nodes obey shared linear role maps, which non-tree language structure is best, whether local subspaces correspond to useful experts, how a Router should select experts, or whether real-model noisy bounds are non-vacuous. Q2 structure extensions and Q3 MoE/Router construction remain parked.

## 15. Review Notes And Protocol Changes

- 2026-07-19: Chinese draft written from the T2--T7 refinement; the main theorem allows rank-reducing role maps and treats injectivity only as a special case.
- 2026-07-19: Researcher approved the registered conditions, smoke guards, and full run. English canonical protocol frozen without expanding to real-language or MoE experiments.
