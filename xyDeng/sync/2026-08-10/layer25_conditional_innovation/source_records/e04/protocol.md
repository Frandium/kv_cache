---
experiment_id: A15_08_E04_strict_conformance_repair
anchor_id: 15_08_target_conditioned_layer_innovation
status: APPROVED_FOR_IMPLEMENTATION_AND_ONE_FRESH_CONFIRMATION_RUN
canonical_language: en
approved_scope: implementation_tests_fresh_confirmation_extraction_one_evaluation_and_complete_record
approved: 2026-08-11
design_frozen_at: 2026-08-11T01:42:30Z
registered_revision: A15_08_E04_v1
---

# Protocol: A15_08_E04 Strict-Conformance Repair

## 0. Approval And Repair Snapshot

- **Approval basis:** the researcher explicitly authorized completion of the
  logically correct first exploration without another approval pause. The
  final E03 cold read found two Protocol-conformance defects. One minimal fresh
  repair that changes no scientific object, threshold, or downstream question
  remains inside that authorization.
- **Why this is a new experiment:** E03 keeps its measured values but is
  reclassified as INELIGIBLE_GUARD because its data schema did not instantiate
  every registered namespaced ID and its required evidence bundle omitted raw
  extraction command receipts and simple episode/bootstrap arrays. E04 cannot
  amend or upgrade E03.
- **Question:** with the same E02 TRAIN/DEVELOPMENT-selected readouts,
  residualizers, directions, ranks, thresholds, and control families frozen
  before a strictly new and fully namespaced confirmation set exists, does the
  layer-25 normalized attention transition add terminal-code accessibility
  beyond the old state; if so, are the same two conditional-innovation
  directions sufficient?
- **Primary condition:** complex two-hop composition. Simple is secondary.
- **Not tested:** adjacent Transformer blocks, global representation rank,
  Shannon-information creation, natural language, native use, expert utility,
  or Router gain.
- **Authorization boundary:** one implementation/test cycle, one fresh
  confirmation generation and extraction, one evaluation, and completion of
  the registered evidence record. No rank ladder or Router run is authorized.

## 1. Frozen Scientific Objects

At the final Answer token of Qwen3-8B block 25,

$$
X=N_{25}(h),\qquad Z=N_{25}(h+a),\qquad U=Z-X,
$$

and

$$
R_U=U-\widehat{\mathbb E}_{lin}[U\mid X].
$$

E04 freezes the same five E02 TRAIN/DEVELOPMENT-only inputs used by E03:

| Object | Required SHA-256 |
| --- | --- |
| frozen linear models | 8793a01646bdb44aef809ccaedbbf0819cfdfe7cc982c90d5a59915dfb764173 |
| frozen directions | 390bcc3d37859a6ac041861294f7e2d7f6a72cfb2e6a9b31b1def745dda671de |
| selection ledger | a0f468f9f27ea630a76edc62e0ed5a09e74589e7cab74ce2c27557547b947dce |
| simple E02 TRAIN representation | cca41dc103e3066d5ebd1d515e561aab8e95002ed41e1d1ac488cc5b839d720c |
| complex E02 TRAIN representation | 4e1032c3c508f7849a58f17bd2c9310d0a6d4db08fc8eb806db19ad1c77d4919 |

No E03 loss, gain, bootstrap sample, verdict, direction, threshold, readout, or
hyperparameter may enter E04. E03 data and map arrays may be read only during
preconfirmation provenance construction to freeze collision hashes; they
cannot construct an E04 record or map and are forbidden after the E04 freeze.

The same rank two, 80% threshold, 64 E02-frozen random rank-two readouts, and
64 E02-frozen TRAIN-label-permutation rank-two readouts are reused without
refit or reselection.

## 2. Hypotheses And Strongest Rival

**H2 — target-conditioned added accessibility.** On fresh complex confirmation
episodes, the frozen full-$R_U$ correction lowers terminal-code cross-entropy
beyond the frozen old-state readout, the frozen $Z$ readout has the same sign,
and the true gain exceeds the within-bootstrap q95 of 64 target-independent
balanced-bank controls.

**H1' — frozen rank-two sufficiency.** Only after H2 Pass, the same two E02
TRAIN/DEVELOPMENT-selected directions retain at least 80% of the full-$R_U$
gain and exceed the q95 of the same 64 random and 64 TRAIN-label-permutation
control models under paired episode resampling.

**Strongest rival:** generic correction capacity or a target-correlated control
construction can mimic an update gain. The same-dimensional state comparison
and post-freeze balanced banks separate this rival. H1' is rank-two retrieval,
not the global meeting hypothesis that representations are low rank.

## 3. Preconfirmation Freeze Contract

Before the E04 run/data roots exist, freeze:

1. this Protocol and every E04 code/test hash;
2. the five E02 identities in Section 1;
3. recomputed E02 TRAIN-only residualizers with DEVELOPMENT reconstruction
   agreement and zero E02/E03 confirmation rows;
4. frozen E02 readouts/directions, rank two, and the 80% threshold;
5. 64 label-free mismatch recipes with seeds 8200--8263 and a new split salt;
6. generation seed 8103, 128 episodes, eight rows per condition and episode,
   prose-table wording, and the exact ID schema in Section 4;
7. 2,000 paired episode bootstrap draws with seed 8281 and NumPy
   method=higher q95 recomputed inside every draw;
8. canonical content hashes of all E02 and E03 confirmation JSON identifiers,
   contexts, texts, and actual mismatch-map arrays, for collision checks only;
9. the exact required artifact inventory in Section 9.

The ledger contains no E04 label, target code, source label, materialized index
map, contingency, per-record feature, loss, gain, or result. The receipt must
attest that both E04 roots were absent and the fresh-label access count was
zero. Any forbidden field or changed frozen hash yields INELIGIBLE_GUARD.

## 4. Fresh Confirmation Data And Concrete ID Schema

After receipt verification, generate only:

| Split | Episodes | Records per condition | Seed | Wording |
| --- | ---: | ---: | ---: | --- |
| CONFIRMATION | 128 | 1,024 | 8103 | prose table |

Every record must contain these distinct fields:

| Field | Concrete contract |
| --- | --- |
| episode | integer group index 0--127 used only for grouping |
| episode_id | string beginning innovation-e04-episode- |
| map_id | string beginning innovation-e04-map- and shared by the simple/complex pair for the episode |
| pair_id | string beginning innovation-e04-pair- |
| record_id | string beginning innovation-e04-record- |
| generation_id | payload-level exact value A15_08_E04_strict_conformance_repair |

Every episode contains all eight terminal targets exactly once in each
condition. All identifier fields must be unique at their intended scope and
consistent with the integer episode group. The collision audit must compare
record_id, pair_id, episode_id, map_id, complete context, and complete text
against E01, E02, and E03 wherever that field exists. Missing historical fields
count as no historical value, not as a pass for the E04 existence/format guard.
Any E04 missing field, prefix mismatch, inconsistency, count/balance error, or
observed collision yields INELIGIBLE_GUARD.

## 5. Extraction And Execution Receipts

- frozen /data/share/Qwen3-8B, bfloat16, SDPA, use_cache=False;
- block index 24 / reported layer 25; final prompt token only;
- $X,Z,U$ computed in float32 under the same post-attention norm;
- restricted-choice accuracy at least .80 simple and .60 complex;
- replay max error at most $10^{-5}$, direct output-projection relative error at
  most $10^{-5}$, and stored $Z-X-U$ relative error at most $10^{-6}$.

Every freeze, generation, extraction, and evaluation stage must write a
structured receipt containing the exact argv including interpreter, UTC start
and finish, working directory, Python/platform/library versions, requested
device, observable GPU identity when available, input/output paths, input and
output hashes, elapsed time, and final status. A failed stage must retain a
failure receipt outside the not-yet-created scientific output root. Missing or
incomplete success receipt yields INELIGIBLE_GUARD.

Capability is a validity guard only. It cannot select a model, condition,
direction, or verdict.

## 6. Post-Freeze Balanced Banks

Only after receipt verification may the evaluator open E04 labels and groups.
Each seed partitions the 128 episodes into blocks of eight and uses randomized
base permutations plus all eight cyclic offsets in randomized order.

For every bank:

- every cell of the $8\times8$ target-source table equals 16;
- empirical target-source mutual information is zero to $10^{-12}$;
- maps are episode-local permutations;
- all 64 E04 arrays are distinct;
- no E04 array equals an E02 or E03 actual confirmation map;
- no frozen model, direction, ridge, rank, or threshold changes.

Actual E04 maps and audits are written only after label access. Any failure
yields INELIGIBLE_GUARD.

## 7. Metrics And Verdict Mapping

For each complex record,

$$
g_{true}=CE(f_X(X))-CE(f_X(X)+g_{full}(R_U)),
$$

$$
g_{state}=CE(f_X(X))-CE(f_Z(Z)).
$$

Treat the complete eight-target episode as the independent unit. Within every
paired bootstrap draw,

$$
T_{cap}=G_{true}-Q_{0.95}^{higher}(G_{mismatch,1:64}),
$$

$$
D_{80}=G_2-0.8G_{true},
$$

$$
T_{random}=G_2-Q_{0.95}^{higher}(G_{random,1:64}),\qquad
T_{perm}=G_2-Q_{0.95}^{higher}(G_{perm,1:64}).
$$

Report percentile 95% intervals.

- H2 Pass: lower bounds of $G_{true}$, $G_{state}$, and $T_{cap}$ are all
  strictly positive.
- H2 Fail: the upper bound of $G_{true}$ is non-positive.
- Otherwise: H2 Insufficient.
- Only after H2 Pass, H1' Pass requires positive lower bounds for $D_{80}$,
  $T_{random}$, and $T_{perm}$.
- H1' Fail: any one of their upper bounds is non-positive.
- Otherwise: H1' Insufficient.

Simple is secondary and cannot change the verdict. Its per-record, per-episode,
bootstrap sampled-index, and bootstrap replicate arrays for $G_{true}$,
$G_{state}$, $G_2$, and $D_{80}$ must still be retained. Simple has no
registered control family; no simple per-control array is expected.

E01/E02/E03 values cannot be pooled, averaged, used as priors, or used to
select E04 wording or any scientific object.

## 8. Required Tests And Stage Stops

| Stage | Required evidence | Stop rule |
| --- | --- | --- |
| S0 implementation | baseline tests; RED tests proving missing ID fields, missing receipts, incomplete simple arrays, and partial manifest; then all-green log | any test failure or missing RED/GREEN receipt |
| S1 freeze | full ledger, receipt, residualizer, prior-collision hashes, exact argv/environment receipt | existing E04 root, forbidden field, incomplete inventory, hash or DEV mismatch |
| S2 data | JSON, public/data manifests, complete ID-schema audit, E01/E02/E03 overlap audit, generation receipt | any ID, balance, namespace, provenance, count, or collision failure |
| S3 extraction | tensors/logits, manifests and structured receipts for both conditions | capability, hook, reconstruction, hash, or receipt failure |
| S4 banks | 64 maps, contingencies, MI values, prior-map audit | invalid, duplicate, reused, or incomplete bank family |
| S5 evaluation | full primary arrays and required secondary arrays, tables, bootstrap, result, figure, plot contract, evaluation receipt | nonfinite value, changed object, missing array, or receipt failure |
| S6 record | final trace, eligibility audit, full-experiment artifact manifest, Summary, Detailed, Anchor, daily report, independent cold read | missing artifact, hash mismatch, lineage defect, or claim-boundary inconsistency |

Tests must fail before implementation for:

1. a payload missing episode_id or map_id;
2. an ID with the wrong prefix or inconsistent integer episode;
3. a historical collision in any comparable field;
4. a success stage without exact argv/environment receipt;
5. simple evidence lacking episode or bootstrap arrays;
6. a full manifest omitting a registered artifact family;
7. reuse of an E03 actual map;
8. any already registered E03 eligibility mutation.

## 9. Required Outputs And Full Manifest

Worker run root:
runs/a15-08-e04-strict-conformance-repair-20260811/

Freeze root:
preconfirmation_freezes/a15-08-e04-20260811/

Canonical record root:
Projects/from-attention-to-search/main/experiments/A15/15_08_target_conditioned_layer_innovation/A15_08_E04_strict_conformance_repair/

The final full-experiment manifest must enumerate and hash, at minimum:

- Protocol, every frozen code/test file, E02 inputs, residualizer and receipts;
- prior-data/map collision guards and the final event trace;
- full/simple/complex JSON, public/data manifest, ID/overlap audit, generation receipt;
- both representation tensors, logits, extraction manifests and receipts;
- materialized banks and audit;
- primary and secondary arrays, per-record and per-episode/control tables,
  metrics, result, eligibility audit, environment/evaluation receipt, plot
  contract, figure PNG/PDF, and worker analysis manifest.

The full manifest may omit only itself and reader documents written after it.
Every entry records path, SHA-256, byte size, producer stage, completeness, and
claim eligibility. Absence or hash mismatch of a required entry makes the
scientific result INELIGIBLE_GUARD.

## 10. Claim Boundary And Exactly One Next Decision

Even an eligible joint Pass establishes only target- and linear-family-specific
added accessibility for one normalized layer-25 attention transition and the
registered rank-two sufficiency clause. It does not establish adjacent-layer
novelty, whole-representation low rank, native use, expert utility, or Router
gain.

If H2 passes and H1' fails, the only next decision is one fresh-data rank
ladder on the same $R_U$ object. If both pass, the only next decision is one
independent functional-admission design. If H2 is insufficient, open only a
precision decision. If H2 fails, close this named linear object. No downstream
branch is active before E04 is eligible.

The scientific body is frozen as A15_08_E04_v1 before E04 data generation.
After the first generation command, any change requires a new experiment; no
post-data amendment may repair an eligibility guard.
