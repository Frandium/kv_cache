# Experiment design

## Conditions

For each backbone kind, hidden size, and seed, train:

- MTP=1 (NTP);
- MTP=3.

Then freeze every model parameter, discard the trained-head advantage, and
train fresh linear and two-layer MLP probes for `h(P_i) -> S_i`.

## Controls

- Prefix-suffix pairs and bones form a Cartesian product.
- A deterministic checkerboard split holds out prefix-bone combinations while
  retaining every prefix and bone in training.
- NTP and MTP probes have identical architecture, initialization seed, data,
  optimizer, and steps.
- Next-token metrics are reported separately for `P->B0`, `B0->B1`, and
  `B1->S` because `P->B0` intentionally has irreducible ambiguity.
- Prefix spectra are centered and reported using normalized energy, effective
  rank, stable rank, and top-1 energy fraction.

## Pass, fail, and insufficient evidence

The conjecture passes for a condition if MTP=3 has consistently higher held-out
fresh-probe accuracy or reaches the same accuracy with materially fewer probe
updates across seeds.

It fails if NTP fresh probes match MTP in final accuracy and learning speed.

Evidence is insufficient if either backbone cannot solve `B1->S` next-token
prediction, results vary strongly across seeds, train/test probe behavior
diverges, or optimization has not converged.

## Stage-level diagnostics

1. Data: verify Cartesian coverage and held-out pairing.
2. Backbone training: verify `B1->S` next-token accuracy.
3. Native MTP: verify offset-3 suffix accuracy for MTP=3.
4. Frozen probe: compare learning histories and held-out accuracy.
5. Geometry: compare spectrum only after behavioral stages pass.

## Focused compositional run

Before repeating the full 54-condition sweep, run only the two-layer MLP with
hidden size 8, MTP in `{1,3}`, and three seeds. Fresh probe inputs are
layer-normalized. Expand to other hidden sizes and attention only if this run
solves the native next-token task and produces a stable NTP-versus-MTP gap on
held-out prefix compositions.
