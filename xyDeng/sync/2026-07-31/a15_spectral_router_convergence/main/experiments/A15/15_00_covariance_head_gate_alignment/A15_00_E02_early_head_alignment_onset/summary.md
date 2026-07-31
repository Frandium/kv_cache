---
experiment_id: A15_00_E02_early_head_alignment_onset
status: completed_early_onset_pass_progressive_strengthening_fail
completed: 2026-07-30
primary_anchor: A15_00_covariance_head_gate_alignment
companion_cn: summary_cn.md
---

# Summary: Early Onset of Linear-Gate Head Alignment

Primary anchor: [A15_00](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/15_00_covariance_head_gate_alignment_anchor.md)  
Protocol: [approved Protocol](protocol.md)  
Detailed record: [detailed.md](detailed.md)

## Conclusion

**Head alignment is already strong at the earliest available 10k checkpoint.**
After removing covariance-amplitude amplification, the LB Gate has 10.42 times
the head-versus-middle gain and 37.11 times the head-versus-tail gain. The
batch-gradient Gate has corresponding ratios of 9.19 and 42.73. All four log
contrasts have paired basis-bootstrap lower bounds above zero and lie far
above singular-value-preserving random-orientation q95 values of 0.034--0.048.
The same-layer actual-input contrast is also much stronger than wrong-layer
and expert-input-basis controls. Thus the 10k state is not explained by raw
input energy or an arbitrary Gate orientation.

**The strong claim that 10k--30k training keeps making the Gate more
head-selective fails.** Endpoint head:middle ratios fall to 5.38 (LB) and 4.99
(batch-gradient) by 30k; head:tail ratios fall to 19.60 and 24.80. Holding the
representation basis fixed, both lineages' Gate-weight changes precisely
reduce head-versus-middle selectivity in both saved intervals. For
head-versus-tail, batch-gradient also dilutes the contrast in both intervals,
whereas LB slightly strengthens it; representation-basis drift more than
offsets that LB effect, so both endpoint ratios still decline.

The supported training description is therefore:

> By the first saved point, the Router–representation system is already
> strongly and non-randomly aligned with the covariance head. From 10k to 30k,
> the endpoint remains head-dominant but becomes less exclusively head-focused,
> as middle and tail access catch up. The common Gate-weight tendency is
> broadening toward middle; tail broadening is lineage-conditioned.

This does not contradict the fact that every net displacement is itself
head-oriented. $B^{update}$ asks where an update vector points when treated as
a Gate by itself; $\Delta_WB$ asks whether adding it to an already more
head-biased Gate increases or decreases the endpoint ratio. Only the latter
answers progressive strengthening.

Because 10k corresponds nominally to about 7.86B training tokens, E02 locates
the onset only as **before 10k / before about 7.86B tokens**. It does not show
the exact onset or prove that Gate gradients, rather than pre-10k joint
Router–representation co-adaptation, caused it.

## Decisive Evidence

![Early-onset decision view](figures/figure0_early_onset_decision.png)

The dashed lines near zero in the endpoint panels are matched
singular-value-preserving random-orientation q95 values. The right panels show
the fixed-basis Gate-weight effect: values above zero strengthen relative head
preference, and values below zero dilute it.

| Lineage | 10k $G_H/G_M$ | 10k $G_H/G_T$ | 30k $G_H/G_M$ | 30k $G_H/G_T$ | $\Delta_WB_{H:M}$, 10→20 / 20→30 | $\Delta_WB_{H:T}$, 10→20 / 20→30 | Typed result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LB | 10.42 | 37.11 | 5.38 | 19.60 | -0.197 / -0.075 | +0.029 / +0.074 | early-present; middle broadening; tail effect mixed |
| batch-gradient | 9.19 | 42.73 | 4.99 | 24.80 | -0.251 / -0.129 | -0.030 / -0.038 | early-present; both contrasts diluted |

All displayed fixed-basis effects have 95% basis-bootstrap intervals entirely
on the stated side of zero. Full intervals are in
[trajectory_decomposition.csv](tables/trajectory_decomposition.csv); all
endpoint intervals and nulls are in
[endpoint_contrasts.csv](tables/endpoint_contrasts.csv).

## What “Can See Middle/Tail” Means Here

Middle and tail gain never become zero. Their current native-route dependence
also grows over the early window:

| Lineage | Step | Median route flip after removing H / M / T |
| --- | ---: | ---: |
| LB | 10k | 0.797 / 0.079 / 0.009 |
| LB | 30k | 0.745 / 0.115 / 0.014 |
| batch-gradient | 10k | 0.743 / 0.056 / 0.008 |
| batch-gradient | 30k | 0.674 / 0.086 / 0.011 |

Thus the accurate Q1 statement is not “the linear Router only sees the head.”
It is “the Router is extremely head-dominant by 10k, while middle and tail are
nonzero and gain relative access/use by 30k.” Route flips are static current-use
diagnostics; they do not establish that middle/tail dispatch improves loss.

The complete 12-band profile is also smooth rather than head-versus-rest only.
F1 is strongest at every endpoint and F2 is above the simultaneous orientation
null envelope. By 30k, F3 also crosses that envelope in both lineages, while
deeper bands remain weaker but nonzero. See
[fine_profile_summary.csv](tables/fine_profile_summary.csv) and the
[full-band figure](figures/figure1_endpoint_full_band_access_use.png).

## Guards and Rival Explanations

- All six 10k/20k/30k checkpoint provenance and coordinate checks passed.
- E02 reused exactly the E01 token tensors: 32 calibration sequences and 64
  held-out documents, each 256 tokens, with identical SHA-256 hashes.
- Direct Gate pre-input relations and native logit replay were exact at all 72
  model × checkpoint × layer cells; top-1 agreement was 1.0.
- All 12 layers at all six endpoints passed coarse half-split basis stability.
- The orientation null preserved nonzero Gate singular values with maximum
  relative error $2.33\times10^{-6}$.
- The full $3\times3$ Gate-weight × representation-basis crossing was used;
  representation drift is reported rather than folded into Gate training.
- The batch-gradient lineage is not a pure gradient-switch control: its
  differentiable batch component also entered the training-time forward
  center. Cross-lineage differences cannot be attributed solely to gradient
  flow through the center.

## Claim Boundary and One Next Decision

E02 establishes that two trained linear Gates are strongly head-aligned by
10k on their actual inputs, and that 10k--30k does not progressively strengthen
that relative preference. It does not establish the exact pre-10k onset,
per-step gradients, a causal covariance mechanism, middle/tail functional
utility, or loss/FLOP benefit.

**One next Q1 decision:** decide whether to authorize E03, a dense online
dynamics run from initialization through at most 2B tokens. Its completion
criterion is a time-resolved decomposition of endpoint $B_t$ into raw Gate
gradient, optimizer-applied update, fixed-probe $U_t$, signed
$W_t$--$\Delta W_t$ band cross terms, margins, flips, and load, with saves
dense enough to locate whether head alignment forms during warmup or later.

