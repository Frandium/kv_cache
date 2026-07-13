# MTP Composition, Capacity, and Causal-Use Handoff

## Scope

This package provides the self-contained 0713 meeting brief and the smallest
curated evidence chain needed to audit its three conclusions:

1. the permutation-composition task has strict joint-evidence necessity, but
   the current small neural model has not passed the task capability gate;
2. a controlled linear model supports structural rank competition and shows
   that curriculum order cannot repair a missing representation direction;
3. informative future-token supervision directly updates the current hidden
   state, while causal use by the shared standard next-token head remains open.

## Reading Order

1. [Meeting brief](meeting_brief_cn.md)
2. [Current causal-use anchor](anchors/11_27_standard_head_causal_representation_use_anchor_cn.md)
3. [Current causal-use protocol](protocol/standard_head_causal_representation_use_protocol_cn.md)
4. Decision evidence under source_results/
5. Mechanism-boundary evidence under supporting_summaries/

## Included Material

- five current or directly supporting Chinese anchors;
- the pending standard-head causal-use protocol;
- protocol, summary, and detailed records for task calibration, explicit
  binding calibration, rank competition, and curriculum boundary;
- four supporting summaries for direct current-state supervision, informative
  horizon inclusion, cross-position transport, and end-to-end behavior;
- the two figures embedded by the meeting brief.

## Excluded Material

Raw logs, checkpoints, datasets, code, complete result directories, unrelated
anchors, and superseded 0712 meeting drafts are intentionally excluded.

## Current Claim Boundary

The package supports a controlled direct-supervision mechanism and a linear
rank-competition mechanism. It does not establish a general MTP advantage,
Transformer scaling law, real multi-hop downstream gain, or curriculum
pretraining benefit.

## Next Action

First establish a shared pretrained-model checkpoint that passes the two-hop
composition capability gate. Only then clone matched NTP and informative
second-future-token conditions and compare causal use through the shared
standard next-token head.
