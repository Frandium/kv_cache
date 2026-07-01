# Feature Center Margin Mechanism Sync

## Scope

This package syncs the June 29 feature-specialization mechanism update from
`from-attention-to-search` into the `kv_cache` `xyDeng` reading surface.

It is a curated research package. It includes a meeting-facing synthesis, a
result index, the next mechanism todo, and the linked theory note. It excludes
raw logs, checkpoints, datasets, full result directories, and experiment code.

## Read Order

1. `report_card.md`  
   Fast decision read: current claim, boundary, and next todo.
2. `meeting_brief_cn.md`  
   Full meeting brief in Chinese: problem chain, minimal model, evidence, and
   next mechanism question.
3. `result_index_cn.md`  
   Compact index of the A06 results that support the brief.
4. `next_todo_cn.md`  
   Operational next step: margin gate and spectral-band margin decomposition.
5. `theory/nested_prediction_spectral_anisotropy_causal_rewrite_v4_zh.md`  
   Advisor theory note used to connect margin preservation with high-gain
   common spectral bands.

## Current Claim

In the controlled no-position synthetic bridge, feature-to-expert
specialization can be formed by label-free feature-center initialization and
can be preserved during training when the initialization creates a positive
margin buffer.

The stronger mechanism update is:

```text
preservation is not active router-center tracking;
it is observed margin-buffer preservation under the tested trajectory.
```

## Claim Boundary

This package does not claim:

- real-language semantic experts;
- expert utility;
- solved real-DCLM training preservation;
- a deployable label-free initializer;
- that positive total margin is always a trustworthy feature margin.

## Next Mechanism Question

The next mechanism audit should ask:

```text
Does the positive margin come from feature residual directions, or from a
high-gain common spectral band?
```

If residual margin predicts preservation, A06_08 discovery and A06_09
preservation can be connected as a feature-residual mechanism. If total margin
is mostly common-band margin, current preservation only proves stable routing,
not trustworthy feature-specific specialization.

