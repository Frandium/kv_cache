# 0622 A06_03 to A06_07 Router Geometry Sync

## Scope

This package summarizes the A06_03-A06_07 router-geometry line from the active `from-attention-to-search` work branch.

It is a curated reading package for the `kv_cache` specialization mainline. It does not include raw logs, checkpoints, datasets, or full result directories.

## Reading Order

1. `main_summary.md`  
   One-page decision summary.
2. `exp_line.md`  
   Handwritten-style experiment line with the reliability audit.
3. `meeting_brief.md`  
   Teaching-oriented explanation for a meeting audience.
4. `source_summaries/`  
   Curated original experiment summaries copied from the active work branch.
5. `figures/`  
   Central figures referenced by the meeting brief.

## Included Files

- `main_summary.md`
- `exp_line.md`
- `meeting_brief.md`
- `source_summaries/A06_03/summary.md`
- `source_summaries/A06_04/summary.md`
- `source_summaries/A06_05/summary.md`
- `source_summaries/A06_06/summary.md`
- `source_summaries/A06_07/summary.md`
- `figures/`

## Current Claim

Uniform feature frequency is not enough for feature-level MoE specialization. The tested sequence separates several mechanisms:

```text
gate-only row norm bias
-> hidden common component
-> residual geometry after centering
-> oracle feature partition reachability
-> failure of simple label-free global controls
```

## Boundary

This package does not claim real DCLM transfer, training stability, expert utility, semantic specialization, or failure of all label-free methods.

## Next Decision

Move from load repair to feature discovery or anti-lockin:

- discover pseudo-feature centers before router initialization; or
- start from oracle / pseudo-oracle routing and test early top-1 training stability.
