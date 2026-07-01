# 2026-07-01 Feature Route Margin And Common Interference Sync

## Scope

This package syncs the current meeting-facing synthesis for the feature-level
expert-specialization line. It updates the story from "initialization works" to
"preservation is explained by positive route margin, while common-component
interference remains a modeling bottleneck."

## Read First

```text
meeting_brief_cn.md
```

## Current Claim

Controlled feature routing can be initialized by representation clustering and
can be preserved in the controlled training window. The supported preservation
mechanism is not active router-center tracking; it is a positive margin buffer.
When the initial margin is made too thin, routing can fail even if the step-0
assignment is correct.

## Claim Boundary

This package does not claim semantic experts, expert utility, or a validated
common-subtraction method. A06_22 shows that real-DCLM collapse is reproduced,
but common/residual dominance flips across common projectors, so the common
operator is not yet closed.

## Next Decision

Define a more physical common operator and test whether routing on
common-subtracted hidden states improves rare-feature margins without hurting
language-model loss.

## Included

- `meeting_brief_cn.md`
- `report_card.md`
- `figures/`

## Excluded

Raw logs, checkpoints, full result directories, data, and experiment code.
