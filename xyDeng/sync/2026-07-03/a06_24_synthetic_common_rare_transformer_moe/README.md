# 2026-07-03 A06_24 Synthetic Common/Rare Transformer-MoE Sync

## Scope

This package syncs the A06_24 synthetic common/rare routing audit into the
`kv_cache` `xyDeng` reading surface. It is a curated research package, not a
raw experiment directory.

## Read First

```text
meeting_brief_cn.md
```

Then audit the source records:

```text
anchors/
experiment/protocol.md
experiment/summary.md
experiment/detailed.md
experiment/figures/
experiment/tables/
```

## Current Claim

In a no-position one-layer Transformer plus one-layer MoE synthetic
common/rare task, simple global common subtraction is not a reliable feature
separator. It can change common bias or load, but it does not replace
route-relevant hidden-state population selection.

## Key Evidence

- Step 0 all-position common-subtracted centers: joint feature score `0.405`
  and rare margin p05 `-2.759`.
- Step 0 route-position residual / oracle centers: joint feature score `0.637`
  and rare margin p05 about `11.6`.
- Final all-position common-subtracted routing: joint feature score `0.432`
  and rare margin p05 `-5.427`.
- Final oracle row-projected routing: joint feature score `0.636` and rare
  margin p05 `8.646`.
- Slot-start NMI guard remains low; step-0 maximum mean is `0.024`.

## Claim Boundary

This package does not claim real-DCLM transfer, semantic experts, expert
utility, or that every task-aware common operator fails. It only sets the
synthetic boundary that simple all-position common subtraction is not the next
main method.

## Next Decision

Open the next method anchor around row-projected preservation or a label-free
route-relevant state selector. The next protocol must treat target accuracy as
a validity guard, not as specialization evidence.

## Included

- `meeting_brief_cn.md`
- `report_card.md`
- `anchors/`
- `experiment/protocol.md`
- `experiment/protocol_cn.md`
- `experiment/summary.md`
- `experiment/detailed.md`
- `experiment/summary.json`
- `experiment/figures/`
- `experiment/tables/`

## Excluded

Raw logs, checkpoints, datasets, full result directories, and experiment code
workspaces are excluded.
