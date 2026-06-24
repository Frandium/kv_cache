# Summary: A07_01 Common / Rare Conflict Metric Audit

## Purpose

Test whether the controlled D07 common/rare conflict surface is valid enough for method testing.

## Conclusion

Supported for controlled synthetic D07. The oracle-relievable rare-loss gap is large and the validity guards pass.

Primary metric: `Delta_vs_best_null = 0.8601 +/- 0.0008` over 8 seeds.

## Exact Setup

Run: `a07_common_rare_conflict_full_20260623_1`

Seeds: `20260623` to `20260630`.

Implementation: controlled synthetic hidden-state audit with dense null, load-only MoE null, oracle conflict-group gate, common-control route, shortcut probes, and conflict readout probes.

Result root: `/data/250010109/Research_System/Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/results/a07_common_rare_conflict/a07_common_rare_conflict_full_20260623_1`

## Key Evidence

| Check | Result | Judgment |
| --- | ---: | --- |
| `Delta_vs_best_null` | `0.8601` | pass |
| dense rare loss | `1.4399` | null worse than oracle |
| load-only rare loss | `1.5300` | null worse than oracle |
| oracle rare loss | `0.5798` | oracle relieves conflict |
| raw hidden conflict readout | `1.0000` | readable |
| common-control phi readout | `1.0000` | preserved |
| strongest shortcut accuracy | `0.5000` | below `0.60` threshold |
| oracle forbidden-variable use | `0` | pass |

## Central Figure

![A07_01 oracle rare gap](figures/oracle_rare_gap.png)

This figure tests whether the rare loss can be relieved by conflict-group routing beyond the best null. The positive bars show dense, load-only, and best-null rare-loss gaps against the oracle. It supports the metric surface, but it does not prove any learned method benefit.

## Claim Boundary

This supports D07 as a controlled synthetic metric surface only. It does not claim real-language transfer, neural checkpoint behavior, or common-control method benefit.

## Next Decision

Use this D07 surface as the dependency for A07_02.
