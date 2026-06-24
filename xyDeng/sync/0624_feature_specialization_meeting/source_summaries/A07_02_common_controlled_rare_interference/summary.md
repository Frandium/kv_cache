# Summary: A07_02 Common-Controlled Rare Interference

## Purpose

Test whether label-free common-control reduces rare loss beyond matched load-only MoE on the A07_01-valid D07 surface.

## Conclusion

Supported for controlled synthetic D07. Common-control strongly reduces rare loss while load balance and conflict-signal guards remain matched.

Primary metric: `RIR_cc = 0.9502 +/- 0.0006`.

## Exact Setup

Run: `a07_common_rare_conflict_full_20260623_1`

Seeds: `20260623` to `20260630`.

Method: unlabeled mean-centering plus k-means routing on $\phi(h_i)$. The expert loss is evaluated on full hidden state; routing uses the common-controlled representation.

Dependency: A07_01 passed with `Delta_vs_best_null = 0.8601`.

## Key Evidence

| Check | Result | Judgment |
| --- | ---: | --- |
| `RIR_cc` | `0.9502` | pass |
| `FoO_cc` | `1.1049` | pass |
| load-only max load | `0.2500` | matched |
| common-control max load | `0.2500` | matched |
| active experts | `4.0000` | matched |
| load entropy | `1.0000` | matched |
| raw conflict readout | `1.0000` | pass |
| phi conflict readout | `1.0000` | pass |

## Central Figure

![A07_02 common-control rare interference](figures/rare_interference_common_control.png)

This figure tests whether common-control recovers rare loss relative to load-only routing. The positive `RIR_cc` bar supports a method effect beyond load matching. It does not prove real expert semantics or real-data transfer.

## Claim Boundary

This supports common-control as a controlled synthetic D07 method benefit. It does not yet claim neural training robustness, natural-language specialization, or deployable routing.

## Next Decision

Use the saved route assignments and expert IDs for A07_03 route-function binding.
