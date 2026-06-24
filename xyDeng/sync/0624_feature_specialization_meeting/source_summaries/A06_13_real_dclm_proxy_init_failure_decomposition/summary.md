# Summary: A06_13 Real-DCLM Proxy-Init Failure Decomposition

## Purpose

This audit localizes where the real-DCLM proxy initialization pipeline first fails.

## Inputs

- A06_10 proxy discovery result.
- A06_11 proxy-to-linear-router bridge result.
- A06_12 training preservation result.

Run:

```text
run_dir: runs/failure_decomposition/a06_13_failure_decomposition_20260622_full01
script: scripts/run_failure_decomposition.py
```

## Result

Decision: the first failed stage is training preservation, $a(0)\to a(t)$.

Proxy discovery passes with boundary. Linear bridge passes for raw-center rows. The pipeline first fails when ordinary DCLM training overwrites the step-0 proxy route.

| Stage | Metric | Value | Guard | Status |
| --- | --- | ---: | ---: | --- |
| proxy discovery | step-0 late-layer residual stability NMI | 0.1923 | 0.0002 | pass with boundary |
| proxy discovery guard | step-10 late-layer residual stability NMI | 0.5451 | 0.0002 | pass |
| linear bridge | step-0 raw-center proxy-route NMI | 0.7449 | 0.0343 | pass |
| training preservation | raw-center step-10 proxy-route NMI | 0.0131 | 0.0047 | fail |
| loss guard | step-10 loss delta raw-center minus random | 0.0032 | 0.0000 | pass |

## Central Figure

![Raw-center proxy retention](figures/full01_raw_center_proxy_retention.png)

This figure shows the fraction of step-0 proxy-route NMI remaining during training. Only 5.4% remains at step 5 and 1.7% remains at step 10.

## Claim Boundary

Can claim: the current failure is training-feedback override, not failure to find a proxy and not failure to linearly route it at step 0.

Cannot claim: final solution, expert utility absence, or impossibility of real-data specialization.

## Next Decision

The next experimental direction should test preservation mechanisms:

- router freeze or delayed router training;
- lower router learning rate;
- proxy-preservation auxiliary loss;
- common/load anti-collapse regularizer.

A06_14/A06_15 should remain parked until proxy routing is preserved.
