# Detailed Result

Anchor: `ntp_style_top1_router_lockin_anchor.md`
Protocol: `01_protocol_for_approval.md`
Summary: `03_summary.md`

## 0. Quick Recap

目的：用最小 NTP-style top-1 proxy 定位 router collapse 的形成机制。

假设：top-1 route-pattern separation 由早期 router geometry 决定。

实验思路：审查 failed seeds、success seed 和 no-router-bias diagnostic 的 trajectory。

结论：failure 是 step-0 lock-in；success 是 step-0 split 被放大；no-bias 只能部分改善 split rate。

证据：核心表和图位于 `tables/` 与 `figures/`。

## 1. Setup

```text
data: h_A=e1, h_B=e2
model: 2 linear experts + linear router + linear LM head
routing / algorithm: top-1 selected expert only
loss: next-token CE
conditions: ntp_no_router_prior, ntp_no_prior_no_expert_advantage, no-router-bias diagnostic
seeds: 0-2 for trajectory; 0-9 for bias comparison
checkpoints: 0, 10, 50, 100, 200, 300
```

## 2. Stage Evidence

| stage | evidence | passed/failed/unclear | failure reason | what this rules out |
|---|---|---|---|---|
| early lock-in | failed seeds collapsed at step 0 | passed | top-1 already same expert | late optimization drift |
| gate amplification | selected gate rises to about 0.985 | passed | router confidence amplifies initial top1 | random final-only artifact |
| expert starvation | unselected expert usage is 0 in collapsed seeds | passed | no token update path | balanced hidden training |
| success dynamics | split seed stays NMI 1 | passed | initial split gets amplified | inevitable collapse |
| no-router-bias | split rate 3/10 -> 6/10 | partial | shared bias removed but random same-sign deltas remain | bias-only explanation |

## 3. Main Results

Early lock-in:

```text
failed seeds: step0 same expert, final NMI=0
success seed: step0 split, final NMI=1
```

Success dynamics:

```text
selected logit margin grows from near 0 to about 2
selected CE becomes much lower than unselected CE
```

No-router-bias:

```text
with bias:  random-router split 3/10
no bias:    random-router split 6/10
known good: split 10/10
```

## 4. Key Figures And Tables

![Early lock-in audit](figures/ntp_style_lockin_audit.png)

Supports: failed runs are collapsed from step 0 and gate confidence amplifies the choice.

![Success dynamics audit](figures/ntp_style_success_dynamics_audit.png)

Supports: initial split is stable and selected experts form counterfactual CE advantage.

![No router bias summary](figures/ntp_style_no_router_bias/summary_by_seed.png)

Supports: removing router bias improves split rate but does not guarantee separation.

Key tables:

```text
tables/ntp_style_lockin_audit.csv
tables/ntp_style_success_dynamics_audit.csv
tables/ntp_style_initial_router_audit.csv
tables/ntp_style_router_bias_comparison.csv
```

## 5. Failure Decomposition

Does this falsify:

- physical prior? no; it supports path dependence.
- mathematical model? no; delta signs explain initial split/collapse.
- operationalization / proxy? not falsified, but remains minimal.
- implementation? no obvious issue after known-good and success controls.
- metric? token accuracy is insufficient because CE can be perfect with NMI 0.

## 6. Known Good / Known Bad / Known Confusing Review

known good: router prior or step-0 split should stay separated.

known bad: step-0 same expert should collapse.

known confusing: token accuracy 1.0 with NMI 0.

what they protected: implementation sanity, lock-in mechanism, metric mismatch.

what remains unprotected: full LM transfer and attention transfer.

## 7. Interpretation

```text
bad init -> same-sign router deltas -> same expert -> starvation -> collapse
good init -> opposite-sign router deltas -> split experts -> co-amplification
router bias -> shared delta shift -> higher chance of same-sign deltas
```

## 8. Claim Boundary

Can claim:

```text
Early top-1 geometry controls route-pattern separation in this minimal proxy.
```

Cannot claim:

```text
utility-aligned specialization or full LM behavior.
```

## 9. Next Decision

```text
Test structured router initialization as the smallest anti-lock-in mechanism.
```

## 10. Artifact / Repro Map

```text
code workspace: daily_research_reports/0531/exp_2x2_understanding/
runner: run_ntp_style_lockin_audit.py
runner: run_ntp_style_success_dynamics_audit.py
runner: run_ntp_style_initial_router_audit.py
runner: run_ntp_style_no_router_bias_audit.py
runner: run_ntp_style_router_bias_comparison.py
result dir: daily_research_reports/0531/exp_2x2_understanding/
figure dir: figures/
table dir: tables/
repro command: conda run -n torch_geo python <runner>
```
