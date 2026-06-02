# Execution Plan: 05_01 Geometric Inhibition For Group Meeting

Anchor:

```text
../../problem_anchors/05_01_geometric_inhibition_anchor.md
```

Approved protocol:

```text
protocol_for_approval.md
```

Story:

用最小 uniform multi-B 实验判断 slot-stable router initialization 和 geometric inhibition 是否能稳定 top-1 MoE routing，并把结论压缩成组会可讲的几何机制图。

## 1. Run Goal

Decision question:

```text
在 uniform multi-B synthetic 中，geometric inhibition 是否在 slot-stable initialization 之外提供额外 routing stabilization？
```

Primary claim to test:

```text
slot prototype 给 router 一个几何初始方向；
token-level margin 和 router-center separation 可以减少训练中的 route drift / slot mixing。
```

Meeting-facing claim boundary:

This is a routing-geometry diagnostic. It does not claim full expert utility, Zipfian robustness, real-data transfer, or label-free specialization.

## 2. What Is Different From 05 Similarity-Prior Routing Inhibition

| Aspect | Previous 05 experiment | New 05_01 geometric inhibition |
| --- | --- | --- |
| Main purpose | Decide whether router init / expert warmup / inhibition jointly induce utility-aligned specialization | Produce a simpler group-meeting diagnostic for routing geometry and inhibition |
| Conditions | 8 dot-product $2^3$ conditions + 3 cosine diagnostic = 11 conditions | 3 dot-product + 3 cosine = 6 conditions |
| Seeds | 4 seeds | 3 seeds by protocol |
| Expert warmup | Included and became decisive | Removed |
| Primary metric | Assign-Utility / forced expert loss diagonal | final route-slot NMI, route drift, route heatmap, geometry |
| Utility metrics | Primary decision evidence | Optional sanity check only |
| Inhibition | token-to-assigned-expert logit margin | token logit margin + router-center separation |
| Main visual story | utility alignment across ablations | geometric separation and routing stabilization |
| Claim allowed | warmup/inhibition can bind prototype assignment to utility in uniform synthetic | geometric inhibition can stabilize slot-aligned routing under external slot assignment |
| Claim not allowed | label-free or full-LM specialization | expert utility solved, Zipfian solved, real data solved |

Key simplification:

```text
drop W/expert-warmup factor;
focus on R/init and G/geometric-inhibition;
use cosine only as router-type comparison, not as a full extra factor.
```

## 3. Data / Case Construction

Use the same uniform multi-B synthetic setup:

```text
slots = 4
B identities = 256
sequence form = [r_start, C_s, B_i, Y_{s,i}, r_end]
primary routing position = B_i position
positive assignment = a(s,i)=s
```

Recommended fixed sizes, to keep comparison with previous 05 understandable:

| Split / Use | Per Slot | Total | Purpose |
| --- | ---: | ---: | --- |
| train | 5000 | 20000 | full causal NTP training |
| eval | 1600 | 6400 | final metrics |
| calibration | 512 | 2048 | slot prototype construction |
| trajectory eval | 256 | 1024 | trajectory figures |

Training:

```text
steps = 1600
batch_size = 384
seeds = 3
```

If time is tight for group meeting:

```text
keep data sizes fixed;
reduce only seeds from 3 to 2 after user approval;
do not shorten steps unless smoke/full separation shows convergence is early.
```

## 4. Conditions

Run six conditions exactly from the protocol:

| Condition | Router | Init | Geometric inhibition | Pair role |
| --- | --- | --- | --- | --- |
| C0 | dot-product | random | no | dot baseline |
| C1 | dot-product | slot-stable | no | dot init effect |
| C2 | dot-product | slot-stable | yes | dot inhibition effect |
| C3 | cosine | random | no | cosine baseline |
| C4 | cosine | slot-stable | no | cosine init effect |
| C5 | cosine | slot-stable | yes | cosine inhibition effect |

Correct comparison pairs:

| Hypothesis | Pair | What It Tests |
| --- | --- | --- |
| H1 dot slot-init | C1 vs C0 | Does slot prototype improve step-0 route-slot alignment? |
| H1 cosine slot-init | C4 vs C3 | Does slot prototype improve cosine route alignment? |
| H2 dot geometric inhibition | C2 vs C1 | Does geometric inhibition stabilize dot routing after init? |
| H2 cosine geometric inhibition | C5 vs C4 | Does geometric inhibition stabilize cosine routing after init? |
| H3 router type | C4 vs C1; C5 vs C2 | Is cosine more stable than dot under the same mechanism? |

Do not add:

```text
expert warmup;
Zipfian data;
unsupervised prototype;
hyperparameter sweep;
full 2^3 ablation.
```

## 5. Model / Objective / Implementation Contract

Base model:

```text
SparseTinyTransformer
top1_selected_gate MoE
num_experts = num_slots = 4
```

Router implementation:

```text
dot-product: z_e(h)=w_e^T h
cosine: z_e(h)=tau * normalize(w_e)^T normalize(h)
```

Selected-gate requirement:

```text
o(h)=sum_e m_e(h) g_e(h) E_e(h)
```

The hard mask can be non-differentiable, but $g_e(h)$ must stay in the graph.

Slot-stable initialization:

```text
collect h_{s,i}^{(0)} at B position
compute centered slot prototype p_s
set w_s(0)=tau*p_s for dot
set w_s(0)=p_s for cosine
use router bias=False, so b_e is absent rather than trainable
```

Geometric inhibition:

```text
L_geo = lambda_tok * L_tok + lambda_sep * L_sep
L = L_NTP + L_geo
```

Token-level margin:

```text
L_tok = mean_{s,i} mean_{e != a(s,i)}
  max(0, m_tok - (z_{a(s,i)}(h_{s,i}) - z_e(h_{s,i})))
```

Router-center separation:

```text
u_e = normalize(w_e)
L_sep = mean_{e != e'} max(0, u_e^T u_{e'} - delta_sep)
```

Implementation note:

The previous 05 code already has dot/cosine router support and token-margin inhibition logic. New work should add center-separation loss and a simplified six-condition runner instead of reusing the full 11-condition runner as-is.

## 6. Files To Modify Or Create

Research-system files:

| Action | Path | Purpose |
| --- | --- | --- |
| existing | `Projects/from-attention-to-search/main/problem_anchors/05_01_geometric_inhibition_anchor.md` | source-of-truth anchor |
| existing | `Projects/from-attention-to-search/main/experiments/05_01_geometric_inhibition_anchor/protocol_for_approval.md` | approved protocol |
| create | `Projects/from-attention-to-search/main/experiments/05_01_geometric_inhibition_anchor/execution_plan.md` | this implementation plan |
| create after run | `Projects/from-attention-to-search/main/experiments/05_01_geometric_inhibition_anchor/summary.md` | group-meeting first reading path |
| create after run | `Projects/from-attention-to-search/main/experiments/05_01_geometric_inhibition_anchor/detailed.md` | full experimental record |
| create after run | `Projects/from-attention-to-search/main/experiments/05_01_geometric_inhibition_anchor/figures/` | curated meeting figures |
| create after run | `Projects/from-attention-to-search/main/experiments/05_01_geometric_inhibition_anchor/tables/` | curated tables |

Runnable workspace files:

| Action | Path | Purpose |
| --- | --- | --- |
| maybe modify | `active/synthetic_data_understanding/src/synthetic_data_understanding/router_prior.py` | add `router_center_separation_loss` if absent |
| reuse / maybe modify | `active/synthetic_data_understanding/src/synthetic_data_understanding/tiny_moe.py` | dot/cosine selected-gate router support |
| create | `active/synthetic_data_understanding/configs/h0603a_geometric_inhibition.json` | fixed six-condition config |
| create | `active/synthetic_data_understanding/scripts/run_h0603a_geometric_inhibition.py` | simplified runner and metric aggregation |
| create | `active/synthetic_data_understanding/scripts/submit_h0603a_geometric_inhibition_4gpu_acp.sh` | dry-run-by-default 4GPU submit wrapper |

## 7. Metrics And Figures

Primary metrics:

| Metric | Decides | False positive risk |
| --- | --- | --- |
| step-0 route-slot NMI | whether slot-stable init works | prototype aligns by chance or label leakage |
| final route-slot NMI | whether routing stays slot-aligned | clean routing without utility |
| route drift | whether training destroys initialization | drift metric may miss permutation symmetry |
| route-token NMI | whether routing follows B identity shortcut | token NMI low does not prove utility |
| target-position accuracy | whether mechanism breaks task learning | accuracy can be 1.0 even without specialization |
| selected gate confidence | whether inhibition only sharpens confidence | confidence may rise without alignment |
| router center pairwise cosine | whether centers are separated | separated centers may not improve assignment |
| prototype-to-router cosine | whether router rows keep slot geometry | high cosine may not imply utility |

Optional sanity metrics:

```text
Assign-Utility
forced expert loss diagonal
```

These are not primary in 05_01, but should be logged to prevent a group-meeting false positive where the geometry improves while expert utility collapses.

Required meeting figures:

| Figure | Required pairs | Message |
| --- | --- | --- |
| route-slot heatmap step0/final | C0/C1/C2 and C3/C4/C5 | visual route stabilization |
| route-slot NMI trajectory | C1 vs C2, C4 vs C5 | inhibition reduces drift |
| selected gate confidence trajectory | C1 vs C2, C4 vs C5 | confidence is not enough unless NMI also improves |
| router center pairwise cosine trajectory | C1 vs C2, C4 vs C5 | center separation effect |
| prototype-to-router cosine trajectory | C1/C2/C4/C5 | whether slot geometry is preserved |
| dot-vs-cosine decision panel | C1 vs C4, C2 vs C5 | whether cosine helps |

## 8. Stage Audit

| Stage | Local Question | Input Evidence | Pass / Fail Rule | Debug Artifact | Handoff |
| --- | --- | --- | --- | --- | --- |
| S0 protocol audit | Does run match 05_01 protocol? | six conditions only | no extra warmup/Zipfian/sweep | resolved config | start smoke |
| S1 selected-gate audit | Does router get gradients? | one backward pass | router grad norm > 0 | gradient check table | start smoke |
| S2 inhibition audit | Does geometric loss affect router? | C2/C5 backward pass | geo grad norm > 0 | geo gradient table | start smoke |
| S3 prototype audit | Are prototypes finite and slot-aligned? | calibration hidden states | finite norms, saved cosine matrix | prototype table | start smoke |
| S4 smoke run | Does every condition produce metrics/figures? | 1 seed short run | all tables and core figures exist | smoke result dir | full run |
| S5 full run | Does final evidence answer pairs? | 6 conditions x 3 seeds | no missing seeds, no NaNs | full result dir | write results |
| S6 report pack | Is group-meeting claim bounded? | summary/detailed/figures | claim separates routing geometry from utility | curated figures/tables | group meeting |

## 9. Pass / Fail Criteria

Supported:

1. C1 > C0 and C4 > C3 on step-0 route-slot NMI.
2. C2 > C1 or C5 > C4 on final route-slot NMI / route drift.
3. target-position accuracy does not decrease.
4. selected gate confidence does not become the only improved signal.
5. router center separation improves together with token assignment.

Weakened:

1. slot-stable init does not improve step-0 route-slot NMI.
2. geometric inhibition increases confidence or center separation but route-slot NMI does not improve.
3. route-token NMI dominates route-slot NMI.
4. cosine hurts target accuracy or optional utility metrics.

Insufficient evidence:

1. router gradient is blocked;
2. geometric inhibition gradient is zero;
3. seed variance is too high for 3 seeds;
4. prototype construction is unstable;
5. route-pattern improves but optional utility metrics collapse.

## 10. Result Location

Raw outputs:

```text
active/synthetic_data_understanding/results/h0603a_geometric_inhibition/<run_name>/
active/synthetic_data_understanding/figures/h0603a_geometric_inhibition/<run_name>/
active/synthetic_data_understanding/logs/acp/
```

Curated meeting outputs:

```text
Projects/from-attention-to-search/main/experiments/05_01_geometric_inhibition_anchor/summary.md
Projects/from-attention-to-search/main/experiments/05_01_geometric_inhibition_anchor/detailed.md
Projects/from-attention-to-search/main/experiments/05_01_geometric_inhibition_anchor/figures/
Projects/from-attention-to-search/main/experiments/05_01_geometric_inhibition_anchor/tables/
```

## 11. Approval Gate

Do not submit jobs until the user confirms:

```text
1. use exactly six conditions C0-C5;
2. keep expert warmup removed;
3. use 3 seeds unless time pressure requires 2;
4. treat utility metrics as sanity checks, not primary claims;
5. produce group-meeting figures before writing final claim.
```
