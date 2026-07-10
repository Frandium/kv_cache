# A11_10 Finite-Step Semantic Efficiency Protocol

Status: completed full run on 2026-07-09.  
Anchor: `../anchors/11_10_finite_step_semantic_efficiency_anchor.md`.  
Scope: A11 theory closure only. No broad ablation, no natural-language experiment, no MoE bridge.

Result records:

```text
summary.md
detailed.md
tables/finite_step_assumption_summary.csv
tables/seed_level_hitting_time_bound.csv
```

## 1. Experiment Purpose

The experiment tests whether the local first-order semantic velocity mechanism can support a finite-step semantic margin hitting-time bound under explicit assumptions.

It does not attempt to prove that K3 is generally stronger. It tests the assumptions needed for the conditional proposition:

$$
M_K(t+1)\ge M_K(t)+\eta g_K-\eta^2B-\epsilon_t.
$$

If the net lower bound is positive,

$$
\eta g_K-\eta^2B-\bar\epsilon>0,
$$

then:

$$
T_\gamma(K)
\le
\left\lceil
\frac{\gamma-M_K(0)}
{\eta g_K-\eta^2B-\bar\epsilon}
\right\rceil.
$$

The protocol therefore verifies four assumptions:

1. velocity persistence;
2. margin-to-recovery calibration;
3. head drift bound;
4. shared / indirect perturbation bound.

## 2. Falsifiable Hypothesis

**Main hypothesis:** In the controlled A11 synthetic setting, direct informative-horizon supervision maintains a positive early-window semantic margin increment after curvature and perturbation corrections, and the resulting margin threshold predicts native informative-horizon recovery or guarded recovery.

Support requires all four assumption audits to pass. Failure of any assumption does not falsify A11_06 / A11_08 / A11_09 first-order theory; it falsifies or weakens the finite-step efficiency closure.

## 3. Data Construction

Use the same controlled branch data family as A11_06 through A11_09.

Early variable:

$$
Z\sim\operatorname{Unif}\{1,\ldots,m\}.
$$

Decision state:

$$
h_z(t)=h_T(Z=z,t).
$$

Primary clean `tau=3` construction:

$$
Y_1=A,\qquad Y_2=C,\qquad Y_3=S_Z.
$$

General-K geometry constructions, only if needed for calibrated velocity levels:

```text
shared_only_k4:     Y1=A, Y2=C, Y3=A,   Y4=C
single_h3:          Y1=A, Y2=C, Y3=S_z, Y4=A
aligned_h3_h4:      Y1=A, Y2=C, Y3=S_z, Y4=T_z
low_conflict_h3_h4: Y1=A, Y2=C, Y3=S_z, Y4=T_z
```

Branch initialization must pass no-leakage gates:

$$
Q(0)\le0.35,\qquad M_K(0)\approx0,\qquad E_Z(0)\approx0,
$$

and native informative-horizon accuracy must be near chance at step 0.

## 4. Model / Objective

Use the same small causal Transformer / direct-head setup as the existing A11 controlled experiments.

Decision-only next-K objective:

$$
L_K^{decision}(h_T)=\sum_{j=1}^{K}\lambda_j\operatorname{CE}(q_j(\cdot\mid h_T),Y_j).
$$

All-position objective for perturbation audit:

$$
L_K^{all}=\sum_t\sum_{j=1}^{K}\lambda_j\operatorname{CE}(q_j(\cdot\mid h_t),X_{t+j}).
$$

Direct decision-prefix informative term:

$$
L_{direct}=\lambda_3\operatorname{CE}(q_3(\cdot\mid h_T),S_Z).
$$

Non-direct term:

$$
L_{non-direct}=L_K^{all}-L_{direct}.
$$

Frozen-head runs are primary for the theorem window. Trainable-head runs are used only for the head drift audit.

## 5. Conditions

### 5.1 Core decision-only theorem-window conditions

| Condition | Data | K | Head status | Purpose |
|---|---|---:|---|---|
| `zero_velocity_guard` | `shared_only_k4` | 4 | frozen | confirms no semantic velocity without informative horizon |
| `single_info_reference` | `single_h3` | 3 | frozen | primary positive finite-step margin trajectory |
| `aligned_high_velocity` | `aligned_h3_h4` | 4 | frozen | tests higher predicted velocity and bound tightness |
| `conflict_low_velocity` | `low_conflict_h3_h4` | 4 | frozen | tests weak/conflicting informative direction boundary |

These are not a K sweep. They are the minimum controlled velocity levels already used by A11_08 / A11_09 to test the theorem assumptions.

### 5.2 Head drift audit conditions

Duplicate only these conditions with trainable heads:

```text
single_info_reference
conflict_low_velocity
```

Purpose: determine whether reference directions stay valid or whether the theorem must be rewritten in current geometry.

### 5.3 All-position perturbation conditions

Use clean `tau=3` data only:

| Condition | Objective | Purpose |
|---|---|---|
| `allpos_K3_full` | full all-position K3 | direct plus non-direct training |
| `allpos_K3_mask_direct` | all-position K3 with `L_direct` removed | estimates non-direct / indirect contribution |
| `allpos_K2_plus_direct` | K2 all-position plus `L_direct` | estimates direct sufficiency under all-position background |
| `allpos_K2_active` | K2 all-position | non-direct baseline guard |

No loss-weight sweep is included.

## 6. Seeds

Primary seeds:

```text
971, 972, 973, 974, 975
```

A run is interpretable only if at least `4/5` seeds pass the no-leakage gate for the relevant condition group.

## 7. Metrics

### 7.1 Velocity persistence

Reference semantic direction:

$$
v_z^{ref}=v_z^{(K)}(0).
$$

Reference hidden semantic velocity:

$$
G_K^{ref}(t)=\frac1m\sum_z v_z^{ref\top}(-\nabla_{h_z}L_K(\theta_t)).
$$

Window lower bound:

$$
g_{min,K}^{W}=\min_{t\in W,\ t<T_\gamma}G_K^{ref}(t).
$$

What it measures: whether the semantic velocity remains positive before the margin threshold.  
Why needed: it supplies the `g_K` term in the finite-step proposition.  
Support: `g_min,K^W > 0` in at least `4/5` seeds for direct informative conditions.  
Falsify: `g_min,K^W <= 0` before threshold in most seeds.  
Insufficient: gradients are unstable or step-0 metrics are saturated.

### 7.2 Semantic margin trajectory

Reference margin:

$$
M_K^{ref}(t)=\frac1m\sum_z v_z^{ref\top}h_z(t).
$$

Margin hitting time:

$$
T_\gamma^M(K)=\inf\{t:M_K^{ref}(t)\ge\gamma\}.
$$

What it measures: the theorem target variable.  
Why needed: the hitting-time proposition is stated for margin, not directly for `Q`.  
Support: `M_K_ref(t)` increases with the certified lower-bound increment and crosses `gamma` before or near native recovery.  
Falsify: margin decreases while predicted increment is positive, or the observed crossing violates the certified bound.  
Insufficient: no margin threshold is crossed.

### 7.3 Margin-to-recovery calibration

Native informative-horizon accuracy:

$$
H3(t)=\Pr[\arg\max_y q_3(y\mid h_T(t))=S_Z].
$$

Guarded recovery:

$$
Q(t)=\min\{A_{decoder}(t),A_{probe}(t),C_{swap}(t)\}.
$$

Recovery hitting times:

$$
T_{0.9}^{H3}=\inf\{t:H3(t)\ge0.9\},
$$

$$
T_{0.9}^{Q}=\inf\{t:Q(t)\ge0.9\}.
$$

Calibration test:

$$
M_K^{ref}(t)\ge\gamma\quad\Rightarrow\quad H3(t)\ge0.9
$$

or, if H3 is not defined for a condition:

$$
M_K^{ref}(t)\ge\gamma\quad\Rightarrow\quad Q(t)\ge0.9.
$$

What it measures: whether semantic margin is a valid proxy for usable recovery.  
Why needed: without calibration, the finite-step theorem only concerns an internal margin.  
Support: margin threshold predicts recovery threshold with lag at most one evaluation interval in at least `4/5` seeds.  
Falsify: high margin repeatedly fails to produce native H3 / `Q`, or recovery occurs without margin.  
Insufficient: neither margin nor recovery crosses threshold.

### 7.4 Head drift bound

Current semantic direction:

$$
v_z^{cur}(t)=\sum_{j\in\mathcal I_K}\lambda_j(u_{j,z}(t)-\bar u_j(t)).
$$

Reference-current direction correlation:

$$
c_u(t)=
\frac{\sum_z v_z^{ref\top}v_z^{cur}(t)}
{\left(\sum_z\|v_z^{ref}\|^2\right)^{1/2}
 \left(\sum_z\|v_z^{cur}(t)\|^2\right)^{1/2}+\epsilon}.
$$

Drift norm:

$$
D_u(t)=\left(\sum_z\|v_z^{cur}(t)-v_z^{ref}\|^2\right)^{1/2}.
$$

What it measures: whether the reference output geometry remains meaningful during trainable-head training.  
Why needed: A11_10 uses reference margin directions; head drift can shrink or invalidate that theorem.  
Support: `c_u(t)` remains positive and above a predeclared floor through the theorem window.  
Falsify: `c_u(t)` crosses zero or drift dominates before recovery.  
Insufficient: reference geometry fails but current-geometry margin remains predictive; this requires a revised theorem.

### 7.5 Shared / indirect perturbation bound

Direct margin contribution:

$$
G_{direct}(t)=\frac1m\sum_z v_z^{ref\top}(-\nabla_{h_z}L_{direct}(\theta_t)).
$$

Non-direct contribution:

$$
P_{non-direct}(t)=\frac1m\sum_z v_z^{ref\top}(-\nabla_{h_z}L_{non-direct}(\theta_t)).
$$

Perturbation lower bound:

$$
P_{non-direct}(t)\ge -p_{max}.
$$

Estimated certified increment:

$$
d_K(t)=\eta G_{direct}(t)+\eta P_{non-direct}(t)-\eta^2\widehat B(t).
$$

What it measures: whether all-position non-direct losses can erase direct semantic margin progress.  
Why needed: A11_07 shows indirect transfer exists; the finite-step theorem must bound it rather than ignore it.  
Support: `P_non-direct(t)` is lower-bounded and the net `d_K(t)` remains positive before threshold.  
Falsify: non-direct contribution is more negative than the direct contribution in most pre-threshold checkpoints.  
Insufficient: direct and non-direct gradients cannot be isolated reliably.

### 7.6 Curvature / finite-difference correction

Observed one-step margin increment:

$$
\Delta M_K^{obs}(t)=M_K^{ref}(t+1)-M_K^{ref}(t).
$$

Estimated second-order residual:

$$
R_2(t)=\eta G_{total}^{ref}(t)-\Delta M_K^{obs}(t).
$$

Curvature bound:

$$
\widehat B=\max_{t\in W}\frac{\max(R_2(t),0)}{\eta^2}.
$$

What it measures: whether the discrete update is close enough to the first-order gradient prediction.  
Why needed: the finite-step proposition subtracts `eta^2 B`.  
Support: residual is small enough that the certified increment remains positive.  
Falsify: residual dominates the first-order term.  
Insufficient: logging does not capture before/after states at the same checkpoint.

### 7.7 Local guards

Local target accuracy and loss:

$$
\operatorname{Acc}_{Y1}(t),\quad \operatorname{Acc}_{Y2}(t),\quad L_{Y1}(t),\quad L_{Y2}(t).
$$

What it measures: whether semantic progress is bought by damaging shared local targets.  
Why needed: clean efficiency cannot be a local prediction tradeoff.  
Support: local accuracy remains near 1.0 and CE does not regress beyond tolerance.  
Falsify: semantic margin gain coincides with local target degradation.  
Insufficient: local targets are not logged or are already unsolved.

## 8. Primary Metric

Primary metric is the certified positive pre-threshold margin increment:

$$
d_{min,K}^{W}=\min_{t\in W,t<T_\gamma}\left(\eta G_K^{ref}(t)-\eta^2\widehat B(t)-\widehat\epsilon_t\right),
$$

where `epsilon_t` is zero in decision-only frozen-head runs and is estimated from non-direct contributions in all-position runs.

Pass condition:

$$
d_{min,K}^{W}>0
$$

in at least `4/5` seeds for the supported direct informative condition, and the observed margin hitting time must not exceed the certified upper bound by more than one evaluation interval.

## 9. Secondary Metrics

- `G_K_ref(t)` and `G_K_current(t)` trajectories.
- `M_K_ref(t)` and `M_K_current(t)` trajectories.
- `T_gamma^M`, `T_0.9^H3`, and `T_0.9^Q`.
- `c_u(t)` and `D_u(t)` for head drift.
- `P_non-direct(t)` and `d_K(t)` for perturbation.
- local `Y1/Y2` CE and accuracy.
- leakage gates at step 0.

## 10. Pass / Fail / Insufficient Evidence

### Pass

A11_10 passes if all conditions hold:

1. no-leakage gates pass;
2. velocity persistence holds for direct informative conditions;
3. the certified increment `d_min,K^W` is positive;
4. margin threshold predicts native H3 or guarded recovery threshold;
5. head drift is either bounded or the reference theorem is explicitly shrunk by the measured drift;
6. all-position non-direct perturbation is bounded and does not dominate the direct term;
7. local target guards pass.

### Fail

A11_10 fails if any of the following occur after no-leakage gates pass:

1. positive step-0 velocity does not persist;
2. margin and native recovery are not calibrated;
3. head drift flips or erases reference direction before recovery;
4. non-direct all-position perturbation dominates the direct term;
5. observed margin hitting time violates the certified bound;
6. semantic progress requires local target degradation.

### Insufficient Evidence

Classify as insufficient evidence if:

1. step-0 leakage invalidates speed metrics;
2. neither margin nor recovery crosses threshold;
3. gradient decomposition is numerically unreliable;
4. before/after checkpointing is missing;
5. conclusions depend on adding unapproved seeds, loss weights, architectures, or K sweeps.

## 11. Failure Meaning

- Failure of velocity persistence means A11 remains a one-step semantic velocity result, not a finite-step efficiency theorem.
- Failure of margin-to-recovery calibration means `M_K` is not yet a valid learning-efficiency proxy.
- Failure of head drift bound means the theorem must be rewritten in current output geometry or restricted to frozen-head dynamics.
- Failure of perturbation bound means all-position training cannot be treated as direct semantic velocity plus small error.
- Failure of local guards means the apparent semantic efficiency is a tradeoff, not a clean objective advantage.

None of these failures would erase the already supported A11_06 / A11_08 / A11_09 first-order conclusions.

## 12. Artifact Map

Expected Research_System artifacts after running:

```text
Projects/from-attention-to-search/main/experiments/A11/A11_10_finite_step_semantic_efficiency/protocol.md
Projects/from-attention-to-search/main/experiments/A11/A11_10_finite_step_semantic_efficiency/summary.md
Projects/from-attention-to-search/main/experiments/A11/A11_10_finite_step_semantic_efficiency/detailed.md
Projects/from-attention-to-search/main/experiments/A11/A11_10_finite_step_semantic_efficiency/tables/finite_step_assumption_summary.csv
Projects/from-attention-to-search/main/experiments/A11/A11_10_finite_step_semantic_efficiency/tables/seed_level_hitting_time_bound.csv
Projects/from-attention-to-search/main/experiments/A11/A11_10_finite_step_semantic_efficiency/figures/velocity_persistence.png
Projects/from-attention-to-search/main/experiments/A11/A11_10_finite_step_semantic_efficiency/figures/margin_recovery_calibration.png
Projects/from-attention-to-search/main/experiments/A11/A11_10_finite_step_semantic_efficiency/figures/head_drift_bound.png
Projects/from-attention-to-search/main/experiments/A11/A11_10_finite_step_semantic_efficiency/figures/perturbation_bound.png
```

Expected remote experiment workspace artifacts:

```text
active/synthetic_data_understanding/scripts/run_a11_10_finite_step_semantic_efficiency.py
active/synthetic_data_understanding/configs/a11_10_finite_step_semantic_efficiency.json
runs/a11_10_finite_step_semantic_efficiency_<timestamp>/metrics.jsonl
runs/a11_10_finite_step_semantic_efficiency_<timestamp>/tables/
runs/a11_10_finite_step_semantic_efficiency_<timestamp>/figures/
```

## 13. Explicit Non-Goals

Do not run:

```text
broad K sweep
lambda sweep
architecture sweep
natural-language corpus experiment
MoE bridge
route-margin experiment
expert-utility experiment
```

The next deliverable after this protocol is a summary that answers only whether the finite-step theorem assumptions are supported, falsified, or insufficient.
