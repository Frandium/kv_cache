# 11_10 Finite-Step Semantic Efficiency Anchor

Status: completed; A11_10 full run partially supports the finite-step assumptions but does not close a tight finite-step efficiency theorem.  
Parent line: `11_long_horizon_mtp_objective`.  
Source package: `daily_research_reports/0710/mtp_long_horizon_semantic_efficiency_theory_closure/`.

## 0. Thinking Card

**Phenomenon:** A11 has a controlled first-order result: when next-K prediction includes an informative future horizon, the loss gives the current decision hidden state `h_T` a direct readout-effective semantic velocity. A11 does not yet prove that this local velocity yields a smaller finite-step hitting time.

**Mechanism guess:** A finite-step efficiency claim needs four additional assumptions: the positive velocity persists beyond step 0, the semantic margin predicts native recovery, output-head drift does not destroy the reference readout direction, and shared / indirect losses create only bounded negative perturbation.

**Key variables:** early variable `Z`; decision state `h_z=h_T(Z=z)`; informative horizon set `I_K`; aggregate semantic direction `v_z^(K)`; reference margin `M_K(t)`; hidden semantic velocity `G_K_hidden(t)`; perturbation `epsilon_t`; margin threshold `gamma`; hitting time `T_gamma(K)`.

**Causal relation:** If every pre-threshold step increases semantic margin by at least `eta g_K - eta^2 B - epsilon_t`, and the net lower bound is positive, then margin reaches threshold within a finite number of steps. If the margin threshold predicts native H3 or guarded recovery, this becomes a finite-step semantic efficiency statement.

**Observable metric:** Primary metric is the certified per-step lower-bound margin increment

$$
d_K(t)=\eta G_K^{ref}(t)-\eta^2\widehat B(t)-\widehat\epsilon_t.
$$

**Rival explanations:** Step-0 velocity may vanish after one update; native recovery may not follow margin; trainable output heads may drift away from reference directions; all-position shared / indirect losses may dominate the direct term; `Q` may recover through indirect transfer without native direct prediction.

**Decision:** A11_10 supports the sign-level mechanism: direct informative-horizon supervision gives persistent positive early semantic velocity, while no-information and masked/non-covering guards have zero direct velocity. It does not support a tight finite-step hitting-time theorem, because the certified lower-bound increment is often nonpositive or extremely loose after curvature correction.

## 1. Decision Question

Under explicit and testable local optimization assumptions, can the first-order semantic velocity result be converted into a finite-step semantic margin hitting-time bound for A11 controlled data?

This is the only decision question. All tests below serve this question.

## 2. Terminology / Definitions

| Term | Plain meaning | Concrete object or computation | Formula / unit | Why it matters | Cannot prove |
|---|---|---|---|---|---|
| early variable `Z` | delayed branch information | `Z in {1,...,m}` | categorical | semantic variable under control | natural-language semantics |
| informative horizon | future position containing new information about `Z` | `j in I_K` | set membership | source of direct semantic signal | every future target is useful |
| aggregate semantic direction | weighted sum of centered output rows | `v_z^(K)` | vector | defines reference margin | downstream utility |
| reference margin | semantic margin measured in fixed reference directions | `M_K(t)` | scalar | target of hitting-time theorem | native recovery without calibration |
| hidden semantic velocity | loss gradient projected onto semantic direction | `G_K_hidden(t)` | scalar | first-order source term | finite-step guarantee by itself |
| velocity persistence | positive velocity remains over an early window | lower bound on `G_K_hidden(t)` | scalar bound | turns step-0 theorem into multi-step assumption | global convergence |
| head drift | output rows move during training | current/reference direction correlation | cosine / norm | controls whether reference margin remains meaningful | no co-adaptation |
| perturbation | negative margin contribution from non-direct losses | `epsilon_t` | scalar | protects all-position theorem | absence of indirect learning |
| hitting time | first threshold crossing | `T_gamma(K)=inf{t:M_K(t)>=gamma}` | training steps | finite-step efficiency object | final asymptotic superiority |
| native H3 accuracy | model's own informative-horizon prediction | accuracy of `q_3(.|h_T)` on `S_Z` | 0 to 1 | tests model-usable readout | downstream route utility |
| guarded recovery `Q` | conservative decodability score | `min(A_decoder,A_probe,C_swap)` | 0 to 1 | protects against probe-only artifacts | direct supervision source |

## 3. Theory Structure

### 3.1 Layer 1: K=2 minimal primitive

Data:

$$
Y_1=A,\qquad Y_2=S_Z.
$$

Definitions:

$$
u_z=u_{S_z}^{(2)},\qquad \bar u=\frac1m\sum_z u_z.
$$

Readout margin:

$$
M_Z^{(2)}(h)=\frac1m\sum_z(u_z-\bar u)^\top h_z.
$$

Under branch-collapsed initialization and fixed output rows, K1 has only the shared target `A`. Its update is branch-common, so centered projections cancel:

$$
G_1^{margin}=0.
$$

K2 adds the informative term `CE(q_2(.|h_z),S_z)`. At the collapsed point:

$$
-\nabla_{h_z}L_2^{info}=\frac{\lambda_2}{m}(u_z-U_2^\top p),
$$

so the common softmax expectation `U_2^T p` cancels after centering and:

$$
G_2^{margin}=\frac{\lambda_2}{m^2}\sum_z\|u_z-\bar u\|^2>0.
$$

**Proof status:** theorem-level in the stated local model.

### 3.2 Layer 2: tau=m coverage theorem

Data:

$$
I(Z;Y_j\mid Y_{<j})=0,\qquad j<m,
$$

$$
Y_m=S_Z.
$$

Claim:

- If `K < m`, active decision-prefix losses contain no direct branch-informative target. Under folded branch states, the semantic margin velocity is zero.
- If `K >= m`, the active loss contains `CE(q_m(.|h_T),S_Z)`, so the same centered-output proof gives a positive semantic margin velocity whenever the centered output rows are nonzero.

This is the first-informative-horizon law.

**Proof status:** theorem-level in the same local model; empirically supported by A11_04 and A11_06 for the `tau=3` case.

### 3.3 Layer 3: general K vector-sum theorem

Definitions:

$$
\mathcal I_K=\{j\le K:Y_j\text{ contains new information about }Z\},
$$

$$
a_{j,z}=\lambda_j(u_{j,z}-\bar u_j),
$$

$$
v_z^{(K)}=\sum_{j\in\mathcal I_K}a_{j,z}.
$$

For the aggregate margin:

$$
M_K(h)=\frac1m\sum_z v_z^{(K)\top}h_z,
$$

the local first-order theorem gives:

$$
G_K=\frac1{m^2}\sum_z\|v_z^{(K)}\|^2.
$$

Adding a new informative horizon `r` changes the first-order velocity by:

$$
G_{K\cup r}-G_K
=\frac1{m^2}\sum_z(2v_z^{(K)\top}a_{r,z}+\|a_{r,z}\|^2).
$$

Consequences:

- A new uninformative horizon gives no direct semantic increment.
- A new aligned informative horizon strengthens semantic velocity.
- A new low/conflicting informative horizon can weaken the velocity.
- Larger `K` is not the causal object.

**Proof status:** theorem-level in the local fixed-head one-step model; empirically supported by A11_08 and A11_09 under controlled frozen-head decision-only dynamics.

### 3.4 Layer 4: finite-step efficiency proposition candidate

Define reference semantic margin:

$$
M_K(t)=\frac1m\sum_z v_z^{ref\top}h_z(t).
$$

Define margin hitting time:

$$
T_\gamma(K)=\inf\{t:M_K(t)\ge\gamma\}.
$$

**Conditional proposition.** Suppose that for all `t < T_gamma(K)` in an early window:

$$
M_K(t+1)\ge M_K(t)+\eta g_K-\eta^2B-\epsilon_t,
$$

with:

$$
g_K>0,\qquad \epsilon_t\le\bar\epsilon,
$$

and:

$$
\eta g_K-\eta^2B-\bar\epsilon>0.
$$

Then:

$$
T_\gamma(K)
\le
\left\lceil
\frac{\gamma-M_K(0)}
{\eta g_K-\eta^2B-\bar\epsilon}
\right\rceil.
$$

**Proof:** Sum the per-step lower bound until the cumulative lower bound reaches `gamma`.

**Status:** proposition is mathematically valid conditional on the stated assumptions. The assumptions are not yet proven for the Transformer training dynamics.

## 4. What Is Proved, Assumed, Empirical, Or Not A Theorem

| Statement | Status | Reason |
|---|---|---|
| K=2 gives positive semantic margin velocity while K1 gives zero | proved in local model | centered-output gradient calculation |
| `K < tau` gives no direct semantic velocity | proved in local model | all active targets are branch-shared at folded state |
| `K >= tau` opens semantic velocity | proved in local model | informative target `S_Z` enters active loss |
| general K velocity equals vector-sum squared norm | proved in local fixed-head one-step model | centered output directions sum linearly |
| next-K inclusion law | supported empirically | A11_09 frozen-head decision-only evidence |
| positive step-0 velocity persists over early training | assumption needing test | A11_08b supports but does not close finite-step theorem |
| semantic margin threshold predicts native H3 or `Q` threshold | assumption needing calibration | A11_07 shows relation but also indirect-transfer boundary |
| reference directions remain valid under trainable heads | assumption needing drift bound | A11_08b shows trainable heads can drift |
| shared / indirect losses are bounded perturbations | assumption needing decomposition | all-position indirect transfer is known to exist |
| `G_K_hidden > 0` unconditionally implies smaller `T_gamma` | not a theorem | missing persistence, drift, perturbation, and calibration assumptions |

## 5. Falsifiable Hypotheses For A11_10

These hypotheses are assumptions of the finite-step proposition, not separate research questions.

### H1: Velocity persistence

For direct informative conditions, early-window reference velocity remains positive:

$$
G_K^{ref}(t)=\frac1m\sum_z v_z^{ref\top}(-\nabla_{h_z}L_K(\theta_t))\ge g_{min}>0
$$

for most checkpoints before margin threshold.

Support: positive lower bound in at least `4/5` seeds with leakage guards passing.  
Falsify: velocity collapses to zero or becomes negative before margin growth.  
Insufficient: gradients are numerically unstable or all conditions saturate at step 0.

### H2: Margin-to-recovery calibration

There exists a threshold `gamma` such that crossing the reference margin predicts native informative-horizon prediction or guarded recovery:

$$
M_K(t)\ge\gamma\quad\Rightarrow\quad \operatorname{NativeH3}(t)\ge0.9
$$

or a predeclared `Q` threshold, with a small lag tolerance.

Support: margin threshold precedes or coincides with native H3 / `Q` threshold in most seeds.  
Falsify: margin crosses high values while native H3 and `Q` stay low, or recovery occurs without margin.  
Insufficient: neither margin nor recovery crosses threshold.

### H3: Head drift bound

For trainable heads, current semantic output directions remain positively aligned with reference directions:

$$
c_u(t)=
\frac{\sum_{j,z}(u_{j,z}^0-\bar u_j^0)^\top(u_{j,z}(t)-\bar u_j(t))}
{\left(\sum_{j,z}\|u_{j,z}^0-\bar u_j^0\|^2\right)^{1/2}
 \left(\sum_{j,z}\|u_{j,z}(t)-\bar u_j(t)\|^2\right)^{1/2}}.
$$

Support: `c_u(t)` remains positive and above a predeclared floor over the theorem window.  
Falsify: `c_u(t)` crosses zero or flips sign before margin/recovery.  
Insufficient: current geometry changes but both reference and current margins remain predictive, requiring a current-geometry theorem instead.

### H4: Shared / indirect perturbation bound

In all-position training, decompose the margin update into direct and non-direct parts:

$$
\Delta M_K(t)=\eta G_{direct}(t)+\eta P_{indirect/shared}(t)-\eta^2B_t.
$$

The perturbation term should satisfy:

$$
P_{indirect/shared}(t)\ge -p_{max}.
$$

Support: direct positive velocity dominates bounded negative perturbation.  
Falsify: non-direct loss frequently contributes a negative margin term larger than the direct term.  
Insufficient: the implementation cannot isolate direct and indirect gradients cleanly.

## 6. Minimal Tests

| Test | Measures | Primary expression | Why it is needed | Support | Falsify | Insufficient |
|---|---|---|---|---|---|---|
| Velocity persistence | Whether step-0 velocity survives early updates | `min_t G_K_ref(t)` over early window | Needed for per-step lower bound | positive in `4/5` seeds | nonpositive before threshold | noisy or saturated |
| Margin-to-recovery calibration | Whether margin is a valid hitting-time proxy | threshold relation between `M_K(t)` and native H3 / `Q` | Converts margin hitting time into semantic recovery | threshold predicts recovery | margin and recovery decouple | no threshold crossing |
| Head drift bound | Whether reference theorem remains valid | `c_u(t)` and drift norm | Bounds reference-to-current geometry mismatch | positive correlation floor | sign flip / drift dominates | need current-geometry theorem |
| Shared / indirect perturbation | Whether all-position losses can be bounded | `P_indirect/shared(t)` lower bound | Protects finite-step theorem in all-position training | bounded negative term | perturbation dominates | decomposition unavailable |

No broad ablation is included: no architecture sweep, no loss-weight sweep, no natural-language task, and no MoE bridge.

## 7. Current Evidence Entering This Anchor

- A11_06 supports the first-order hidden readout-margin mechanism: `G_M_hidden` is near zero for K1/K2 and positive for K3 under no leakage.
- A11_07 supports direct decision-prefix horizon-3 supervision as the stronger all-position path, but also shows that `Q` alone is not direct-supervision evidence.
- A11_08 supports the general-K vector-sum formula: aligned output directions give much larger hidden semantic velocity than low/conflicting directions under the same K and the same informative horizons.
- A11_08b shows the velocity mechanism is not only a step-0 artifact under frozen heads, but trainable heads drift enough that current geometry must be recorded.
- A11_09 supports next-K inclusion law: velocity turns on at the first informative horizon, shared added horizons add no one-step increment, aligned informative horizons amplify, and low/conflict informative horizons weaken.
- A11_10 full run supports velocity persistence and the zero-velocity guard, partially supports margin-to-recovery calibration and all-position perturbation control, and weakens the strong finite-step theorem claim because the empirical certified hitting-time bound is loose or nonpositive in many seeds.

Key A11_10 result records:

```text
Projects/from-attention-to-search/main/experiments/A11/A11_10_finite_step_semantic_efficiency/summary.md
Projects/from-attention-to-search/main/experiments/A11/A11_10_finite_step_semantic_efficiency/detailed.md
```

Compact evidence:

| Audit | Result | Interpretation |
|---|---|---|
| zero-velocity guard | `shared_only_k4` has `G_total_min=0`, final `Q=0.25`, final `M_Z_ref=0` | no semantic horizon gives no direct semantic velocity |
| direct velocity persistence | direct informative conditions have positive early `G_total_min` in `5/5` seeds | sign-level mechanism persists beyond step 0 |
| all-position direct path | `K3_active` and `K2_plus_direct_H3` reach final native H3 `1.0`, final `Q=1.0` | direct H3 supervision gives model-owned readout recovery |
| all-position indirect controls | `K2_active` and `K3_mask_direct_H3` recover nontrivial `Q` but final native H3 remains `0.0` | `Q` alone is not direct-supervision evidence |
| finite-step certificate | support rates are partial; `d_cert_total` often nonpositive or gives huge bounds | no tight finite-step theorem yet |

## 8. Pass / Fail / Insufficient Decision Rule And A11_10 Outcome

Original pass rule: pass A11_10 only if all four assumption audits pass:

1. `G_K_ref(t)` has a positive early-window lower bound.
2. `M_K(t)` threshold predicts native H3 or guarded recovery threshold.
3. reference/current output geometry remains positively aligned or the shrinkage bound is explicitly applied.
4. shared / indirect perturbation is bounded so that the net lower-bound increment is positive.

Observed A11_10 outcome:

| Assumption | Outcome | Evidence |
|---|---|---|
| velocity persistence | supported for direct informative conditions | pass rate `1.0`; zero guard passes |
| margin-to-recovery calibration | partial | all-position direct conditions pass more often; decision-only and low/conflict are seed-dependent |
| head drift bound | supported as an audit condition | frozen drift is zero; trainable reference/current alignment stays positive, but drift is nonzero |
| shared / indirect perturbation | partial with important boundary | direct all-position conditions recover native H3; indirect-only controls recover `Q` without native H3 |
| finite-step hitting-time theorem | not closed | certified bound is loose or nonpositive in many seeds |

Fail if any of the following occurs after leakage guards pass:

1. step-0 positive velocity disappears before margin growth;
2. margin does not predict native H3 or `Q`;
3. head drift flips the reference direction;
4. indirect/shared perturbation dominates the direct semantic term;
5. the observed hitting time violates the certified bound.

Insufficient evidence if leakage gates fail, no condition crosses the relevant threshold, gradient decomposition is numerically unreliable, or finite-step comparison depends on unapproved hyperparameter sweeps.

## 9. Claim Boundary

Can claim after A11_10:

```text
In the controlled A11 setup, direct informative-horizon supervision gives persistent positive early semantic velocity for h_T. This explains why direct all-position conditions recover native H3 and guarded Q more reliably than masked or non-covering controls.
```

Cannot claim after A11_10:

```text
A11_10 proves a tight finite-step learning-efficiency theorem.
MTP is more efficient in natural language.
NTP cannot learn the delayed variable.
Larger K is generally better.
All-position indirect transfer follows the clean decision-only law.
MoE routing preservation or expert utility follows.
Q alone proves direct semantic supervision.
```

## 10. Minimal Next Action

Do not open a broader experiment. The next theoretical decision is to refine the finite-step theorem:

1. prove a sharper local smoothness / curvature bound that makes the certified increment positive and useful; or
2. state a weaker theorem: persistent positive semantic velocity plus empirical margin calibration predicts early recovery in the controlled family, without claiming a tight hitting-time certificate.
