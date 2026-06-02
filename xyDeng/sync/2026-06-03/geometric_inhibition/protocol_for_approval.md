# 05_01_geometric_inhibition_protocol.md

## 1. Goal

Does slot-stable initialization plus geometric inhibition make top-1 MoE routing more stable in uniform multi-B synthetic data?

It compares ordinary dot-product routing and cosine-similarity routing.

This protocol does not test Zipfian data, real data, expert warmup, or a full gating architecture.

---

## 2. Dataset

Use existing uniform multi-B synthetic.

Sequence:

$$
x_{s,i}=[r_{\mathrm{start}}, C_s, B_i, Y_{s,i}, r_{\mathrm{end}}]
$$

where:

* $s\in{1,\dots,S}$ is slot label;
* $i\in{1,\dots,N}$ is reused $B_i$ token identity;
* $Y_{s,i}$ is target token.

Sampling:

$$
(s,i)\sim \mathrm{Uniform}({1,\dots,S}\times{1,\dots,N})
$$

Primary routing position:

$B_i$ position.

---

## 3. Prototype Construction

Collect initial hidden states at the $B_i$ position:

$$
h_{s,i}^{(0)}
$$

Compute global mean:

$$
\mu=\frac{1}{SN}\sum_{s=1}^{S}\sum_{i=1}^{N}h_{s,i}^{(0)}
$$

Compute slot centroid:

$$
\tilde p_s=\frac{1}{N}\sum_{i=1}^{N}h_{s,i}^{(0)}-\mu
$$

Normalize:

$$
p_s=\frac{\tilde p_s}{|\tilde p_s|_2+\epsilon}
$$

Assume $E=S$ for this first protocol.

Positive assignment:

$$
a(s,i)=s
$$

Do not define $a(s,i)$ from current router top-1 output.

---

## 4. Router Implementations

### 4.1 Dot-product router

Logit:

$$
z_e(h)=w_e^\top h
$$

Random init condition:

use existing random initialization.

Slot-stable init condition:

$$
w_s(0)=\tau p_s
$$

Use no router bias in this protocol:

$$
b_e \equiv 0
$$

Implementation must use `bias=False`, not a trainable bias initialized to zero.

---

### 4.2 Cosine router

Logit:

$$
z_e(h)=\tau\cdot
\frac{w_e^\top h}
{(|w_e|_2+\epsilon)(|h|_2+\epsilon)}
$$

Random init condition:

use random $w_e$, normalized inside forward pass.

Slot-stable init condition:

$$
w_s(0)=p_s
$$

Use no router bias in this protocol:

$$
b_e \equiv 0
$$

Implementation must use `bias=False`, not a trainable bias initialized to zero.

---

## 5. Sparse Top-1 MoE Forward

For both router types:

Gate:

$$
g_e(h)=\frac{\exp(z_e(h))}{\sum_{k=1}^{E}\exp(z_k(h))}
$$

Hard top-1 mask:

$$
m_e(h)=\mathbf{1}\left[e=\arg\max_k z_k(h)\right]
$$

Selected-gate sparse output:

$$
o(h)=\sum_{e=1}^{E}m_e(h)g_e(h)E_e(h)
$$

Implementation requirement:

$m_e(h)$ can be stop-gradient, but $g_e(h)$ must remain differentiable.

Do not implement:

$$
o(h)=E_{\arg\max z}(h)
$$

because router gradients will be blocked.

---

## 6. Geometric Inhibition

Enable geometric inhibition only in C2 and C5.

### 6.1 Token-level margin

For each sample $(s,i)$:

$$
e^+=a(s,i)
$$

Margin loss:

\frac{1}{B}
\sum_{(s,i)}
\frac{1}{E-1}
\sum_{e\neq e^+}
\max\left(
0,;
m_{\mathrm{tok}}-\left[z_{e^+}(h_{s,i})-z_e(h_{s,i})\right]
\right)
$$

This enforces:

$$
z_{e^+}(h_{s,i})\ge z_e(h_{s,i})+m_{\mathrm{tok}}
$$

for all $e\neq e^+$.

### 6.2 Router-center separation

Define:

$$
u_e=\frac{w_e}{|w_e|_2+\epsilon}
$$

Separation loss:

\frac{1}{E(E-1)}
\sum_{e\neq e'}
\max\left(
0,;
u_e^\top u_{e'}-\delta_{\mathrm{sep}}
\right)
$$

### 6.3 Total loss

NTP loss:

\frac{1}{B}
\sum_{(s,i)}
\left[
-\sum_{t=1}^{T-1}
\log P_{\theta}(x_{t+1}\mid x_{\leq t})
\right]
$$

Geometric loss:

\lambda_{\mathrm{tok}}L_{\mathrm{tok}}
+
\lambda_{\mathrm{sep}}L_{\mathrm{sep}}
$$

Total loss:

$$
L=
L_{\mathrm{NTP}}+L_{\mathrm{geo}}
$$

Use fixed $m_{\mathrm{tok}}$, $\delta_{\mathrm{sep}}$, $\lambda_{\mathrm{tok}}$, and $\lambda_{\mathrm{sep}}$.

Do not sweep hyperparameters in this protocol.

---

## 7. Conditions

Run six conditions:

| Condition | Router      | Init        | Geometric inhibition |
| --------- | ----------- | ----------- | -------------------- |
| C0        | dot-product | random      | no                   |
| C1        | dot-product | slot-stable | no                   |
| C2        | dot-product | slot-stable | yes                  |
| C3        | cosine      | random      | no                   |
| C4        | cosine      | slot-stable | no                   |
| C5        | cosine      | slot-stable | yes                  |

Use same seeds, model size, training steps, and evaluation code for all conditions.

Recommended seeds:

$$
3
$$

---

## 8. Metrics

Primary metrics:

1. step-0 route-slot NMI;
2. final route-slot NMI;
3. route-token NMI;
4. route heatmap;
5. selected gate confidence;
6. target-position accuracy;
7. seed stability.

Geometry diagnostics:

1. router center pairwise cosine;
2. prototype-to-router cosine;
3. router drift from initialization.

Optional utility metrics:

1. Assign-Utility;
2. forced expert loss diagonal.

---

## 9. Decision Rules

### Slot-stable init supported if:

Dot-product:

$$
\mathrm{NMI}_{\mathrm{slot}}^{\mathrm{step0}}(C1)

>

\mathrm{NMI}_{\mathrm{slot}}^{\mathrm{step0}}(C0)
$$

Cosine:

$$
\mathrm{NMI}_{\mathrm{slot}}^{\mathrm{step0}}(C4)

>

\mathrm{NMI}_{\mathrm{slot}}^{\mathrm{step0}}(C3)
$$

### Geometric inhibition supported if:

Dot-product:

$$
\mathrm{NMI}_{\mathrm{slot}}^{\mathrm{final}}(C2)

>

\mathrm{NMI}_{\mathrm{slot}}^{\mathrm{final}}(C1)
$$

Cosine:

$$
\mathrm{NMI}_{\mathrm{slot}}^{\mathrm{final}}(C5)

>

\mathrm{NMI}_{\mathrm{slot}}^{\mathrm{final}}(C4)
$$

and target-position accuracy does not decrease.

### Cosine router supported if:

$$
C4 > C1
$$

or

$$
C5 > C2
$$

on route stability, without hurting target-position accuracy.

---

## 10. Failure Modes

This experiment weakens the mechanism if:

1. slot-stable initialization does not improve step-0 route-slot NMI;
2. geometric inhibition increases confidence but does not improve route-slot NMI;
3. route-token NMI dominates route-slot NMI;
4. cosine router hurts target-position accuracy;
5. router-center separation improves but token assignment does not.

This experiment is insufficient if:

1. router gradients are zero;
2. seed variance is too high;
3. prototype construction is unstable;
4. route-pattern metrics improve but utility metrics collapse.

---

## 11. Implementation Checks

Before running full conditions:

### Check 1: router gradient

After one backward pass:

$$
|\nabla W_r|_2>0
$$

### Check 2: inhibition gradient

For C2 and C5:

$$
|\nabla W_r^{\mathrm{geo}}|_2>0
$$

### Check 3: no current-top1 positive

Verify:

$$
a(s,i)=s
$$

not:

$$
a(s,i)=\arg\max_e z_e(h_{s,i})
$$

### Check 4: selected-gate path

Verify output uses:

$$
o(h)=\sum_e m_e(h)g_e(h)E_e(h)
$$

### Check 5: prototype sanity

Before training, save:

1. prototype cosine matrix;
2. step-0 route heatmap under slot-stable init;
3. route-slot NMI under slot-stable init.

---

## 12. Expected Outputs

For each condition:

1. config;
2. training log;
3. step-0 route heatmap;
4. final route heatmap;
5. metric table;
6. router gradient check;
7. geometry diagnostic table.

Final summary:

1. compare C0/C1/C2;
2. compare C3/C4/C5;
3. compare dot-product vs cosine;
4. decide whether next step is expert warmup, Zipfian, or route-function binding.
