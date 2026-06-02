# 05_01_geometric_inhibition_anchor.md

## 0. Tiny Summary

**在 uniform multi-B synthetic 中，slot-stable router initialization 加上 geometric inhibition，是否能让 top-1 MoE routing 更稳定？**

我们比较两种 router：

1. ordinary dot-product router；
2. cosine-similarity router。

并比较三种设置：

1. random initialization；
2. slot-stable initialization；
3. slot-stable initialization + geometric inhibition。

本实验不处理 Zipfian，不处理真实数据，不提出完整 gating architecture。

Current result:

H0603a 已完成。slot-stable init 在 dot/cosine 下都提升 step-0 route-slot NMI；geometric inhibition 在 init 之外把 final route-slot NMI 稳定到 1.000，并且 target accuracy 不下降。confidence-only rival 被削弱，因为 selected gate confidence 上升同时 route-slot NMI 也大幅上升。

---

## 1. Problem Definition

### Parent problem

ordinary top-1 MoE router 在 NTP 训练下容易形成不稳定 routing，导致 feature-level expert specialization 失败。

### Current small question

在已有 multi-B synthetic 中，如果我们用 slot centroid 构造稳定初始化，并加入几何 inhibition，是否能让 routing 更稳定地对齐 slot feature？

### Decision question

几何 inhibition 是否在 slot-stable initialization 之外，提供额外的 routing stabilization？

并且：

cosine-similarity router 是否比 ordinary dot-product router 更适合这个几何初始化 / inhibition 机制？

### What this anchor does not decide

本 anchor 不回答：

1. 真实语言数据中怎么获得 slot label；
2. Zipfian 高频/低频问题；
3. expert utility 是否已经完全解决；
4. 完整 MoE gating architecture；
5. route-function binding 的最终形式。

---

## 2. Physical Prior

### Prior

top-1 selected-gate router 会强化 early assignment。

如果初始 routing 是随机的，那么 early wrong assignment 可能被训练放大。

slot-stable initialization 的作用是：

让不同 slot 的 hidden states 在 step 0 更稳定地进入不同 experts。

geometric inhibition 的作用是：

在训练过程中维持 slot-assigned expert 与其他 experts 之间的几何 margin，减少 routing drift / mixing。

cosine router 的可能优势是：

它主要看角度，不看 hidden-state norm，因此可能更接近“feature direction similarity”。

ordinary dot-product router 同时看方向和范数：

$$
w_e^\top h = |w_e||h|\cos(w_e,h)
$$

所以它可能受到 hidden norm 或 router row norm 的影响。

### Why this prior could be wrong

1. slot centroid 可能不能代表真正的 feature structure；
2. geometric inhibition 可能只让 routing heatmap 更整齐，但不提升 expert utility；
3. cosine router 去掉 norm 后，可能丢掉有用的 confidence / salience 信息；
4. center separation 可能让 experts 分开，但不能决定 token 应该去哪个 expert；
5. 如果 positive assignment 来自当前 top-1，而不是 slot prior，它会变成 lock-in amplifier。

---

## 3. Hypotheses

### H1: slot-stable initialization hypothesis

相比 random initialization，slot-stable initialization 会提高 step-0 route-slot alignment。

Prediction:

slot-init condition 的 step-0 route-slot NMI 高于 random baseline。

### H2: geometric inhibition hypothesis

相比 slot-stable initialization alone，slot-stable initialization + geometric inhibition 会提高 final route stability。

Prediction:

geometric inhibition condition 的 final route-slot NMI 更高，route drift 更小，seed stability 更好。

### H3: router type hypothesis

cosine router 可能比 dot-product router 更稳定地利用 slot prototype，因为它减少 hidden-state norm 和 router-row norm shortcut。

Prediction:

cosine + slot-init 在 step-0 或 final route-slot NMI 上优于 dot-product + slot-init，或者 route-token shortcut 更弱。

### Rival explanation 1: inhibition only sharpens confidence

geometric inhibition 可能只是提高 selected gate confidence，而没有改善 slot alignment。

Observable:

selected gate confidence 上升，但 route-slot NMI / Assign-Utility 不变。

### Rival explanation 2: center separation is not enough

expert centers 被拉开，但 token assignment 没有改善。

Observable:

router center pairwise cosine 下降，但 route-slot NMI 不提高。

### Rival explanation 3: cosine loses useful norm signal

cosine router 去掉 norm 后，可能降低 target accuracy 或 utility。

Observable:

route heatmap 改善，但 target-position accuracy 或 forced expert utility 下降。

---

## 4. Mathematical Modeling

Dataset:

multi-B synthetic sequence:

$$
x_{s,i} = [r_{\mathrm{start}}, C_s, B_i, Y_{s,i}, r_{\mathrm{end}}]
$$

where $s$ is slot / feature label, and $i$ is reused $B_i$ token identity.

Let $h_{s,i}$ be the hidden state at the $B_i$ position.

Slot prototype:

$$
\mu = \frac{1}{SN}\sum_{s=1}^{S}\sum_{i=1}^{N} h_{s,i}^{(0)}
$$

$$
\tilde p_s = \frac{1}{N}\sum_{i=1}^{N} h_{s,i}^{(0)} - \mu
$$

$$
p_s = \frac{\tilde p_s}{|\tilde p_s|_2+\epsilon}
$$

Assume $E=S$ for the first experiment, and assign expert $e=s$.

Positive assignment:

$$
a(s,i)=s
$$

Important:

$a(s,i)$ comes from slot prior, not from current top-1 routing.

### Dot-product router

$$
z_e(h)=w_e^\top h
$$

Slot-stable initialization:

$$
w_s(0)=\tau p_s
$$

### Cosine router

$$
z_e(h)=\tau \cdot \frac{w_e^\top h}{(|w_e|_2+\epsilon)(|h|_2+\epsilon)}
$$

Slot-stable initialization:

$$
w_s(0)=p_s
$$

### Selected-gate top-1 MoE

Router gate:

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

This keeps router gradients through $g_e(h)$.

Do not use:

$$
o(h)=E_{\arg\max_e z_e(h)}(h)
$$

because that blocks router gradients.

---

## 5. Geometric Inhibition

This anchor tests a small geometric inhibition mechanism with two components.

### Token-to-assigned-expert margin

For positive expert $e^+=a(s,i)$:

\mathbb{E}*{s,i}*
*\left[*
*\frac{1}{E-1}*
*\sum*{e\neq e^+}
\max\left(
0,;
m_{\mathrm{tok}}-\left[z_{e^+}(h_{s,i})-z_e(h_{s,i})\right]
\right)
\right]
$$

This says:

$$
z_{e^+}(h_{s,i}) \ge z_e(h_{s,i}) + m_{\mathrm{tok}}
$$

for every $e\neq e^+$.

### Router-center separation

Define normalized router center:

$$
u_e = \frac{w_e}{|w_e|_2+\epsilon}
$$

Center separation loss:

\frac{1}{E(E-1)}
\sum_{e\neq e'}
\max\left(
0,;
u_e^\top u_{e'}-\delta_{\mathrm{sep}}
\right)
$$

This encourages different router centers to remain separated.

### Total geometric inhibition loss

\lambda_{\mathrm{tok}}L_{\mathrm{tok}}
+
\lambda_{\mathrm{sep}}L_{\mathrm{sep}}
$$

Total training objective:

L_{\mathrm{NTP}}
+
L_{\mathrm{geo}}
$$

where:

\mathbb{E}*{s,i}*
*\left[*
*-\sum*{t=1}^{T-1}
\log P_{\theta}(x_{t+1}\mid x_{\leq t})
\right]
$$

Primary evaluation is at $B_i\rightarrow Y_{s,i}$.

---

## 6. Minimal Test

Use uniform multi-B only.

Compare six conditions:

| Condition | Router      | Init        | Geometric inhibition |
| --------- | ----------- | ----------- | -------------------- |
| C0        | dot-product | random      | no                   |
| C1        | dot-product | slot-stable | no                   |
| C2        | dot-product | slot-stable | yes                  |
| C3        | cosine      | random      | no                   |
| C4        | cosine      | slot-stable | no                   |
| C5        | cosine      | slot-stable | yes                  |

Primary metrics:

1. step-0 route-slot NMI;
2. final route-slot NMI;
3. route-token NMI;
4. route heatmap;
5. selected gate confidence;
6. router center pairwise cosine;
7. target-position accuracy;
8. seed stability.

Secondary metrics:

1. Assign-Utility;
2. forced expert loss diagonal.

Success:

The mechanism is supported if:

1. C1 > C0 on step-0 route-slot NMI;
2. C2 > C1 on final route-slot NMI or route stability;
3. C4 > C3 on step-0 route-slot NMI;
4. C5 > C4 on final route-slot NMI or route stability;
5. improvements do not reduce target-position accuracy.

Failure:

The mechanism is weakened if:

1. slot-stable init does not improve step-0 route-slot NMI;
2. geometric inhibition only increases confidence but not route-slot NMI;
3. center separation improves but token assignment does not;
4. cosine router improves route pattern but hurts accuracy / utility.

Insufficient evidence:

The experiment is insufficient if:

1. router gradient is blocked;
2. seed variance is too large;
3. route-slot NMI improves but Assign-Utility / forced expert utility clearly worsens;
4. prototype construction is unstable.

---

## 7. Next Decision

Result after H0603a:

slot-stable init works and geometric inhibition improves stability.

Next decision:

Use this as a meeting-facing clean mechanism result. The next research test should be Zipfian multi-B or less-oracle prototype assignment.

Claim boundary:

The current result supports routing stabilization under external slot assignment $a(s,i)=s$. It does not prove label-free specialization, real-data transfer, or complete expert utility specialization.

Cosine-router update:

Cosine is somewhat better than dot in init-only final route-slot NMI, but with geometric inhibition both dot and cosine reach 1.000. Therefore cosine is helpful but not necessary under this mechanism.
