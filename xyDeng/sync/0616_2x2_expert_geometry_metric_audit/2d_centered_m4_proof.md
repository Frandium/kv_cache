# 2D Centered M4 Proof Draft

Purpose:
This note turns the A06_E01 experimental observation into a small mathematical
claim that can be discussed before moving to high-dimensional features.

## 0. Tiny Summary

结论：原始 M4 分数 $q_e(x)=x^\top A_e^\top A_ex$ 不能保证随机高斯 experts 在二维均匀圆上自然 50/50 分发，因为两个 experts 的平均响应能量可能不同。

可证明版本：如果先减掉每个 expert 在均匀圆上的平均响应能量，得到中心化 M4 分数

$$
\tilde q_e(x)=x^\top A_e^\top A_ex-\frac{1}{2}\mathrm{tr}(A_e^\top A_e),
$$

那么对 $x\sim\mathrm{Unif}(S^1)$，两个 experts 的分发严格是 50/50；只要两个 experts 的方向性响应不完全相同，平均分数优势为正。

M3 top-1 也可以证明 50/50：它比较两个一维方向线上的投影平方；只要两条方向线不重合，均匀圆上两边各占一半，平均分数优势是 $2|\sin\alpha|/\pi$。

baseline 直觉：M1/M2 这类方向规则在二维均匀圆上通常也会 50/50，所以“均分”本身不是强证据；关键还要看方向是否稳定、margin 是否大、是否保留完整 expert 矩阵响应。

高维风险：二维中“去均值”自动给出 50/50，是因为二维 traceless quadratic form 的特征值必为 $\lambda,-\lambda$。高维中去均值只保证平均分数差为 0，不自动保证正负区域各占一半。

## 1. Setting

Two experts are represented by matrices:

$$
A_1,A_2\in\mathbb{R}^{2\times2}.
$$

For each expert define:

$$
G_e=A_e^\top A_e.
$$

M4 matrix response score is:

$$
q_e(x)=x^\top G_ex=\|A_ex\|^2.
$$

The ideal static routing rule assigns $x$ to the expert with larger score:

$$
g(x)=\arg\max_{e\in\{1,2\}} q_e(x).
$$

The input feature is uniformly distributed on the unit circle:

$$
x=(\cos\theta,\sin\theta),\qquad \theta\sim\mathrm{Unif}[0,2\pi).
$$

The score difference between expert 1 and expert 2 is:

$$
D(x)=q_1(x)-q_2(x)=x^\top Hx,
$$

where:

$$
H=G_1-G_2.
$$

Expert 1 receives $x$ when $D(x)>0$; expert 2 receives $x$ when $D(x)<0$.

## 2. Why Raw M4 Does Not Guarantee Uniform Load

Write:

$$
H=
\begin{pmatrix}
h_{11} & h_{12}\\
h_{12} & h_{22}
\end{pmatrix}.
$$

For $x=(\cos\theta,\sin\theta)$,

$$
D(\theta)
=h_{11}\cos^2\theta+2h_{12}\sin\theta\cos\theta+h_{22}\sin^2\theta.
$$

Using the double-angle identities:

$$
\cos^2\theta=\frac{1+\cos2\theta}{2},\qquad
\sin^2\theta=\frac{1-\cos2\theta}{2},\qquad
2\sin\theta\cos\theta=\sin2\theta,
$$

we get:

$$
D(\theta)
=\mu+a\cos2\theta+b\sin2\theta,
$$

where:

$$
\mu=\frac{h_{11}+h_{22}}{2}=\frac{1}{2}\mathrm{tr}(H),
$$

$$
a=\frac{h_{11}-h_{22}}{2},\qquad b=h_{12}.
$$

Equivalently:

$$
D(\theta)=\mu+\rho\cos(2\theta-\phi),
$$

where:

$$
\rho=\sqrt{a^2+b^2}.
$$

Interpretation:

- $\mu$ is the average energy bias between the two experts.
- $\rho$ is the direction-dependent expert difference.

The raw M4 assignment is exactly balanced only in special cases. If $\mu=0$ and
$\rho>0$, then positive and negative regions each occupy half of the circle.
But for random Gaussian $A_1,A_2$, the quantity

$$
\mu=\frac{1}{2}\left(\mathrm{tr}(G_1)-\mathrm{tr}(G_2)\right)
=\frac{1}{2}\left(\|A_1\|_F^2-\|A_2\|_F^2\right)
$$

is not guaranteed to be zero.

Therefore raw M4 cannot prove natural 50/50 load for random Gaussian experts.
This matches the A06_E01 C1 result: raw M4 had strong average score advantage
but natural load was imbalanced.

## 3. Centered M4

The average M4 response over the unit circle is:

$$
\mathbb{E}_{x\sim\mathrm{Unif}(S^1)}[x^\top G_ex]
=\frac{1}{2}\mathrm{tr}(G_e).
$$

So define centered M4:

$$
\tilde q_e(x)=x^\top G_ex-\frac{1}{2}\mathrm{tr}(G_e).
$$

The centered score difference is:

$$
\tilde D(x)=\tilde q_1(x)-\tilde q_2(x)
=x^\top H_0x,
$$

where:

$$
H_0=H-\frac{1}{2}\mathrm{tr}(H)I.
$$

By construction:

$$
\mathrm{tr}(H_0)=0.
$$

For a two-dimensional symmetric traceless matrix:

$$
H_0=
\begin{pmatrix}
a & b\\
b & -a
\end{pmatrix}.
$$

Therefore:

$$
\tilde D(\theta)=a\cos2\theta+b\sin2\theta
=\rho\cos(2\theta-\phi).
$$

## 4. Proof Of 50/50 Load In 2D

Assume $\rho>0$. Since $\theta$ is uniform on $[0,2\pi)$, the angle
$2\theta-\phi$ is also uniform modulo $2\pi$.

Thus:

$$
\mathbb{P}[\tilde D(\theta)>0]
=\mathbb{P}[\cos(2\theta-\phi)>0]
=\frac{1}{2}.
$$

Similarly:

$$
\mathbb{P}[\tilde D(\theta)<0]=\frac{1}{2}.
$$

So centered M4 gives exact uniform load:

$$
\mathrm{load}_1=\mathrm{load}_2=\frac{1}{2}.
$$

If $\rho=0$, then $\tilde D(\theta)=0$ for all $\theta$, so the rule is
degenerate and has no expert preference. This is the correct failure mode.

## 5. Proof Of Positive Average Score Advantage

The selected-expert score advantage is:

$$
\Delta(\theta)=|\tilde D(\theta)|.
$$

Since $\tilde D(\theta)=\rho\cos(2\theta-\phi)$,

$$
\mathbb{E}[\Delta(\theta)]
=\rho\mathbb{E}[|\cos T|],
$$

where $T\sim\mathrm{Unif}[0,2\pi)$.

Because:

$$
\mathbb{E}[|\cos T|]=\frac{2}{\pi},
$$

we have:

$$
\mathbb{E}[\Delta(\theta)]=\frac{2\rho}{\pi}.
$$

Therefore, whenever $\rho>0$, centered M4 has positive average score advantage.

For independent random Gaussian $A_1,A_2$, the event $\rho=0$ has probability
zero, because it requires the traceless parts of $G_1$ and $G_2$ to match
exactly. So centered M4 has positive average score advantage almost surely.

Important boundary:
This is not a uniform lower bound. Random experts can be very close in their
directional response with small probability, making $\rho$ and the margin small.

## 6. Relation To Calibration In The Experiment

The A06_E01 experiment also evaluated calibrated routing:

$$
g_\tau(x)=\arg\max_e(q_e(x)-\tau_e).
$$

For two experts, choosing:

$$
\tau_e=\frac{1}{2}\mathrm{tr}(G_e)
$$

turns raw M4 into centered M4.

So centered M4 can be viewed as the analytic version of load calibration for
uniform 2D inputs. The important distinction is:

- raw M4 measures total response energy and can be load-imbalanced;
- centered M4 removes average energy bias and keeps only direction-dependent
  expert preference.

This explains why C1 random Gaussian experts can have strong M4 score advantage
but imbalanced natural load, while calibrated M4 can recover balanced load
without destroying the score advantage.

## 7. What This Proves

Can claim in the 2D uniform-circle setting:

- Raw M4 does not guarantee natural 50/50 load under random Gaussian experts.
- Centered M4 gives exact 50/50 load whenever the centered score difference is
  nonzero.
- Centered M4 gives positive average score advantage almost surely for
  independent random Gaussian experts.
- The proof separates average expert energy from directional expert preference.

Cannot claim:

- A trained router will learn centered M4.
- The same exact 50/50 proof holds in high dimension.
- Positive score advantage implies real task utility.
- Random Gaussian experts have a fixed margin lower bound independent of the
  sampled initialization.

## 8. M3 Top-1 Projection Also Has A 2D 50/50 Proof

M3 top-1 uses only the most sensitive input direction of each expert.
Let those two unit directions be:

$$
u_1,u_2\in S^1.
$$

M3 top-1 score is:

$$
q_e^{\mathrm{top1}}(x)=|u_e^\top x|^2.
$$

Because the score is squared, $u_e$ and $-u_e$ define the same direction line.
Let the angle between the two direction lines be $\alpha\in[0,\pi)$.
Without loss of generality, set:

$$
u_1=(1,0),\qquad u_2=(\cos\alpha,\sin\alpha).
$$

For $x=(\cos\theta,\sin\theta)$:

$$
q_1^{\mathrm{top1}}(x)=\cos^2\theta,
$$

$$
q_2^{\mathrm{top1}}(x)=\cos^2(\theta-\alpha).
$$

The score difference is:

$$
D_{\mathrm{top1}}(\theta)
=\cos^2\theta-\cos^2(\theta-\alpha).
$$

Using $\cos^2 t=(1+\cos2t)/2$:

$$
D_{\mathrm{top1}}(\theta)
=-\sin\alpha\cdot \sin(2\theta-\alpha).
$$

If the two direction lines are distinct, then $\sin\alpha\neq 0$. Since
$2\theta-\alpha$ is uniform modulo $2\pi$, the sign of
$D_{\mathrm{top1}}(\theta)$ is positive on exactly half of the circle and
negative on exactly half of the circle.

Therefore:

$$
\mathbb{P}[q_1^{\mathrm{top1}}(x)>q_2^{\mathrm{top1}}(x)]=\frac{1}{2},
$$

and:

$$
\mathbb{P}[q_2^{\mathrm{top1}}(x)>q_1^{\mathrm{top1}}(x)]=\frac{1}{2}.
$$

The average selected-expert score advantage is:

$$
\mathbb{E}\left[|D_{\mathrm{top1}}(\theta)|\right]
=|\sin\alpha|\cdot\mathbb{E}[|\sin T|]
=\frac{2|\sin\alpha|}{\pi},
$$

where $T\sim\mathrm{Unif}[0,2\pi)$.

So M3 top-1 gives exact 50/50 load and positive average score advantage
whenever the two top input directions are not the same line.

Degenerate case:
If $u_1=\pm u_2$, then $\sin\alpha=0$ and
$q_1^{\mathrm{top1}}(x)=q_2^{\mathrm{top1}}(x)$ for every $x$. The rule has no
expert preference.

For independent random Gaussian expert matrices, the top right singular
directions are distinct with probability one when the top singular direction is
well-defined. Therefore M3 top-1 also has an almost-sure 2D 50/50 proof.

Important boundary:
This proof uses only direction lines. It ignores singular-value strength and
the second input direction. Therefore it supports M3 top-1 as a low-dimensional
or low-rank proxy, not as evidence that it captures the full expert matrix
response.

## 9. Why Direction Baselines Often Give 50/50 In 2D

The same geometric reason explains why M1 and M2 often show exact 50/50 load
in A06_E01.

M1 signed prototype uses:

$$
q_e^{\mathrm{M1}}(x)=x^\top m_e.
$$

For fixed non-identical prototypes $m_1,m_2$, the assignment is:

$$
x^\top m_1>x^\top m_2
\quad\Longleftrightarrow\quad
x^\top(m_1-m_2)>0.
$$

This is a half-plane through the origin, so a uniform circle is split exactly
50/50.

M2 unsigned prototype uses:

$$
q_e^{\mathrm{M2}}(x)=(x^\top m_e)^2.
$$

For fixed non-parallel prototype lines, this is the same geometry as M3 top-1:
two squared projections onto two lines. It also splits the uniform circle
50/50.

So the direction-based baselines can be expected to produce 50/50 load in the
ideal 2D uniform setting. But this is not enough to make them the core metric.

Reasons a baseline may fail or become weak:

- Degenerate directions: if two prototype lines or top-1 directions coincide,
  the scores are tied for every input.
- Undefined directions: equal singular values make the selected singular
  direction arbitrary, so the prototype or top-1 direction has no stable
  meaning.
- Sign sensitivity: M1 and the prototype construction can change when SVD
  signs are flipped, so the partition may be 50/50 but not stable.
- Missing matrix strength: M2 and M3 top-1 can split evenly while ignoring
  singular-value scale and secondary directions.
- Non-uniform features: the 50/50 proof relies on uniform symmetry of the
  circle or disk; a real feature distribution need not preserve the half-plane
  or line-projection symmetry.
- Finite samples: sampled load may deviate slightly from exact population load.

This is the main interpretation point:
50/50 load is easy for many symmetric direction rules in 2D. The harder
question is whether the rule represents expert-specific matrix response with a
meaningful score advantage and stable construction.

## 10. Why High Dimension Is Nontrivial

For $d$ dimensions, a natural centered score would be:

$$
\tilde q_e(x)=x^\top G_ex-\frac{1}{d}\mathrm{tr}(G_e),
\qquad x\sim\mathrm{Unif}(S^{d-1}).
$$

This guarantees:

$$
\mathbb{E}[\tilde q_1(x)-\tilde q_2(x)]=0.
$$

But it does not automatically guarantee:

$$
\mathbb{P}[\tilde q_1(x)>\tilde q_2(x)]=\frac{1}{2}.
$$

The reason is structural. In 2D, trace zero forces a symmetric matrix to have
eigenvalues $\lambda,-\lambda$, so the quadratic form has symmetric positive
and negative regions on the circle. In higher dimensions, trace zero only says
the eigenvalues sum to zero. It does not force the positive and negative
regions on the sphere to have equal measure.

Therefore the high-dimensional theory needs an additional argument. Candidate
directions:

1. prove approximate 50/50 under random-matrix assumptions, perhaps because the
   centered quadratic score difference becomes approximately symmetric;
2. add an explicit threshold or quantile calibration step to enforce load;
3. study whether the margin survives that calibration;
4. extend from two experts to $K$ experts using balanced assignment or
   thresholded score competition.

## 11. Next Discussion Question

The next theoretical decision is:

Should the high-dimensional candidate be raw M4, centered M4, or calibrated M4?

Current recommendation:
Use centered/calibrated M4 as the theoretical candidate, because raw M4 cannot
guarantee uniform load even in the 2D random Gaussian case.
