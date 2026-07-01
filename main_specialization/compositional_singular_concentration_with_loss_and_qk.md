# Compositional Language Learning Can Concentrate Features Into High-Gain Singular Subspaces

## Abstract

Language has a nested and compositional structure. A short phrase can express a simple feature, and a longer phrase can preserve that feature while adding new constraints. For example, "city" can become "beautiful city", then "NYC is a beautiful city", then "NYC is a beautiful city in winter". Each longer expression is not independent of the shorter one. It reuses earlier meaning and adds a new component.

This draft studies a mechanistic consequence of that structure. If many simple and intermediate features are learned first, then the model parameters can develop a small number of high-gain singular directions. Once those directions have large singular values, later features have an optimization bias to reuse them, because a small component along a large-singular-value direction produces a large change in logits or attention scores. This can make learning efficient, but it can also mix many different features into a shared low-dimensional subspace, creating interference between common and long-tail features.

We give a formal gradient argument for this bias and support it with controlled transformer experiments on nested-prefix data. In the experiment, a one-layer causal transformer learns sequences such as \(A\), \(AB\), \(ABC\), \(ABCD\), up to \(ABCDEFG.\). The model predicts all structured continuations with \(100\%\) accuracy, while the newly added "extra" feature at each composition step is mostly concentrated in the top singular directions of \(W_{\mathrm{out}}\), \(W_Q\), and \(W_K\), but not \(W_V\). This suggests that composition uses a compact high-gain subspace for routing and prediction, while value vectors preserve token-specific content more diffusely.

The working conclusion is:

$$
\boxed{
\text{compositional learning creates reusable high-gain subspaces, and those subspaces attract later features.}
}
$$

This is useful for learning but dangerous for long-tail features, because feature sharing can become feature interference.

## 1. Research Question

Autoregressive language models are trained by next-token prediction:

$$
p_\theta(x_{t+1}\mid x_{\leq t}),
$$

with cross-entropy loss:

$$
L_t=-\log p_\theta(x_{t+1}\mid x_{\leq t}).
$$

The model is not directly trained to build a clean semantic hierarchy. It is trained to reduce prediction loss. Therefore, if semantic features emerge, they should be explainable through how they help reduce loss.

The question in this draft is:

**Why do compositional features tend to concentrate in a small number of high-gain singular-vector directions?**

The second question is:

**Why can this concentration cause interference between common features and long-tail features?**

The motivating intuition is that language learning is progressive:

$$
\text{simple feature}
\rightarrow
\text{composed feature}
\rightarrow
\text{more complex composed feature}.
$$

Simple and frequent features are encountered many times. They are learned early and repeatedly updated. Their gradients can create dominant singular directions in model parameter matrices. Later features then tend to reuse those directions because they are already high-gain.

This creates a tradeoff:

$$
\text{shared low-dimensional subspace}
\quad
\Longrightarrow
\quad
\text{efficient reuse}
\quad
\text{and}
\quad
\text{feature interference}.
$$

## 2. Language Analogy

The synthetic experiment uses nested symbolic prefixes:

$$
A,\quad AB,\quad ABC,\quad ABCD,\quad ABCDE,\quad ABCDEF,\quad ABCDEFG,\quad ABCDEFG.
$$

This is not natural language. It is a controlled analogy for one property of language:

$$
\text{longer meaning}
=
\text{shorter meaning}
+
\text{new constraint}.
$$

For natural language, one might have:

$$
\text{city}
$$

then:

$$
\text{beautiful city}
$$

then:

$$
\text{NYC is a beautiful city}
$$

then:

$$
\text{NYC is a beautiful city in winter}.
$$

Each phrase preserves part of the earlier meaning while adding something new. This motivates the decomposition:

$$
u_{AB}
=
u_A
+
\Delta_{A\to AB},
$$

$$
u_{ABC}
=
u_{AB}
+
\Delta_{AB\to ABC},
$$

$$
u_{ABCD}
=
u_{ABC}
+
\Delta_{ABC\to ABCD}.
$$

Here \(u_X\) is the hidden direction for prefix \(X\), and \(\Delta_{X\to Y}\) is the extra feature added when moving from \(X\) to \(Y\).

The goal is to understand the geometry of these extra features:

1. Are they all the same direction?
2. Are they independent directions?
3. Do they live in a shared low-dimensional high-gain subspace?
4. Do they interfere with each other because of that sharing?

## 3. Core Conjecture

The central conjecture is:

$$
\boxed{
\text{nested compositional learning creates high-gain singular subspaces that later features reuse.}
}
$$

More explicitly:

1. Simple and frequent features are learned first.
2. Their gradients repeatedly update similar model directions.
3. Those directions become large-singular-value directions.
4. Later, more complex or long-tail features receive stronger gradient along those high-gain directions.
5. Therefore, later features tend to reuse those directions instead of spreading uniformly across all available dimensions.
6. This reuse makes learning efficient, but it also couples features together.

This coupling is the source of feature interference.

## 4. Formal Model

Consider a hidden representation:

$$
h\in\mathbb{R}^d.
$$

Let a model matrix map the hidden state to logits:

$$
z = Wh.
$$

Let the singular value decomposition of \(W\) be:

$$
W = U\Sigma V^\top
=
\sum_i \sigma_i u_i v_i^\top.
$$

Here:

- \(v_i\) is a hidden-side singular vector;
- \(u_i\) is an output-side singular vector;
- \(\sigma_i\) is the singular value, or gain, of direction \(i\).

Expand the hidden feature in the right singular-vector basis:

$$
h=\sum_i c_i v_i.
$$

Then:

$$
Wh
=
\sum_i \sigma_i c_i u_i.
$$

This already shows the forward-pass gain:

$$
\boxed{
\text{the same hidden coefficient } c_i \text{ has output effect proportional to } \sigma_i.
}
$$

If \(\sigma_1\gg\sigma_{20}\), then a component along \(v_1\) changes the output much more than the same-sized component along \(v_{20}\).

## 5. Gradient Proof: New Features Are Biased Toward High-Gain Directions

For cross-entropy loss:

$$
L(z,y)=-\log p_y,
\qquad
p=\operatorname{softmax}(z).
$$

The gradient with respect to logits is:

$$
\nabla_z L = p-e_y.
$$

The gradient with respect to the hidden representation is:

$$
\nabla_h L
=
W^\top \nabla_z L.
$$

Substitute the SVD:

$$
\nabla_h L
=
V\Sigma U^\top (p-e_y).
$$

Define:

$$
\beta_i = u_i^\top(p-e_y).
$$

Then:

$$
\nabla_h L
=
\sum_i \sigma_i\beta_i v_i.
$$

A gradient descent update to the hidden feature is:

$$
\Delta h
=
-\eta\nabla_h L
=
-\eta\sum_i\sigma_i\beta_i v_i.
$$

Therefore, the update coefficient along singular direction \(v_i\) is:

$$
\Delta c_i
=
-\eta\sigma_i\beta_i.
$$

Taking magnitude:

$$
|\Delta c_i|
=
\eta\sigma_i|\beta_i|.
$$

This proves the key local statement:

$$
\boxed{
\text{for the same error projection }|\beta_i|,\text{ a larger }\sigma_i\text{ gives a larger feature update.}
}
$$

If the error vector is not specially orthogonal to the top output singular directions, then:

$$
\mathbb{E}\left[(\Delta c_i)^2\right]
=
\eta^2\sigma_i^2\mathbb{E}\left[\beta_i^2\right].
$$

If:

$$
\mathbb{E}\left[\beta_i^2\right]\approx \text{constant},
$$

then:

$$
\mathbb{E}\left[(\Delta c_i)^2\right]
\propto
\sigma_i^2.
$$

So the learning energy of a new feature is biased toward large-singular-value directions.

The exact condition is:

$$
\boxed{
\Delta c_i \propto \sigma_i\langle u_i,p-e_y\rangle.
}
$$

A high singular value alone is not sufficient. The output error must also have nonzero projection onto the corresponding output singular vector. But if a large-singular-value direction helps reduce the current loss, gradient descent naturally favors it.

### 5.1 End-to-End Loss Decrease Along A Singular Direction

The argument above shows that the hidden update is larger along high-singular-value directions. But this is not yet the full end-to-end reason why optimization should prefer those directions. The missing step is to show that increasing the useful component along such a direction actually decreases the cross-entropy loss.

Consider a small perturbation of the hidden state along one right singular vector:

$$
\delta h = \epsilon v_i.
$$

Here \(\delta h\) is the change in the hidden representation, \(\epsilon\) is the perturbation size, and \(v_i\) is the hidden-side singular vector of \(W\). Since the logits are \(z=Wh\), the induced logit change is:

$$
\delta z
=
W\delta h
=
\epsilon Wv_i
=
\epsilon \sigma_i u_i.
$$

Here \(\delta z\) is the change in logits, \(\sigma_i\) is the singular value, and \(u_i\) is the output-side singular vector. Therefore, the same hidden perturbation size \(\epsilon\) produces a logit change whose magnitude is proportional to \(\sigma_i\).

Using the first-order Taylor expansion of the loss around the current logits gives:

$$
\delta L
\approx
\left\langle \nabla_z L,\delta z\right\rangle.
$$

For softmax cross entropy, \(\nabla_z L=p-e_y\). Substituting the logit perturbation gives:

$$
\delta L
\approx
\epsilon \sigma_i\left\langle p-e_y,u_i\right\rangle.
$$

This equation is the end-to-end descent criterion. If \(\left\langle p-e_y,u_i\right\rangle>0\), then moving in the direction \(-v_i\) decreases the loss. If \(\left\langle p-e_y,u_i\right\rangle<0\), then moving in the direction \(+v_i\) decreases the loss. Therefore, the best first-order loss decrease obtainable by moving only along singular direction \(v_i\) with perturbation norm \(\epsilon\) is:

$$
\max_{\delta h\in\operatorname{span}(v_i),\ \|\delta h\|=\epsilon}
(-\delta L)
\approx
\epsilon\sigma_i\left|\left\langle p-e_y,u_i\right\rangle\right|.
$$

This proves the stronger statement:

$$
\boxed{
\text{for the same perturbation size and the same error alignment, a larger }\sigma_i\text{ gives a larger first-order loss decrease.}
}
$$

Thus the high-gain direction is not merely a direction with a larger representation update. It is a direction with larger descent efficiency, provided that the output error has nonzero alignment with the corresponding output-side singular vector.

We can connect this directly to the gradient update derived above. The gradient step coefficient along \(v_i\) is:

$$
\Delta c_i
=
-\eta\sigma_i\beta_i,
\qquad
\beta_i=\left\langle u_i,p-e_y\right\rangle.
$$

The induced hidden perturbation along this direction is:

$$
\delta h_i
=
\Delta c_i v_i.
$$

The corresponding logit perturbation is:

$$
\delta z_i
=
W\delta h_i
=
-\eta\sigma_i^2\beta_i u_i.
$$

The first-order loss change caused by this singular-direction component is therefore:

$$
\Delta L_i
\approx
\left\langle p-e_y,\delta z_i\right\rangle
=
-\eta\sigma_i^2\beta_i^2.
$$

Equivalently:

$$
\boxed{
\Delta L_i
\approx
-\eta\sigma_i^2\left\langle u_i,p-e_y\right\rangle^2.
}
$$

Here \(\Delta L_i\) is the first-order loss change contributed by the gradient step component along \(v_i\). This formula closes the loop: high-singular-value directions are favored because, when aligned with the current prediction error, they produce larger cross-entropy decrease per gradient step.

The exact claim should therefore be stated as:

$$
\boxed{
\text{future features are attracted to high-singular-value directions that are also aligned with the current prediction error.}
}
$$

This condition is important. A large singular value by itself does not guarantee loss reduction. If \(\left\langle p-e_y,u_i\right\rangle=0\), then changing the hidden state along \(v_i\) produces no first-order loss decrease. The mechanism requires both gain and usefulness: \(\sigma_i\) supplies gain, while \(\left\langle p-e_y,u_i\right\rangle\) supplies task alignment.

This gives the end-to-end optimization story. Early frequent features repeatedly reduce loss through certain output directions. Repeated updates increase the corresponding singular values. Later related features often produce prediction errors with nonzero projection onto those same output-side directions. Because those directions now have large \(\sigma_i\), a small hidden movement along the corresponding \(v_i\) creates a large logit correction and a large first-order loss decrease. Therefore, gradient-based learning tends to reuse them.

## 6. Softmax Makes The Q/K Bias Stronger

For attention:

$$
q_t=W_Qh_t,
\qquad
k_i=W_Kh_i.
$$

The attention score is:

$$
s_{ti}
=
\frac{q_t^\top k_i}{\sqrt d}.
$$

The attention weight is:

$$
\alpha_{ti}
=
\frac{\exp(s_{ti})}{\sum_{j\leq t}\exp(s_{tj})}.
$$

For two positions \(i\) and \(j\):

$$
\frac{\alpha_{ti}}{\alpha_{tj}}
=
\exp(s_{ti}-s_{tj}).
$$

So softmax amplifies score differences. A score gap of \(2\) gives:

$$
\exp(2)\approx 7.39.
$$

A score gap of \(5\) gives:

$$
\exp(5)\approx 148.4.
$$

This explains why a useful attention-score gap can dominate routing once it exists. But by itself, this does not yet prove why gradient-based learning chooses large-singular-value directions in \(W_Q\) or \(W_K\) to create that gap. We need the same end-to-end loss-descent argument as in the output-logit case.

### 6.1 Loss Descent Through A Query Singular Direction

Let the attention output at position \(t\) be:

$$
o_t
=
\sum_{i\leq t}\alpha_{ti}r_i,
\qquad
r_i=W_Vh_i.
$$

Here \(o_t\) is the attention output, \(\alpha_{ti}\) is the attention weight from query position \(t\) to key position \(i\), and \(r_i\) is the value vector carried from position \(i\).

Let the downstream loss be \(L\). Define the score gradient:

$$
g_{ti}
=
\frac{\partial L}{\partial s_{ti}}.
$$

For softmax attention, this gradient is:

$$
g_{ti}
=
\alpha_{ti}\left\langle \frac{\partial L}{\partial o_t}, r_i-o_t\right\rangle.
$$

Here \(\partial L/\partial o_t\) is the downstream gradient arriving at the attention output. The term \(r_i-o_t\) measures how different value \(r_i\) is from the current attention average. This formula gives the exact usefulness condition for a score. If increasing \(s_{ti}\) would make the output more aligned with the downstream gradient direction, then \(g_{ti}\) is nonzero and gradient descent will change that score.

Now decompose the query matrix:

$$
W_Q
=
U_Q\Sigma_QV_Q^\top
=
\sum_a \sigma_a^{Q}u_a^{Q}(v_a^{Q})^\top.
$$

Here \(v_a^{Q}\) is a hidden-side singular vector of \(W_Q\), \(u_a^{Q}\) is a query-side singular vector, and \(\sigma_a^{Q}\) is the query gain in direction \(a\).

Consider a small perturbation of the query-side hidden state along \(v_a^{Q}\):

$$
\delta h_t
=
\epsilon v_a^{Q}.
$$

The query changes by:

$$
\delta q_t
=
W_Q\delta h_t
=
\epsilon\sigma_a^{Q}u_a^{Q}.
$$

The induced score change for key position \(i\) is:

$$
\delta s_{ti}
=
\frac{(\delta q_t)^\top k_i}{\sqrt d}
=
\frac{\epsilon\sigma_a^{Q}}{\sqrt d}\left\langle u_a^{Q},k_i\right\rangle.
$$

Therefore, the first-order loss change is:

$$
\delta L
\approx
\sum_{i\leq t}g_{ti}\delta s_{ti}
=
\epsilon\sigma_a^{Q}\left\langle u_a^{Q},
\frac{1}{\sqrt d}\sum_{i\leq t}g_{ti}k_i
\right\rangle.
$$

Define the query-side score-gradient vector:

$$
\gamma_t^{Q}
=
\frac{1}{\sqrt d}\sum_{i\leq t}g_{ti}k_i.
$$

This is exactly the gradient of the loss with respect to the query vector:

$$
\gamma_t^{Q}
=
\frac{\partial L}{\partial q_t}.
$$

Thus:

$$
\delta L
\approx
\epsilon\sigma_a^{Q}\left\langle u_a^{Q},\gamma_t^{Q}\right\rangle.
$$

The best first-order loss decrease obtainable by moving only along query hidden-side singular direction \(v_a^{Q}\) with perturbation norm \(\epsilon\) is:

$$
\max_{\delta h_t\in\operatorname{span}(v_a^{Q}),\ \|\delta h_t\|=\epsilon}
(-\delta L)
\approx
\epsilon\sigma_a^{Q}\left|\left\langle u_a^{Q},\gamma_t^{Q}\right\rangle\right|.
$$

This proves the query-side selection rule:

$$
\boxed{
\text{large query singular values are favored only when their query-side singular vectors align with the score-gradient direction.}
}
$$

The corresponding gradient step gives an even clearer descent-efficiency formula. The gradient with respect to \(h_t\) through \(W_Q\) is:

$$
\nabla_{h_t}^{Q}L
=
W_Q^\top\gamma_t^{Q}
=
\sum_a \sigma_a^{Q}\left\langle u_a^{Q},\gamma_t^{Q}\right\rangle v_a^{Q}.
$$

The gradient step component along \(v_a^{Q}\) is:

$$
\Delta h_{t,a}^{Q}
=
-\eta\sigma_a^{Q}\left\langle u_a^{Q},\gamma_t^{Q}\right\rangle v_a^{Q}.
$$

Substituting this component back into the first-order loss change gives:

$$
\Delta L_a^{Q}
\approx
-\eta (\sigma_a^{Q})^2
\left\langle u_a^{Q},\gamma_t^{Q}\right\rangle^2.
$$

This is the end-to-end reason high-gain query directions are selected. When two query singular directions have comparable alignment with the current score-gradient vector, the direction with larger \(\sigma_a^{Q}\) gives a larger first-order loss decrease.

### 6.2 Loss Descent Through A Key Singular Direction

The same argument applies to the key matrix. Decompose:

$$
W_K
=
U_K\Sigma_KV_K^\top
=
\sum_b \sigma_b^{K}u_b^{K}(v_b^{K})^\top.
$$

Perturb the hidden state at key position \(i\) along \(v_b^{K}\):

$$
\delta h_i
=
\epsilon v_b^{K}.
$$

Then:

$$
\delta k_i
=
W_K\delta h_i
=
\epsilon\sigma_b^{K}u_b^{K}.
$$

For a fixed query position \(t\), the score change is:

$$
\delta s_{ti}
=
\frac{q_t^\top\delta k_i}{\sqrt d}
=
\frac{\epsilon\sigma_b^{K}}{\sqrt d}\left\langle q_t,u_b^{K}\right\rangle.
$$

If the same key vector affects multiple query positions, the first-order loss change sums over all such queries:

$$
\delta L
\approx
\epsilon\sigma_b^{K}
\left\langle u_b^{K},
\frac{1}{\sqrt d}\sum_t g_{ti}q_t
\right\rangle.
$$

Define the key-side score-gradient vector:

$$
\gamma_i^{K}
=
\frac{1}{\sqrt d}\sum_t g_{ti}q_t.
$$

This is the gradient of the loss with respect to the key vector:

$$
\gamma_i^{K}
=
\frac{\partial L}{\partial k_i}.
$$

Therefore:

$$
\delta L
\approx
\epsilon\sigma_b^{K}\left\langle u_b^{K},\gamma_i^{K}\right\rangle.
$$

The best first-order loss decrease obtainable by moving only along key hidden-side singular direction \(v_b^{K}\) with perturbation norm \(\epsilon\) is:

$$
\max_{\delta h_i\in\operatorname{span}(v_b^{K}),\ \|\delta h_i\|=\epsilon}
(-\delta L)
\approx
\epsilon\sigma_b^{K}\left|\left\langle u_b^{K},\gamma_i^{K}\right\rangle\right|.
$$

The gradient-step contribution satisfies:

$$
\Delta L_b^{K}
\approx
-\eta (\sigma_b^{K})^2
\left\langle u_b^{K},\gamma_i^{K}\right\rangle^2.
$$

So the key-side selection rule is:

$$
\boxed{
\text{large key singular values are favored only when their key-side singular vectors align with the score-gradient direction.}
}
$$

Again, singular value alone is not sufficient. A large \(\sigma_b^{K}\) is useful only if \(\langle u_b^{K},\gamma_i^{K}\rangle\neq 0\). The optimization preference is jointly determined by gain and task alignment.

### 6.3 Why Softmax Strengthens The Selection

The previous two subsections show why high-singular-value Q/K directions are selected by loss descent. Softmax then strengthens the effect because useful score changes are converted into multiplicative routing changes.

For two positions \(i\) and \(j\), define the score margin:

$$
m_{tij}
=
s_{ti}-s_{tj}.
$$

The attention odds ratio is:

$$
\frac{\alpha_{ti}}{\alpha_{tj}}
=
\exp(m_{tij}).
$$

Suppose a query hidden perturbation along \(v_a^{Q}\) changes this margin. Since:

$$
\delta s_{ti}
=
\frac{\epsilon\sigma_a^{Q}}{\sqrt d}\left\langle u_a^{Q},k_i\right\rangle,
$$

and:

$$
\delta s_{tj}
=
\frac{\epsilon\sigma_a^{Q}}{\sqrt d}\left\langle u_a^{Q},k_j\right\rangle,
$$

the margin change is:

$$
\delta m_{tij}
=
\frac{\epsilon\sigma_a^{Q}}{\sqrt d}
\left\langle u_a^{Q},k_i-k_j\right\rangle.
$$

Thus, to obtain a desired margin change \(\Delta m\), the required hidden perturbation size is:

$$
\epsilon
=
\frac{\sqrt d\,\Delta m}
{\sigma_a^{Q}\left\langle u_a^{Q},k_i-k_j\right\rangle},
$$

assuming the denominator is nonzero and has the desired sign. This makes the efficiency claim explicit: for the same useful margin direction, a larger query singular value \(\sigma_a^{Q}\) requires a smaller hidden movement to produce the same multiplicative attention-odds change.

The analogous key-side expression is:

$$
\delta m_{tij}
=
\frac{\epsilon\sigma_b^{K}}{\sqrt d}
\left\langle u_b^{K},q_t\right\rangle
$$

when only key \(i\) is perturbed relative to key \(j\). Again, larger \(\sigma_b^{K}\) creates the same score-margin change with a smaller hidden movement, provided the singular vector is aligned with the useful query direction.

Therefore, the corrected Q/K claim is:

$$
\boxed{
\text{Q/K high-gain directions are selected because they give larger loss decrease through useful score gradients and larger attention-margin change per hidden movement.}
}
$$

This is stronger than saying that softmax merely amplifies score gaps. The full mechanism is:

$$
\text{loss gradient identifies useful score changes}
\quad\rightarrow\quad
\text{large Q/K singular values implement those changes efficiently}
\quad\rightarrow\quad
\text{softmax converts score gaps into sharp routing.}
$$

The useful score matrix is:

$$
S=QK^\top.
$$

If the useful attention structure is low-rank:

$$
\operatorname{rank}(S_{\mathrm{useful}})=r,
$$

then there exist:

$$
\tilde Q,\tilde K\in\mathbb{R}^{n\times r}
$$

such that:

$$
S_{\mathrm{useful}}=\tilde Q\tilde K^\top.
$$

Therefore, if the task only needs a few relational factors, Q/K naturally need only a small number of useful directions. In the nested-prefix task, these factors are things like:

$$
\text{inside structured prefix},
\qquad
\text{current prefix stage},
\qquad
\text{continue the chain},
\qquad
\text{boundary or terminal marker}.
$$

This gives the low-rank Q/K claim:

$$
\boxed{
W_Q \text{ and } W_K \text{ become low-dimensional when the useful attention-score structure is low-rank.}
}
$$

The softmax then strengthens this effect because it focuses attention on dominant score gaps. But the reason large-singular-value directions are chosen is the descent-efficiency formula above, not softmax amplification alone.

## 7. Why \(W_V\) Is Different

The value projection is:

$$
v_i=W_Vh_i.
$$

Unlike \(W_Q\) and \(W_K\), \(W_V\) is not used to compute pairwise similarity. It is used to carry content after attention has selected a token or state.

So \(W_Q\) and \(W_K\) answer:

$$
\text{Which tokens or states should interact?}
$$

But \(W_V\) answers:

$$
\text{What content should be carried once that interaction is selected?}
$$

If values collapsed too strongly into the same high-gain subspace, token-specific content could be lost. Therefore, we expect:

$$
W_Q,\ W_K,\ W_{\mathrm{out}}
\quad
\text{to show stronger concentration for extra features than}
\quad
W_V.
$$

This is exactly what the current experiment shows.

## 8. Controlled Experiment

### 8.1 Dataset

The main dataset contains nested prefix segments:

$$
A,\ AB,\ ABC,\ ABCD,\ ABCDE,\ ABCDEF,\ ABCDEFG,\ ABCDEFG.
$$

Each structured prefix occurs with probability approximately \(5\%\). Random filler tokens from \(J,\ldots,Z\) fill the rest:

$$
\Pr(\text{random filler})\approx 60\%.
$$

This setup is deliberately simple. It lets us know exactly which feature is inherited and which feature is newly added.

### 8.2 Model

The model is a one-layer, one-head causal transformer:

$$
h_t
=
\operatorname{LN}
\left(
e(x_t)
+
W_O\sum_{i\leq t}\alpha_{ti}W_Ve(x_i)
\right).
$$

The output logits are:

$$
z_t=W_{\mathrm{out}}h_t.
$$

Training uses next-token cross entropy. The main comparisons are:

| setting | value |
|---|---:|
| layers | 1 |
| heads | 1 |
| context length | 8 |
| position embedding | none |
| batch size | 256 |
| training steps | 3000 |
| optimizer | AdamW |
| learning rate | \(2\times 10^{-3}\) |
| dimensions tested | \(d=32\), \(d=64\) |

### 8.3 Feature Directions

For each prefix \(F\), define:

$$
u_F
=
\operatorname{normalize}
\left(
\mathbb{E}[h_t\mid F]
-
\mathbb{E}[h_t\mid \text{random filler}]
\right).
$$

For a transition \(X\to Y\), define the inherited component:

$$
u_{\mathrm{common}}(X\to Y)
=
(u_Y^\top u_X)u_X.
$$

Define the extra component:

$$
u_{\mathrm{extra}}(X\to Y)
=
\operatorname{normalize}
\left(
u_Y-(u_Y^\top u_X)u_X
\right).
$$

The extra component is the main object of this draft. It measures the new representation direction created when a longer prefix adds one token.

### 8.4 Singular-Vector Alignment

For each matrix:

$$
W=U\Sigma V^\top,
$$

and each feature direction \(u\), define the mass on singular vector \(v_i\):

$$
m_i(u)=\langle u,v_i\rangle^2.
$$

The top-\(k\) mass is:

$$
M_k(u)=\sum_{i=1}^k m_i(u).
$$

High top-\(k\) mass means the feature lies mostly in the highest-singular-value subspace.

## 9. Results

### 9.1 The Model Learned The Structured Continuations

For \(d=64\), all structured continuations were predicted with \(100\%\) accuracy:

| continuation | accuracy | mean target probability | cross entropy |
|---|---:|---:|---:|
| \(A\to AB\) | \(100\%\) | \(0.884\) | \(0.124\) |
| \(AB\to ABC\) | \(100\%\) | \(0.879\) | \(0.129\) |
| \(ABC\to ABCD\) | \(100\%\) | \(0.831\) | \(0.185\) |
| \(ABCD\to ABCDE\) | \(100\%\) | \(0.810\) | \(0.211\) |
| \(ABCDE\to ABCDEF\) | \(100\%\) | \(0.780\) | \(0.249\) |
| \(ABCDEF\to ABCDEFG\) | \(100\%\) | \(0.655\) | \(0.423\) |
| \(ABCDEFG\to ABCDEFG.\) | \(100\%\) | \(0.556\) | \(0.587\) |

So the model did learn the nested chain. The geometry we inspect is not a failed-training artifact.

### 9.2 Extra Features Are Different Directions, But They Share A High-Gain Subspace

The pairwise distance between normalized extra directions is:

$$
d(u_i,u_j)=\sqrt{2-2u_i^\top u_j}.
$$

For \(d=64\), the extra-feature distances were:

| statistic | value |
|---|---:|
| minimum distance | \(1.242\) |
| mean distance | \(1.372\) |
| maximum distance | \(1.477\) |

A distance of roughly \(1.41\) means near-orthogonal. Therefore, the extra features are not all the same direction.

But they still share the high-gain \(W_{\mathrm{out}}\) subspace. For \(d=64\), top-10 \(W_{\mathrm{out}}\) hidden-side mass was:

| extra feature | top-10 mass |
|---|---:|
| \(A\to AB\) | \(93.7\%\) |
| \(AB\to ABC\) | \(92.1\%\) |
| \(ABC\to ABCD\) | \(91.4\%\) |
| \(ABCD\to ABCDE\) | \(89.9\%\) |
| \(ABCDE\to ABCDEF\) | \(90.6\%\) |
| \(ABCDEF\to ABCDEFG\) | \(86.0\%\) |
| \(ABCDEFG\to ABCDEFG.\) | \(16.9\%\) |

The key lesson is:

$$
\boxed{
\text{different feature direction} \neq \text{different singular subspace}.
}
$$

Many distinct extra features can live inside the same top singular-vector subspace.

### 9.3 Higher Dimension Spreads Features Slightly But Does Not Remove Concentration

Comparing \(d=32\) and \(d=64\):

| metric | \(d=32\) | \(d=64\) |
|---|---:|---:|
| final validation CE | \(1.338\) | \(1.338\) |
| \(W_{\mathrm{out}}\) largest singular value | \(2.379\) | \(2.069\) |
| \(W_{\mathrm{out}}\) effective rank | \(15.57\) | \(15.93\) |
| mean extra-feature distance | \(1.370\) | \(1.372\) |

The \(d=64\) model gives the representation more room, and internal top-10 masses are slightly lower than in \(d=32\). But the qualitative pattern remains: internal extra features are still concentrated in the top \(W_{\mathrm{out}}\) directions.

This suggests that simply increasing dimension does not automatically prevent shared-subspace reuse.

### 9.4 Adjacent Extra Features Are More Separated Than Remote Extra Features

For \(d=64\):

| pair type | mean distance | mean cosine |
|---|---:|---:|
| adjacent extra features | \(1.437\) | \(-0.033\) |
| remote extra features | \(1.355\) | \(0.080\) |

Adjacent transitions include:

$$
A\to AB
\quad \text{vs.} \quad
AB\to ABC.
$$

Remote transitions include pairs such as:

$$
A\to AB
\quad \text{vs.} \quad
ABCD\to ABCDE.
$$

This supports the idea that adjacent prefix states must stay distinguishable. The model must know whether it is at \(AB\), \(ABC\), or \(ABCD\), because these states predict different next tokens. Remote states can reuse more of the same general "continue the chain" structure.

### 9.5 \(W_Q\), \(W_K\), \(W_V\), And \(W_O\) Play Different Roles

For \(d=64\), average top-10 singular-vector mass for extra features was:

| matrix side | average top-10 mass |
|---|---:|
| \(W_Q\) input side | \(52.4\%\) |
| \(W_K\) input side | \(49.8\%\) |
| \(W_V\) input side | \(6.2\%\) |
| \(W_{\mathrm{out}}\) hidden input side | high for internal transitions, usually \(86\%-94\%\) |

This is an important mechanistic split.

The query and key matrices are used to compute attention similarity:

$$
s_{ti}=q_t^\top k_i.
$$

Therefore, they naturally concentrate relational structure. They ask:

$$
\text{which prefix state should interact with which context state?}
$$

The value matrix carries content:

$$
v_i=W_Vh_i.
$$

Therefore, it has less reason to collapse token-specific information into the same high-gain relational subspace.

This gives the interpretation:

$$
\boxed{
\text{Q/K concentrate compositional routing, while V preserves token-specific content.}
}
$$

## 10. Feature Interference

The useful part of this concentration is efficiency. Many related features can share the same high-gain subspace.

The dangerous part is interference.

Suppose two features \(f\) and \(g\) are:

$$
h_f=\sum_i c_{f,i}v_i,
\qquad
h_g=\sum_i c_{g,i}v_i.
$$

Their overlap is:

$$
\langle h_f,h_g\rangle
=
\sum_i c_{f,i}c_{g,i}.
$$

If both features concentrate in the same top-\(k\) directions, then:

$$
\langle h_f,h_g\rangle
\approx
\sum_{i=1}^k c_{f,i}c_{g,i}.
$$

Now consider the output effect:

$$
Wh_f
=
\sum_i \sigma_i c_{f,i}u_i.
$$

If \(\sigma_i\) is large for shared directions, then a small overlap in feature space can create a large overlap in output effect.

The same is true for gradients:

$$
\nabla_h L
=
W^\top\nabla_z L
=
\sum_i \sigma_i
\langle u_i,\nabla_z L\rangle
v_i.
$$

So if two features share high-\(\sigma_i\) directions, training one feature updates directions that also affect the other feature.

This is the feature-interference mechanism:

$$
\boxed{
\text{shared high-gain subspace couples feature learning.}
}
$$

For common features, this can be beneficial because many examples reinforce the same computation. For rare features, it can be harmful because their weaker gradients must modify a subspace already dominated by common-feature gradients.

## 11. Claim Boundary

The current evidence supports this claim:

$$
\boxed{
\text{In controlled nested-prefix learning, new compositional features become distinct directions inside shared high-gain singular subspaces.}
}
$$

The evidence also supports:

$$
\boxed{
\text{Q/K concentrate relational routing, while V remains less concentrated for these extra features.}
}
$$

The current evidence does not yet prove the full natural-language claim. Natural language contains many kinds of composition, ambiguity, hierarchy, and world knowledge. The controlled dataset isolates one ingredient: nested compositional continuation.

Therefore, the correct claim boundary is:

**This experiment shows a plausible mechanism by which compositional learning can create shared high-gain subspaces and feature interference. It does not yet prove that all language semantics use this mechanism.**

## 12. Proposed Next Experiments

### 12.1 Controlled Semantic Phrases

Replace symbolic prefixes with semi-natural phrases:

```text
city
beautiful city
NYC is a beautiful city
NYC is a beautiful city in winter
```

Then define the same objects:

$$
u_{\mathrm{extra}}(\text{short phrase}\to\text{long phrase}).
$$

Test whether the extra feature again concentrates in \(W_Q\), \(W_K\), and \(W_{\mathrm{out}}\), but less in \(W_V\).

### 12.2 Common Versus Rare Compositions

Create pairs of features that share a high-level composition structure but differ in frequency:

```text
common: the capital of France is Paris
rare: the capital of Tuvalu is Funafuti
```

Measure whether rare features reuse the same high-gain subspace and whether their learning is delayed or distorted by common-feature directions.

### 12.3 Causal Direction Intervention

For a learned rare feature, decompose its hidden direction into:

$$
h_{\mathrm{rare}}
=
h_{\mathrm{top}}
+
h_{\mathrm{tail}},
$$

where \(h_{\mathrm{top}}\) is the projection onto top singular vectors and \(h_{\mathrm{tail}}\) is the remaining component.

Then intervene:

$$
h' = h-\lambda h_{\mathrm{top}},
$$

or:

$$
h' = h-\lambda h_{\mathrm{tail}}.
$$

If removing \(h_{\mathrm{top}}\) damages both common and rare features, then the features share the same high-gain subspace. If removing \(h_{\mathrm{tail}}\) selectively damages rare features, then rare features also need lower-gain directions for specificity.

### 12.4 Training Objective Comparison

Compare standard cross entropy with label smoothing or adaptive smoothing. The prediction is:

$$
\text{standard CE}
\quad
\text{pushes common high-gain directions harder}.
$$

Label smoothing should reduce the incentive to keep increasing already-confident common-feature margins:

$$
\text{label smoothing}
\quad
\Rightarrow
\quad
\text{less extreme singular concentration}.
$$

This connects the current composition story to the earlier results where smoothing reduced matrix norms and extreme parameter values while preserving prediction accuracy.

## 13. Final Working Theory

The current working theory is:

1. Language learning is compositional.
2. Many complex features reuse simpler features.
3. Frequent simple features are learned early and repeatedly.
4. Repeated gradients create high-gain singular directions.
5. Later features are biased toward those directions because they reduce loss efficiently.
6. This creates a compact composition subspace.
7. The compact subspace improves reuse and generalization.
8. But it also mixes features together.
9. This mixing causes interference, especially for long-tail features.

In one sentence:

$$
\boxed{
\text{language composition encourages low-dimensional reuse, and low-dimensional reuse creates both efficient learning and feature interference.}
}
$$

That is the main research proposal emerging from the current study.

