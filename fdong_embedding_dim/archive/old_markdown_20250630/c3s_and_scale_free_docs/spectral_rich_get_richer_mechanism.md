# Spectral Rich-Get-Richer Mechanism in LLM Training

## 0. Purpose

This note organizes a mathematical story for why LLM training tends to reinforce already large singular modes instead of naturally allocating new learning to small singular-value directions.

The central question is:

> Why do LLMs tend to keep increasing already large top singular values and use those high-gain directions to distinguish more data, rather than learning new data through small singular-value directions?

The proposed answer is:

> Cross-entropy rewards immediate margin gain. Already large, stable singular modes provide immediate high-resolution separation. Small singular modes could be more capacity-efficient if they were aligned, but they face a direction-learning cold start, weak residual pressure, and orthogonality constraints.

---

## 1. Local SVD Setup

Consider a local linear block in an LLM:

$$
z = Wh,
$$

where:

- $h \in \mathbb{R}^d$ is a token hidden representation;
- $z$ is a logit vector or local output;
- $W$ is a trainable matrix block.

Let the singular value decomposition be

$$
W = U \Sigma V^\top
= \sum_{i=1}^r \sigma_i u_i v_i^\top .
$$

The singular vectors satisfy orthogonality:

$$
u_i^\top u_j = \delta_{ij},
\qquad
v_i^\top v_j = \delta_{ij}.
$$

For example $n$, define a local margin direction $q_n$. The margin is

$$
m_n = q_n^\top W h_n .
$$

Substituting the SVD gives

$$
m_n
=
q_n^\top
\left(
\sum_i \sigma_i u_i v_i^\top h_n
\right)
=
\sum_i \sigma_i
\left(q_n^\top u_i\right)
\left(v_i^\top h_n\right).
$$

Define

$$
a_{n,i} = q_n^\top u_i,
\qquad
c_{n,i} = v_i^\top h_n,
$$

and

$$
b_{n,i} = a_{n,i}c_{n,i}.
$$

Then

$$
m_n = \sum_i \sigma_i b_{n,i}.
$$

This equation says that singular mode $i$ contributes to the margin through three factors:

$$
\text{mode contribution}
=
\sigma_i b_{n,i}
=
\sigma_i a_{n,i}c_{n,i}.
$$

Here:

- $\sigma_i$ is the mode gain;
- $a_{n,i}$ is output-side alignment;
- $c_{n,i}$ is input-side hidden coefficient;
- $b_{n,i}$ is the current usefulness of mode $i$ for example $n$.

---

## 2. Cross-Entropy Residual Pressure

For a margin loss such as logistic cross-entropy,

$$
\ell(m_n) = \log(1+\exp(-m_n)).
$$

Define residual learning pressure

$$
r_n = -\ell'(m_n).
$$

For logistic cross-entropy,

$$
r_n = \frac{1}{1+\exp(m_n)}.
$$

Therefore, for any finite margin,

$$
r_n > 0.
$$

This means cross-entropy continues to reward larger margin even after an example is already correctly classified.

---

## 3. Singular-Value Dynamics Under Finite Scale

Pure cross-entropy on separable data can drive margins indefinitely. To model finite singular values, introduce a spectral scale penalty:

$$
\mathcal{L}
=
\sum_n p_n \ell(m_n)
+
\frac{\lambda}{2}
\sum_i \sigma_i^2.
$$

Here:

- $p_n$ is the frequency or probability of example type $n$;
- $\lambda > 0$ controls the finite scale penalty;
- $\sum_i \sigma_i^2 = \|W\|_F^2$ is total spectral energy.

Since

$$
m_n = \sum_i \sigma_i b_{n,i},
$$

we have

$$
\frac{\partial m_n}{\partial \sigma_i}
=
b_{n,i}.
$$

Thus,

$$
\frac{\partial \mathcal{L}}{\partial \sigma_i}
=
\sum_n p_n \ell'(m_n)b_{n,i}
+
\lambda\sigma_i.
$$

Using

$$
r_n=-\ell'(m_n),
$$

gradient flow gives

$$
\frac{d\sigma_i}{dt}
=
\sum_n p_n r_n b_{n,i}
-
\lambda\sigma_i.
$$

Define the cumulative residual utility of mode $i$:

$$
G_i
=
\sum_n p_n r_n b_{n,i}.
$$

Then

$$
\frac{d\sigma_i}{dt}
=
G_i-\lambda\sigma_i.
$$

At equilibrium,

$$
\lambda\sigma_i = G_i.
$$

Therefore, a singular value becomes large when its mode receives large cumulative residual utility.

---

## 4. Revised Spectral Rich-Get-Richer Mechanism

### Step 1: Language Data Are Heavy-Tailed

Common features occur much more often than rare features. A simple model is

$$
p_i \propto i^{-\alpha},
\qquad
\alpha > 0.
$$

Here $p_i$ is the frequency of feature or pattern $i$.

Thus common features produce many more gradient updates than rare features.

---

### Step 2: Frequent Features Create Stable High-Gain Modes

For singular mode $i$, the cumulative residual utility is

$$
G_i=\sum_n p_n r_n b_{n,i}.
$$

Frequent reusable features contribute to many examples. If a mode aligns with such features, then $b_{n,i}$ is nonzero for many high-frequency examples.

Therefore $G_i$ becomes large, and the singular-value dynamics

$$
\frac{d\sigma_i}{dt}=G_i-\lambda\sigma_i
$$

increase $\sigma_i$.

At equilibrium,

$$
\sigma_i = \frac{G_i}{\lambda}.
$$

Thus frequent, reusable, well-aligned modes become high-gain modes.

---

### Step 3: Top Modes Become High-Resolution Axes

Along singular mode $i$, an input-coordinate difference $\Delta c_i$ produces output separation

$$
\Delta z_i = \sigma_i \Delta c_i.
$$

To achieve output separation at least $\gamma$, the required hidden-coordinate separation is

$$
|\Delta c_i|
\geq
\frac{\gamma}{\sigma_i}.
$$

If $\sigma_i$ is large, then only a small hidden-coordinate difference is needed to distinguish data.

Therefore, top singular modes act as high-resolution axes:

$$
\boxed{
\text{large } \sigma_i
\Rightarrow
\text{small hidden differences become large output differences.}
}
$$

This makes already large singular directions immediately useful for fitting or separating additional data.

---

### Step 4: Cross-Entropy Keeps Rewarding Finite-Margin Improvement

For logistic cross-entropy,

$$
r_n=\frac{1}{1+\exp(m_n)}.
$$

For every finite margin,

$$
r_n>0.
$$

So even already learned common examples can keep producing residual pressure.

If a stable top mode contributes positively to the margin,

$$
m_n
=
\sigma_1 b_{n,1}
+
\sum_{j\geq 2}\sigma_j b_{n,j},
$$

then increasing $\sigma_1$ further still improves the cross-entropy loss as long as $r_n>0$.

Thus cross-entropy does not naturally stop training a common direction once it is already correct. It continues to reward larger confidence.

---

### Step 5: Small Modes Face a Direction-Learning Cold Start

For a small singular mode $j$, using it effectively requires the mode to become aligned with some useful residual feature.

A useful local abstraction for direction-learning speed is

$$
\text{direction-learning rate of mode } j
\sim
r_n \sigma_j.
$$

If $\sigma_j$ is small, the direction aligns slowly.

Thus small singular modes suffer from a cold start:

$$
\boxed{
\sigma_j \text{ small}
\Rightarrow
\text{weak direction-learning signal.}
}
$$

This does not mean small singular modes are useless. It means they require an initial alignment investment before increasing their singular value becomes useful.

---

### Step 6: Orthogonality Makes Small Modes Residual-Only

Singular vectors are orthogonal:

$$
v_i^\top v_j = 0
\qquad
(i\neq j).
$$

Suppose the top $k$ singular vectors already span common learned directions:

$$
\mathcal{V}_k = \operatorname{span}(v_1,\dots,v_k).
$$

Let

$$
P_k = \sum_{i=1}^k v_i v_i^\top
$$

be the projector onto this top singular subspace.

For a feature direction $x_n$, the residual component outside the top subspace is

$$
x_{n,\perp}^{(k)}
=
(I-P_k)x_n.
$$

A new small singular direction $v_j$, with $j>k$, cannot duplicate the already learned top directions. It can only learn from the residual component

$$
x_{n,\perp}^{(k)}.
$$

If

$$
\|x_{n,\perp}^{(k)}\| \ll 1,
$$

then the useful signal available to small modes is weak.

Thus orthogonality implies:

$$
\boxed{
\text{small modes cannot copy top modes; they can only learn residual directions.}
}
$$

---

### Step 7: Top Modes Reduce Residual Pressure

Once top modes increase the margin, cross-entropy residual pressure decreases.

For logistic cross-entropy,

$$
r_n
=
\frac{1}{1+\exp(m_n)}.
$$

If the top mode dominates the margin,

$$
m_n \approx \sigma_1 b_{n,1},
$$

then for large positive margin,

$$
r_n
\approx
\exp(-m_n)
\approx
\exp(-\sigma_1 b_{n,1}).
$$

So as top singular values grow, residual pressure shrinks.

This leaves even weaker learning signal for small modes:

$$
\frac{d\sigma_j}{dt}
=
\sum_n p_n r_n b_{n,j}
-
\lambda\sigma_j.
$$

The small mode update is limited by:

1. small residual pressure $r_n$;
2. small current usefulness $b_{n,j}$;
3. small initial scale $\sigma_j$ for direction learning;
4. limited residual feature component due to orthogonality.

---

## 5. Final Mechanism Summary

Large modes are immediately useful:

$$
\boxed{
\text{large modes are already aligned, high-resolution, and receive repeated updates.}
}
$$

Small modes are potentially efficient but hard to activate:

$$
\boxed{
\text{small modes are capacity-efficient after alignment, but require slow alignment under weak residual pressure.}
}
$$

Therefore standard cross-entropy training creates a spectral rich-get-richer effect:

$$
\boxed{
\text{top singular modes keep growing because they give immediate margin gain.}
}
$$

Meanwhile,

$$
\boxed{
\text{small singular modes remain underused because they require delayed alignment investment.}
}
$$

This mechanism can produce and reinforce a Zipf-like singular-value distribution.

---

## 6. Capacity Inefficiency of Spectral Concentration

The total spectral scale is

$$
B=\sum_i\sigma_i^2=\|W\|_F^2.
$$

For a fixed parameter matrix and finite training scale, $B$ is a proxy for total available spectral capacity.

However, useful dimensional capacity depends on how $B$ is distributed.

Define effective spectral dimension:

$$
d_{\mathrm{eff}}
=
\frac{\left(\sum_i\sigma_i^2\right)^2}
{\sum_i\sigma_i^4}.
$$

If scale is evenly spread across $k$ modes,

$$
\sigma_1^2=\cdots=\sigma_k^2=\frac{B}{k},
$$

then

$$
d_{\mathrm{eff}}=k.
$$

If one singular value dominates,

$$
\sigma_1^2\approx B,
$$

then

$$
d_{\mathrm{eff}}\approx 1.
$$

So, for the same total spectral scale $B$, a flatter spectrum has larger effective dimensional capacity than a concentrated spectrum.

Another coding-capacity view is:

$$
N_i
\approx
1+\frac{R\sigma_i}{\gamma},
$$

where:

- $R$ is hidden representation norm scale;
- $\gamma$ is the required output resolution;
- $N_i$ is the approximate number of distinguishable levels along mode $i$.

Across orthogonal modes,

$$
N_{\mathrm{total}}
\approx
\prod_i
\left(
1+\frac{R\sigma_i}{\gamma}
\right).
$$

For fixed

$$
\sum_i\sigma_i^2=B,
$$

this product is larger when the scale is distributed across more modes, rather than concentrated in only a few modes.

Thus:

$$
\boxed{
\text{using small singular directions can improve model-capacity utilization.}
}
$$

The problem is that cross-entropy does not directly optimize this capacity objective.

---

## 7. Modeling Sanity Check

### 7.1 What the Model Explains Well

This model explains why top singular values continue to grow during further training:

$$
\frac{d\sigma_i}{dt}
=
\sum_n p_n r_n b_{n,i}
-
\lambda\sigma_i.
$$

If top modes have large $b_{n,i}$ over many examples, they continue to receive cumulative residual utility.

The model also explains why small modes are underused:

$$
\text{direction-learning rate}
\sim
r_n\sigma_j.
$$

When both $r_n$ and $\sigma_j$ are small, small modes align slowly.

The model further respects singular-vector orthogonality:

$$
v_i^\top v_j=\delta_{ij}.
$$

Small modes cannot simply learn the same direction as top modes. They must learn residual directions.

---

### 7.2 Important Caveats

#### Caveat 1: The SVD of a changing matrix is not a fixed coordinate system

In a real neural network, $U$, $V$, and $\Sigma$ all change during training. The equation

$$
\frac{d\sigma_i}{dt}
=
G_i-
\lambda\sigma_i
$$

is most accurate after the top singular subspace is approximately stable.

Therefore, the model should be applied mainly to the late or continual-training stage, where top singular directions drift slowly.

---

#### Caveat 2: $b_{n,i}$ is not fixed

The quantity

$$
b_{n,i}=a_{n,i}c_{n,i}
$$

depends on both representation $h_n$ and singular vectors $u_i,v_i$. These can change during training.

Thus, the model is not a full closed-form training theory. It is a local or slow-timescale approximation.

---

#### Caveat 3: Small modes can learn if residual errors require them

The model does not claim that small singular modes never grow.

If new data contains errors that cannot be reduced by top modes, then $r_n$ remains large for those examples and top-mode usefulness $b_{n,1}$ may be small. In that case, residual modes can receive strong gradients.

So the correct claim is:

$$
\boxed{
\text{small modes are underused when top modes can already provide substantial immediate loss reduction.}
}
$$

---

#### Caveat 4: The scale penalty is a simplified proxy

The penalty

$$
\frac{\lambda}{2}\sum_i\sigma_i^2
$$

models finite singular scale. In real LLMs, finite scale may come from a combination of:

- AdamW weight decay;
- normalization layers;
- finite precision;
- optimizer dynamics;
- residual scaling;
- architectural constraints.

Thus, $\lambda$ should be interpreted as an effective finite-scale regularization, not necessarily only explicit weight decay.

---

#### Caveat 5: Direction-learning rate is a proxy

The expression

$$
\text{direction-learning rate}\sim r_n\sigma_j
$$

is a local abstraction. It captures the key dependence on residual pressure and singular gain, but exact singular-vector dynamics for a full matrix with multiple modes can include additional terms, especially when singular values are close.

Therefore, experiments should measure direction drift directly rather than rely only on this proxy.

---

## 8. Minimal Experiments to Test the Mechanism

The goal is to verify each step of the proposed mechanism with measurable quantities.

---

### Experiment 1: Verify Heavy-Tailed Feature or Token Statistics

#### Hypothesis

Language data contain heavy-tailed features:

$$
p_i\propto i^{-\alpha}.
$$

#### Measurement

Use one of the following feature definitions:

1. token frequency;
2. $n$-gram frequency;
3. SAE feature activation frequency;
4. cluster frequency of hidden representations.

For each feature $i$, estimate frequency $p_i$.

Plot:

$$
\log p_i
\quad \text{vs.} \quad
\log i.
$$

#### Expected Result

A roughly linear trend on log-log scale supports a power-law or heavy-tailed distribution.

#### Supports

This supports Step 1.

---

### Experiment 2: Verify Singular Spectrum Is Heavy-Tailed

#### Hypothesis

LLM matrix blocks have Zipf-like singular spectra.

#### Measurement

For selected matrices such as

$$
W_{\mathrm{out}},
\quad
W_Q,
\quad
W_K,
\quad
W_V,
\quad
W_{\mathrm{up}},
\quad
W_{\mathrm{down}},
$$

compute singular values

$$
\sigma_1\geq\sigma_2\geq\cdots.
$$

Plot:

$$
\log\sigma_i
\quad \text{vs.} \quad
\log i.
$$

Also compute

$$
d_{\mathrm{eff}}
=
\frac{\left(\sum_i\sigma_i^2\right)^2}
{\sum_i\sigma_i^4}.
$$

#### Expected Result

Top singular values dominate, and $d_{\mathrm{eff}}$ is much smaller than the matrix rank.

#### Supports

This supports the spectral concentration part of the model.

---

### Experiment 3: Verify Top Singular Subspace Stabilizes Earlier Than Singular Values

#### Hypothesis

Top singular directions become stable earlier, while top singular values continue growing.

#### Measurement

Save checkpoints during training. For each selected matrix, compute top-$k$ right singular subspace:

$$
P_k(t)=V_k(t)V_k(t)^\top.
$$

Measure subspace drift:

$$
\delta_k(t)
=
\|P_k(t)-P_k(t-\Delta)\|_2.
$$

Measure top-$k$ singular energy:

$$
S_k(t)=\sum_{i=1}^k\sigma_i(t)^2.
$$

#### Expected Result

Late training shows:

$$
\delta_k(t)\approx 0,
$$

but

$$
S_k(t)-S_k(t-\Delta)>0.
$$

#### Supports

This supports the claim that direction learning stabilizes while gain amplification continues.

---

### Experiment 4: Verify Frequent Features Align More With Top Singular Modes

#### Hypothesis

Frequent reusable features have larger top-subspace mass.

#### Measurement

Given feature directions $f_i$, compute top-$k$ mass:

$$
M_k(f_i)
=
\|P_k f_i\|^2.
$$

Compare $M_k(f_i)$ across frequency groups:

- high-frequency features;
- medium-frequency features;
- low-frequency features.

#### Expected Result

High-frequency features have larger

$$
M_k(f_i)
$$

than rare features.

#### Supports

This supports Step 2.

---

### Experiment 5: Verify Top Modes Are High-Resolution Axes

#### Hypothesis

Along a large singular mode, small hidden-coordinate differences create large output differences.

#### Measurement

For token pairs $(a,b)$, compute hidden-coordinate difference along mode $i$:

$$
\Delta c_i
=
v_i^\top h_a - v_i^\top h_b.
$$

Compute output difference along $u_i$:

$$
\Delta z_i
=
u_i^\top W h_a - u_i^\top W h_b.
$$

Check whether

$$
\Delta z_i \approx \sigma_i \Delta c_i.
$$

Then compare the magnitude of $\Delta z_i$ for top modes and small modes at matched $|\Delta c_i|$.

#### Expected Result

For matched hidden-coordinate differences, top modes produce larger output separation:

$$
|\Delta z_i| \propto \sigma_i|\Delta c_i|.
$$

#### Supports

This supports Step 3.

---

### Experiment 6: Verify Cross-Entropy Residual Pressure Remains Positive for Learned Examples

#### Hypothesis

Even learned examples continue to produce small but nonzero residual pressure.

#### Measurement

For examples with correct-label probability $p_y$, compute

$$
r = 1-p_y
$$

for standard softmax cross-entropy.

Equivalently, for binary margin form,

$$
r = \frac{1}{1+\exp(m)}.
$$

Group examples by confidence:

- $p_y \in [0.5,0.7]$;
- $p_y \in [0.7,0.9]$;
- $p_y \in [0.9,0.99]$;
- $p_y > 0.99$.

Measure gradient norm or singular-mode update contribution in each group.

#### Expected Result

High-confidence examples still have nonzero residual pressure and nonzero gradient contribution, though smaller than low-confidence examples.

#### Supports

This supports Step 4.

---

### Experiment 7: Verify Small Modes Have Direction-Learning Cold Start

#### Hypothesis

Small singular modes rotate or align more slowly because their direction-learning rate scales with

$$
r_n\sigma_j.
$$

#### Measurement

For each checkpoint, compute the singular-vector drift of mode $j$:

$$
D_j(t)
=
1-\left|v_j(t)^\top v_j(t-\Delta)\right|.
$$

Alternatively, for a known feature direction $f$, compute alignment change:

$$
A_j(t)
=
\left|v_j(t)^\top f\right|^2.
$$

Measure

$$
\Delta A_j(t)
=
A_j(t)-A_j(t-\Delta).
$$

Compare $\Delta A_j(t)$ with

$$
\sigma_j(t)
$$

and average residual pressure on examples associated with $f$.

#### Expected Result

Modes with larger $\sigma_j$ show faster alignment changes, after controlling for residual pressure and feature frequency.

#### Supports

This supports Step 5.

---

### Experiment 8: Verify Small Modes Learn Residual Components Due to Orthogonality

#### Hypothesis

After top modes capture common directions, small modes can only learn residual components.

#### Measurement

Let

$$
P_k(t)=\sum_{i=1}^k v_i(t)v_i(t)^\top.
$$

For feature direction $f$, compute residual norm:

$$
\rho_k(f,t)
=
\|(I-P_k(t))f\|^2.
$$

For small modes $j>k$, measure their future alignment increase:

$$
\Delta A_j(f,t)
=
|v_j(t+\Delta)^\top f|^2
-
|v_j(t)^\top f|^2.
$$

Test whether $\Delta A_j(f,t)$ is predicted better by residual norm

$$
\rho_k(f,t)
$$

than by raw feature frequency alone.

#### Expected Result

Small modes grow toward features with larger residual component outside the top subspace.

#### Supports

This supports Step 6.

---

### Experiment 9: Verify Top Modes Reduce Residual Pressure for Small Modes

#### Hypothesis

When top-mode margin contribution is large, residual pressure becomes small and small-mode updates are suppressed.

#### Measurement

For each example $n$, compute top-$k$ margin contribution:

$$
m_n^{(k)}
=
\sum_{i=1}^k \sigma_i b_{n,i}.
$$

Compute residual pressure:

$$
r_n=\frac{1}{1+\exp(m_n)}
$$

or for softmax CE,

$$
r_n=1-p_{y,n}.
$$

Compute small-mode gradient utility:

$$
G_{j,n}
=
r_n b_{n,j}.
$$

Analyze whether examples with larger $m_n^{(k)}$ have smaller $r_n$ and smaller small-mode utility.

#### Expected Result

As

$$
m_n^{(k)}
$$

increases,

$$
r_n
$$

decreases, and small-mode update utility decreases.

#### Supports

This supports Step 7.

---

### Experiment 10: Test Capacity Inefficiency Directly

#### Hypothesis

A more evenly distributed singular spectrum improves capacity utilization for learning new or rare data.

#### Measurement

Compare training variants:

1. baseline cross-entropy;
2. confidence-capped cross-entropy;
3. spectral flattening regularization;
4. top-subspace suppression;
5. explicit small-mode routing or residual-subspace training.

Measure:

- validation loss on common data;
- validation loss on rare or new data;
- singular spectrum;
- effective spectral dimension $d_{\mathrm{eff}}$;
- top-subspace feature mass;
- small-mode feature alignment;
- forgetting or interference.

#### Expected Result

If the theory is correct, methods that encourage residual small-mode learning should increase

$$
d_{\mathrm{eff}},
$$

increase rare-feature alignment with small modes, and improve rare or continual-learning performance without heavily harming common-feature retention.

#### Supports

This tests the overall capacity-efficiency claim.

---

## 9. Minimal Experiment Set

If only a small number of experiments can be run, use these four.

### Minimal Experiment A: Spectrum and Stability

Measure:

$$
\sigma_i(t),
\qquad
d_{\mathrm{eff}}(t),
\qquad
\delta_k(t)=\|P_k(t)-P_k(t-\Delta)\|_2.
$$

Expected:

- spectrum is heavy-tailed;
- top subspace stabilizes;
- top singular energy continues growing.

This supports the basic spectral rich-get-richer observation.

---

### Minimal Experiment B: Feature Frequency vs Top-Subspace Mass

Measure:

$$
M_k(f_i)=\|P_k f_i\|^2
$$

for frequent and rare features.

Expected:

- frequent features have larger top-subspace mass;
- rare features have more residual mass.

This supports the frequency-to-top-mode link.

---

### Minimal Experiment C: Residual Pressure vs Small-Mode Growth

Measure:

$$
r_n,
\qquad
m_n^{(k)},
\qquad
G_{j,n}=r_n b_{n,j}.
$$

Expected:

- large top-mode margin reduces $r_n$;
- small-mode utility is weak when top-mode margin is high.

This supports residual-pressure starvation.

---

### Minimal Experiment D: Intervention to Encourage Small-Mode Learning

Compare baseline CE against one intervention, such as:

1. confidence capping;
2. suppressing updates in the top singular subspace;
3. adding effective-rank regularization;
4. routing rare examples toward residual subspace.

Measure:

$$
d_{\mathrm{eff}},
$$

rare-feature performance, and common-feature retention.

Expected:

- intervention increases small-mode usage;
- rare/new data performance improves;
- common performance is preserved or only slightly reduced.

This tests the practical implication of the theory.

---

## 10. What Would Falsify the Model?

The model would be weakened if experiments show:

1. top singular directions do not stabilize before singular values continue growing;
2. frequent features do not align more with top singular subspaces;
3. small singular modes align just as quickly as top modes after controlling for residual pressure;
4. top-mode margin does not reduce residual pressure;
5. increasing effective spectral dimension does not improve rare or continual-learning performance.

The strongest falsification would be:

$$
\text{small-mode usage increases naturally under CE without any intervention,}
$$

while

$$
\text{top singular values do not dominate further training.}
$$

That would contradict the proposed spectral rich-get-richer mechanism.
