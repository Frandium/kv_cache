# Long-Horizon Semantic Learning with Multi-Token Prediction

Version: v0  
Purpose: self-contained story + theory proof for the controlled MTP vs NTP line.  
Scope: objective-level theory first; efficiency and architecture claims are empirical or conditional.

---

## 0. One-sentence claim

Multi-token prediction has a mathematically guaranteed advantage over next-token prediction **inside a fixed long-horizon semantic framework**: when an early semantic variable is not needed for the immediate next token but is needed for a farther future token, next-$K$ prediction with $K$ covering that future token directly forces the prefix hidden state to contain the early semantic variable, while next-token prediction at the same prefix does not.

---

## 1. Why this framework has real value

The original question is broad:

> Why is MTP better than NTP for learning long-horizon semantics?

This cannot be proved without defining what “long-horizon semantics” and “better” mean. The value of the current framework is that it turns this open question into a measurable and provable object.

### 1.1 Forecast-state value

A next-token-trained hidden state only has to support

$$
h_T \to Y_1 .
$$

A multi-token-trained hidden state has to support

$$
h_T \to Y_1,Y_2,\ldots,Y_K .
$$

Thus MTP encourages $h_T$ to become a **forecast state**: a representation that exposes information needed for future continuation, not only the immediate next token.

This matters for long-context generation, reasoning, instruction following, code generation, theorem proving, and planning-like language use, where an early condition affects later output after many locally predictable tokens.

### 1.2 Reduced local-confidence collapse

NTP can collapse different semantic branches into the same hidden state when the immediate next token is shared:

$$
h_T^{(1)}=h_T^{(2)}=\cdots=h_T^{(m)}
$$

and still predict the shared next token perfectly. This creates a local-confidence shortcut: the model is confident about the next token while hiding the fact that future continuations differ.

MTP penalizes this collapse once any supervised future token differs across branches. Therefore MTP can reduce reuse of the same local confidence state for semantically different prefixes.

### 1.3 Learning-efficiency value

The proof gives an objective-level guarantee: if the future token is branch-specific, low MTP loss forces semantic information in $h_T$. It does **not** by itself prove faster optimization.

Efficiency must be tested by learning curves:

$$
T_\theta(K)=\inf\{t:Q_K(t)\ge \theta\},
$$

where $Q_K(t)$ is a guarded long-horizon semantic score. If MTP reaches a semantic threshold earlier without hurting local next-token loss, then we can claim learning-efficiency gain.

### 1.4 Relation to nested learning

Two-token training creates a marginal-to-refined prediction pair:

$$
p^{(2)}(x_{t+2}\mid x_{\le t})
$$

predicts $x_{t+2}$ before $x_{t+1}$ is seen, while

$$
p^{(1)}(x_{t+2}\mid x_{\le t+1})
$$

predicts the same token after $x_{t+1}$ is revealed.

This gives a clean nested-learning interpretation: earlier states learn coarse future forecasts; later states refine them after intermediate evidence arrives. The current proof does not require this interpretation, but it gives a natural bridge to future work on hierarchical or nested prediction.

### 1.5 Relation to routing / MoE

If a router observes the hidden state $h_T$, then MTP-induced semantic information can make routing depend on long-horizon semantic variables rather than only local next-token patterns. This is a potential extension, not part of the current theorem. The theorem supplies the missing prerequisite: when does $h_T$ provably contain the long-horizon variable?

---

## 2. Definitions

### 2.1 Sequence and prefix state

Let a causal model read a prefix $X_{\le T}$ and produce a hidden state

$$
h_T=f_\theta(X_{\le T}).
$$

Define future tokens

$$
Y_j=X_{T+j},\qquad j=1,2,\ldots .
$$

### 2.2 Long-horizon semantic variable

A long-horizon semantic variable is an early variable $Z$ that is available in the prefix, is not necessarily useful for the immediate next token, but controls some later continuation.

In synthetic experiments, $Z$ is the branch key $B$. In natural language, $Z$ can be an early story premise, user instruction, program variable, theorem assumption, entity identity, or discourse constraint.

The key property is delayed relevance:

$$
Z \text{ appears early, but its predictive value appears later.}
$$

### 2.3 First informative horizon

Let

$$
\Delta_j=I(Z;Y_j\mid Y_{<j})
$$

be the additional information about $Z$ revealed by the $j$-th future token after earlier future tokens are known.

Define the first informative horizon

$$
\tau=\min\{j:\Delta_j>0\}.
$$

Equivalently, for every $K<\tau$,

$$
I(Z;Y_{1:K})=0,
$$

while

$$
I(Z;Y_{1:\tau})>0.
$$

This is the mathematical version of “the early semantic factor first matters at horizon $\tau$.”

### 2.4 NTP and next-K objectives

NTP at prefix $T$ is

$$
L_{\mathrm{NTP},T}=\operatorname{CE}(q_1(\cdot\mid h_T),Y_1).
$$

Next-$K$ prediction is

$$
L_{K,T}=\sum_{j=1}^{K}\lambda_j\operatorname{CE}(q_j(\cdot\mid h_T),Y_j),
$$

with $\lambda_j>0$ for active horizons.

### 2.5 Single-prefix future prediction

Single-prefix future prediction asks whether the same hidden state $h_T$ can predict future tokens:

$$
h_T\to Y_j.
$$

This differs from autoregressive rollout, which predicts one token, appends it, recomputes the state, and then predicts the next token.

---

## 3. Minimal K=2 model

Let

$$
Z\sim \operatorname{Unif}\{1,\ldots,m\}.
$$

Construct the sequence

$$
X=(\operatorname{BR}_Z,F_1,\ldots,F_L,U,A,S_Z).
$$

Here:

- $\operatorname{BR}_Z$ reveals the early semantic branch;
- $F_1,\ldots,F_L$ are filler tokens;
- $U$ is shared local context;
- $A$ is the shared first future token;
- $S_Z$ is a branch-specific second future token;
- $Z\mapsto S_Z$ is one-to-one.

The decision state is

$$
h_T^{(Z)}=f_\theta(\operatorname{BR}_Z,F_1,\ldots,F_L,U).
$$

The future targets are

$$
Y_1=A,
$$

$$
Y_2=S_Z.
$$

Therefore

$$
I(Z;Y_1)=0,
$$

but

$$
I(Z;Y_2\mid Y_1)=H(Z)=\log m.
$$

This is the smallest clean case where the immediate next token does not require $Z$, but the second token does.

---

## 4. K=2 proof

### Lemma 1: branch-blind states cannot predict the branch-specific second token

Assume

$$
h_T^{(1)}=\cdots=h_T^{(m)}=c.
$$

Then any predictor $q_2(\cdot\mid h_T)$ has

$$
\inf_{q_2}\mathbb E[-\log q_2(S_Z\mid h_T)]=\log m.
$$

**Proof.** If $h_T=c$ for every branch, the predictor must use one distribution $q_2(\cdot\mid c)$. Since $S_Z$ is uniform over $m$ distinct tokens, the expected cross-entropy is

$$
\frac{1}{m}\sum_{z=1}^{m}-\log q_2(S_z\mid c).
$$

This is minimized by $q_2(S_z\mid c)=1/m$ for all $z$, giving $\log m$. A deterministic top-1 predictor has accuracy at most $1/m$. ∎

### Lemma 2: next-token loss at the decision point can be solved without branch information

The NTP target is $Y_1=A$, shared by all branches. If

$$
h_T^{(1)}=\cdots=h_T^{(m)}=c
$$

and

$$
q_1(A\mid c)=1,
$$

then

$$
\operatorname{CE}(q_1(\cdot\mid h_T),A)=0
$$

for every branch. Thus NTP at this prefix does not require $h_T$ to expose $Z$. ∎

### Proposition 1: NTP success does not imply single-prefix second-token decodability

There exists a branch-blind prefix state with zero NTP loss at $T$, but optimal second-token loss $\log m$.

**Proof.** Combine Lemma 1 and Lemma 2. ∎

### Proposition 2: next-two loss excludes branch-blind prefix states

The next-two loss is

$$
L_{2,T}=\operatorname{CE}(q_1(\cdot\mid h_T),A)+\lambda\operatorname{CE}(q_2(\cdot\mid h_T),S_Z).
$$

If $h_T$ is branch-blind, then by Lemma 1

$$
\mathbb E[L_{2,T}]\ge \lambda\log m.
$$

If $h_T$ uniquely identifies $Z$, a realizable model can drive both terms to zero. Therefore next-two distinguishes branch-blind and branch-aware states, while NTP at $T$ does not. ∎

### Proposition 3: low second-token loss forces semantic information

If

$$
\mathbb E[-\log q_2(S_Z\mid h_T)]\le \varepsilon,
$$

then

$$
I(h_T;Z)\ge \log m-\varepsilon.
$$

**Proof.** For any predictor,

$$
H(S_Z\mid h_T)\le \mathbb E[-\log q_2(S_Z\mid h_T)].
$$

Since $Z\mapsto S_Z$ is one-to-one,

$$
H(Z\mid h_T)=H(S_Z\mid h_T).
$$

Thus

$$
I(h_T;Z)=H(Z)-H(Z\mid h_T)\ge \log m-\varepsilon.
$$

∎

### Proposition 4: next-two gives a branch-dependent gradient to $h_T$

With linear heads

$$
z_j=W_jh_T+b_j,
$$

$$
p_j=\operatorname{softmax}(z_j),
$$

NTP gives

$$
\nabla_{h_T}L_{\mathrm{NTP},T}=W_1^\top(p_1-e_A).
$$

Next-two gives

$$
\nabla_{h_T}L_{2,T}=W_1^\top(p_1-e_A)+\lambda W_2^\top(p_2-e_{S_Z}).
$$

The second term depends explicitly on $Z$. Therefore next-two sends a branch-dependent error signal directly to the same prefix state $h_T$. ∎

### K=2 theorem

In the distribution

$$
X=(\operatorname{BR}_Z,F_1,\ldots,F_L,U,A,S_Z),
$$

where $A$ is shared and $S_Z$ is branch-specific, next-token prediction at $T$ can be solved by a branch-blind prefix state, but next-two prediction cannot be solved with low loss unless $h_T$ contains information about $Z$. Therefore next-two imposes a direct long-horizon semantic representation requirement that NTP at the same prefix does not impose. ∎

---

## 5. General Next-K theory

### 5.1 First-informative-horizon distribution

Let $Z\sim P_Z$. A distribution belongs to the clean first-informative-horizon class $\mathcal D_\tau$ if:

1. $Z$ is available in the prefix $X_{\le T}$;
2. for every $j<\tau$, the future token $Y_j$ contains no information about $Z$:

$$
I(Z;Y_{1:j})=0;
$$

3. at horizon $\tau$, the future target contains semantic information about $Z$:

$$
I(Z;Y_{1:\tau})>0;
$$

4. in the strongest deterministic case,

$$
Y_\tau=S_Z
$$

with $Z\mapsto S_Z$ one-to-one.

The K=2 model is the special case $\tau=2$:

$$
Y_1=A,
\qquad
Y_2=S_Z.
$$

The K=3 model is

$$
Y_1=A,
\qquad
Y_2=C,
\qquad
Y_3=S_Z.
$$

### 5.2 Theorem 1: non-coverage gives no direct semantic requirement

Let $K<\tau$. Suppose $Y_{1:K}$ is independent of $Z$. Then there exists a $Z$-blind prefix representation that is Bayes-optimal for all supervised next-$K$ targets.

**Proof.** If $Y_{1:K}$ is independent of $Z$, the Bayes-optimal distributions $P(Y_j\mid X_{\le T})$ for $j\le K$ do not require distinguishing $Z$. A representation $h_T=c$ can output the correct branch-averaged distribution for every supervised horizon $j\le K$. Therefore the next-$K$ objective with $K<\tau$ does not force $h_T$ to expose $Z$. ∎

In the deterministic shared-token case, this is even stronger: if

$$
Y_j=a_j\quad\text{for }j<\tau,
$$

then a constant $h_T=c$ can achieve zero supervised loss for every $j\le K<\tau$.

### 5.3 Theorem 2: coverage forces semantic information

Let $K\ge\tau$, $\lambda_\tau>0$, and $Y_\tau=S_Z$ with $Z\mapsto S_Z$ one-to-one. If the offset-$\tau$ loss satisfies

$$
\mathbb E[-\log q_\tau(S_Z\mid h_T)]\le \varepsilon,
$$

then

$$
I(h_T;Z)\ge H(Z)-\varepsilon.
$$

If only the weighted contribution is bounded as

$$
\lambda_\tau\mathbb E[-\log q_\tau(S_Z\mid h_T)]\le \epsilon,
$$

then

$$
I(h_T;Z)\ge H(Z)-\frac{\epsilon}{\lambda_\tau}.
$$

**Proof.** Same as K=2 Proposition 3, replacing $S_B$ by $S_Z$ and horizon 2 by horizon $\tau$. ∎

### 5.4 Theorem 3: next-K has a semantic coverage advantage over smaller horizons

For two horizons $K'<K$, define the semantic coverage of the supervised suffix by

$$
C(K)=I(Z;Y_{1:K}).
$$

By the chain rule,

$$
C(K)-C(K')=I(Z;Y_{K'+1:K}\mid Y_{1:K'}).
$$

Therefore next-$K$ imposes a strictly stronger semantic information requirement than next-$K'$ exactly when

$$
I(Z;Y_{K'+1:K}\mid Y_{1:K'})>0.
$$

In particular, compared with NTP $(K'=1)$, next-$K$ has additional long-horizon semantic pressure iff

$$
I(Z;Y_{2:K}\mid Y_1)>0.
$$

**Interpretation.** The advantage is not “larger $K$ is always better.” The advantage appears when the added horizons contain new information about the early semantic variable.

### 5.5 Theorem 4: multiple partial semantic tokens

Sometimes no single future token identifies $Z$, but several future tokens jointly identify it.

Let $J\subseteq\{1,\ldots,K\}$. Suppose

$$
Y_J=(Y_j)_{j\in J}=G(Z)
$$

and $G$ is one-to-one. If each active head satisfies

$$
\mathbb E[-\log q_j(Y_j\mid h_T)]\le \varepsilon_j,
\qquad j\in J,
$$

then

$$
I(h_T;Z)\ge H(Z)-\sum_{j\in J}\varepsilon_j.
$$

**Proof.** For each $j$,

$$
H(Y_j\mid h_T)\le \varepsilon_j.
$$

By subadditivity of conditional entropy,

$$
H(Y_J\mid h_T)\le \sum_{j\in J}H(Y_j\mid h_T)\le\sum_{j\in J}\varepsilon_j.
$$

Since $Y_J=G(Z)$ and $G$ is one-to-one,

$$
H(Z\mid h_T)=H(Y_J\mid h_T).
$$

Thus

$$
I(h_T;Z)=H(Z)-H(Z\mid h_T)
\ge H(Z)-\sum_{j\in J}\varepsilon_j.
$$

∎

This theorem is important for realistic semantics: a semantic variable may be expressed over a phrase, not in a single token.

### 5.6 Gradient proposition for Next-K

For linear heads,

$$
L_{K,T}=\sum_{j=1}^{K}\lambda_j\operatorname{CE}(q_j(\cdot\mid h_T),Y_j),
$$

$$
z_j=W_jh_T+b_j,
\qquad q_j=\operatorname{softmax}(z_j).
$$

Then

$$
\nabla_{h_T}L_{K,T}
=\sum_{j=1}^{K}\lambda_j W_j^\top(q_j-e_{Y_j}).
$$

Thus every active future offset sends its error signal to the same prefix state. If $Y_j$ depends on $Z$, the corresponding term is a direct $Z$-dependent gradient. If $Y_j$ is shared or branch-independent, that term does not add semantic pressure.

---

## 6. What is mathematically guaranteed?

The theory guarantees the following:

1. If $K<\tau$, next-$K$ need not expose $Z$ in $h_T$.
2. If $K\ge\tau$ and the offset-$\tau$ target identifies $Z$, low next-$K$ loss forces $h_T$ to contain $Z$.
3. If added future tokens contain additional information about $Z$, the representation requirement is strictly stronger than NTP.
4. If added future tokens contain no additional information about $Z$, there is no guaranteed semantic advantage, although there may still be regularization or optimization effects.

This is a deterministic theoretical advantage, but it is an **objective-level semantic coverage advantage**, not an unconditional performance theorem.

---

## 7. What remains empirical or conditional?

The following claims require experiments or extra optimization assumptions:

1. MTP learns $Z$ faster than NTP.
2. MTP is more sample-efficient.
3. MTP improves downstream long-context performance.
4. Larger $K$ always helps.
5. Separate experts solve horizon-gradient conflict.
6. The synthetic mechanism transfers directly to natural language.

The correct next claim to test is:

> When $K$ first covers $\tau$, MTP reaches a guarded long-horizon semantic score faster or more robustly than horizons that do not cover $\tau$.

---

## 8. Metrics for “better”

Define a guarded long-horizon semantic score

$$
Q_K(t)=\min\{A_K(t),P_K(t),S_K(t)\}.
$$

Here:

- $A_K(t)$: informative-horizon prediction accuracy, $h_T\to Y_\tau$;
- $P_K(t)$: frozen evaluation probe accuracy, $h_T\to Z$;
- $S_K(t)$: branch-swap consistency.

The minimum is deliberate: branch probe alone can be misleading if the native future prediction head does not use the information.

Define time-to-threshold:

$$
T_\theta(K)=\inf\{t:Q_K(t)\ge\theta\text{ for }r\text{ consecutive evaluations}\}.
$$

Define area under curve:

$$
\operatorname{AUC}_Q(K)=\frac{1}{T_{\max}}\int_0^{T_{\max}}Q_K(t)\,dt.
$$

Define local-loss guard:

$$
L_1^{K}(t)\le L_1^{1}(t)+\epsilon_{\mathrm{local}}.
$$

MTP is better in learning efficiency only if it improves $Q_K$ without destroying local next-token learning.

---

## 9. Experimental story closure

The story should close with two experiments.

### Experiment A: K=2 efficiency

Use

$$
Y_1=A,
\qquad
Y_2=S_Z.
$$

This tests whether the K=2 representation guarantee becomes a learning-efficiency advantage.

Expected support:

$$
T_\theta(2)<T_\theta(1)
$$

or

$$
\operatorname{AUC}_Q(2)>\operatorname{AUC}_Q(1)
$$

with local-loss guard passing.

### Experiment B: K=3 first-informative-horizon

Use

$$
Y_1=A,
\qquad
Y_2=C,
\qquad
Y_3=S_Z.
$$

Compare $K=1,2,3$. The first informative horizon is $\tau=3$, so the theory predicts

$$
Q_3(t)>Q_2(t)\approx Q_1(t).
$$

This tests whether the mechanism is truly “coverage of first informative horizon,” not a special property of next-two.

---

## 10. Final story

The self-contained story is:

1. Long-horizon semantics means delayed relevance: an early semantic variable $Z$ determines a future continuation while local context near the prediction site is insufficient.
2. NTP trains $h_T\to Y_1$. If $Y_1$ is shared or locally predictable, NTP at $T$ need not expose $Z$.
3. MTP trains $h_T\to Y_{1:K}$. If some supervised future token contains additional information about $Z$, then low MTP loss forces $h_T$ to contain that information.
4. K=2 is the minimal proof: $Y_1=A$, $Y_2=S_Z$. NTP can be solved branch-blind; next-two cannot.
5. Next-K generalizes this through the first informative horizon $\tau$. The advantage appears when $K\ge\tau$, not merely when $K$ is large.
6. The mathematical guarantee is an objective-level semantic representation advantage.
7. Efficiency, robustness, interference, and real-language transfer require experiments.

The precise headline claim should be:

> MTP is better than NTP for long-horizon semantic learning in the sense that, when its prediction horizon covers future tokens that reveal an early semantic variable not required by the immediate next token, it imposes a provably stronger single-prefix semantic representation requirement on the hidden state.

