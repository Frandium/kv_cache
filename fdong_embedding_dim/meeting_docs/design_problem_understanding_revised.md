# Design: Diagnosing the Origin of High-Learning-Rate SGD Instability in Language Models

Created: 2026-07-01.

## 1. Purpose

This document reframes the SGD-Adam gap as a problem-understanding project, not as an optimizer-tweaking project. Existing optimizer papers show that plain SGD underperforms in LLM pretraining, that Adam moves with much larger effective learning rates, and that targeted clipping, column-wise normalization, or partial momentum can recover much of the gap. Those results are important, but they mostly identify proximal symptoms. The central question here is deeper: what creates the gradient and update anisotropy that makes high-learning-rate SGD unstable in the first place?

**$\text{What is the causal origin of the gradient anisotropy that makes high-learning-rate SGD unstable in language-model pretraining?}$**

Here gradient anisotropy means that different token, feature, layer, and spectral directions receive radically different gradient magnitudes, noise levels, and loss sensitivity. The goal is to identify the source of this anisotropy before designing or endorsing a solution.

## 2. Methodological Position

The optimizer literature often starts from a known failure mode and searches for a stabilizing algorithm. This is natural in classical scientific computing, where the problem structure is usually well understood. LLM pretraining is different. The optimization problem itself is not yet well understood. Therefore, this project must first diagnose the mechanism that produces instability.

The design principle is:

**$\text{diagnose the cause before selecting the optimizer intervention.}$**

Here the cause is the latent mechanism that creates extreme gradient variation, output-head imbalance, singular-value concentration, and rare-feature suppression. Optimizers such as AdamW, SGD-LL, SCALE, label smoothing, spectral clipping, or manifold-aware training should be treated as causal probes. They are not the explanation by themselves.

## 3. Limitation Of Existing Optimizer Explanations

The SGD-Adam gap papers identify several robust empirical facts. SGD under-moves at safe learning rates. Adam has much larger effective learning rates. High-learning-rate SGD diverges because of layer-wise gradient spikes and output-layer per-token-class imbalance. Column-wise normalization and per-token-class clipping improve stability. SCALE shows that column-wise gradient normalization and last-layer momentum are sufficient to recover much of Adam-like behavior with little optimizer memory.

These observations answer the question: what stabilizes training? They do not fully answer the question: why does the gradient field become so uneven that stabilization is necessary?

This document therefore separates two levels of explanation. The first level is the proximal optimizer diagnosis: SGD fails because one global learning rate cannot handle a highly nonuniform gradient field. The second level is the mechanistic representation diagnosis: the gradient field becomes nonuniform because language frequency, next-token prediction, and compositional feature reuse create high-gain singular subspaces that dominate learning.

The second level is the research target.

## 4. Core Causal Hypothesis

The working hypothesis is frequency-driven singular amplification.

**$\text{Zipfian frequency creates repeated common-feature gradients, repeated common-feature gradients grow high-gain singular directions, and high-gain singular directions amplify later gradients enough to destabilize high-learning-rate SGD.}$**

Here Zipfian frequency means that common tokens and common contexts occur much more often than rare ones. Common-feature gradients are coherent updates associated with those frequent patterns. High-gain singular directions are large-singular-value modes of output-facing or attention-routing matrices. Later gradients include gradients from rarer, longer, or more specific features that partially project into those same high-gain modes.

The proposed causal chain is:

**$\text{Zipfian frequency} \to \text{common-feature gradient coherence} \to \text{singular-value growth} \to \text{gradient amplification} \to \text{high-learning-rate SGD instability}.$**

Each arrow is a separate experimental claim. The hypothesis is not proven unless every arrow is supported by profiling or intervention.

## 5. Why Singular Amplification Can Be Both Useful And Dangerous

For a readout matrix $W$ with singular value decomposition $W=U\Sigma V^\top$, a hidden feature component along a right singular vector $v_i$ produces an output effect scaled by the singular value $\sigma_i$.

**$Wh=\sum_i \sigma_i c_i u_i.$**

Here $h=\sum_i c_i v_i$ is the hidden representation decomposed in the right singular-vector basis, $c_i$ is the coefficient of $h$ along $v_i$, $u_i$ is the output-side singular vector, and $\sigma_i$ is the gain of that direction. This follows directly from the definition of the singular value decomposition.

For cross-entropy, the local first-order loss decrease from a gradient-aligned movement along mode $i$ scales with the squared singular value.

**$\Delta L_i \approx -\eta\sigma_i^2\langle u_i,p-e_y\rangle^2.$**

Here $\Delta L_i$ is the first-order loss change from updating through singular mode $i$, $\eta$ is the learning rate, $p$ is the softmax distribution, $e_y$ is the one-hot target vector, and $\langle u_i,p-e_y\rangle$ is the projection of the prediction error onto the output-side singular direction. This is justified by the cross-entropy objective and the SVD-based local descent calculation.

This equation explains why high-singular-value directions are attractive. They reduce loss efficiently. But the same mechanism creates instability when a large global SGD learning rate is used.

**$\|\Delta z_i\| \propto \eta\sigma_i\|P_i(G)\|.$**

Here $\Delta z_i$ is the logit movement caused by the update component in singular mode $i$, $P_i(G)$ is the projection of the gradient onto that mode, and $\sigma_i$ amplifies the resulting logit change. This is justified by the linear readout geometry. A high-gain mode can be useful at moderate step size and destructive at high step size.

## 6. Main Conjectures

### 6.1 Frequency Creates Singular Concentration

Common tokens and common contexts are observed repeatedly. Their gradients accumulate coherently and preferentially grow a small number of output-facing or routing-facing singular directions.

**$\sigma_1,\ldots,\sigma_k \text{ grow faster for matrices that repeatedly serve common predictive features.}$**

Here $\sigma_1,\ldots,\sigma_k$ are the top singular values of a matrix such as $W_{\mathrm{out}}$, $W_Q$, or $W_K$. This statement is a falsifiable conjecture. It must be tested by tracking singular values over training and correlating their growth with feature frequency.

### 6.2 Later Features Reuse High-Gain Directions

Later, rarer, or more specific features do not necessarily occupy independent directions. They may project into the same high-gain subspace because that subspace gives larger loss decrease per unit movement.

**$M_k(e_{\mathrm{extra}})=\sum_{i=1}^k\langle e_{\mathrm{extra}},v_i\rangle^2$** measures how much an extra feature lies in the top-$k$ singular subspace. Here $e_{\mathrm{extra}}$ is the extra-feature direction added by a transition such as $AB\to ABC$, and $v_i$ is the $i$-th right singular vector. This is an experimental definition.

The hypothesis predicts that $M_k(e_{\mathrm{extra}})$ is high for standard cross-entropy training and lower when singular amplification is controlled.

### 6.3 High-Learning-Rate SGD Fails Along The Amplified Modes

High-learning-rate SGD is not expected to fail uniformly. It should fail first through a small number of high-energy directions: layer-wise spike directions, frequent-token output-head directions, and high-gain singular modes.

**$\text{instability events should be spectrally localized rather than isotropic.}$**

Here an instability event means a loss spike, exploding update norm, sudden logit-margin explosion, or abrupt change in top singular values. This claim is falsified if instability events are evenly distributed across spectral bands or unrelated to frequency-shaped directions.

## 7. Profiling Before Intervention

The first experiment must be diagnostic. It should not begin by asking which optimizer wins. It should ask which geometric quantity predicts instability.

The minimum profiling set is as follows.

First, track output-head token-column gradient norms against token frequency. If the hypothesis is correct, frequent tokens should show larger or more frequent gradient energy, especially early in training.

Second, track the singular spectrum of $W_{\mathrm{out}}$, $W_Q$, and $W_K$ over training. If the hypothesis is correct, top singular values should grow in parallel with common-feature learning.

Third, track gradient projection into spectral bands. Define top, middle, and tail singular bands, and measure how much gradient and update energy each optimizer places in each band.

**$B_{\mathrm{top}}(G)=\frac{\|GV_{\mathrm{top}}\|_F^2}{\|G\|_F^2+\epsilon}.$**

Here $B_{\mathrm{top}}(G)$ is the fraction of gradient energy in the top singular subspace, $G$ is the gradient matrix, $V_{\mathrm{top}}$ contains the top right singular vectors, and $\epsilon$ is a small numerical stabilizer. This is an experimental measurement.

Fourth, track common-feature and rare-feature losses separately. The hypothesis predicts that common features become easy early, while rare or longer features either lag or become overly dependent on the common high-gain subspace.

Fifth, log every high-learning-rate SGD instability event and decompose the responsible update into token-column, layer-wise, and spectral-band components.

## 8. Controlled Synthetic Benchmark

The first benchmark should be small enough to inspect every update. The purpose is to create a synthetic language distribution where frequency and nested composition are controlled.

Use nested prefix sequences:

**$A\to AB\to ABC\to ABCD\to ABCDE\to ABCDEF\to ABCDEFG.$**

Here each transition adds a new extra feature while preserving inherited prefix structure. This gives a direct way to measure whether new features occupy independent latent directions or collapse into the same high-gain subspace.

The dataset should include common prefix patterns, rare extensions, and random filler tokens. The frequency ratio should be controlled so that the experiment can separate the effect of frequency from the effect of sequence structure.

The model should start with a one-layer, one-head causal transformer. The hidden dimension should be small enough for full SVD and per-step logging. A dimension such as $d=32$ is sufficient for the first pass.

## 9. Measured Objects

For every checkpoint, measure the following objects.

The first object is the singular spectrum of the output and attention-routing matrices: $\sigma_i(W_{\mathrm{out}})$, $\sigma_i(W_Q)$, and $\sigma_i(W_K)$. These values quantify whether training creates high-gain directions.

The second object is the effective rank of extra-feature directions. Build the extra-feature matrix $E$ whose rows are normalized extra features from transitions such as $A\to AB$ and $AB\to ABC$.

**$\operatorname{erank}(E)=\exp\left(-\sum_i p_i\log p_i\right),\quad p_i=\frac{s_i^2}{\sum_j s_j^2}.$**

Here $s_i$ is the $i$-th singular value of the extra-feature matrix $E$, and $p_i$ is normalized singular energy. This quantity measures how many independent directions are used by the extra features.

The third object is top-$k$ feature mass $M_k(e_{\mathrm{extra}})$. This measures whether different extra features share the same top singular subspace.

The fourth object is token-column gradient imbalance. For the output head, compare $\|G_{:,j}\|_2$ against token frequency $\mathrm{freq}(j)$, assuming the convention that column $j$ corresponds to token $j$. If the implementation uses rows for tokens, use $\|G_{j,:}\|_2$ instead.

The fifth object is instability localization. When high-learning-rate SGD spikes, identify whether the spike is dominated by a small number of token columns, layers, or singular modes.

## 10. Optimizers As Causal Probes

Optimizers should be introduced only after the profiling baseline is clear. Their role is to test the proposed cause.

AdamW is a diagonal metric adapter. It tests whether coordinate-wise scale control compensates for the anisotropic gradient geometry.

SGD-LL is a targeted instability suppressor. It tests whether instability can be reduced by clipping only the layer-wise and per-token-class directions that profiling identifies as pathological.

SCALE-like optimization is a frequency-normalization probe. Column-wise normalization tests whether token-column imbalance is a primary driver of instability. Last-layer momentum tests whether the output head is the dominant variance source.

Label smoothing is a margin-pressure probe. It tests whether reducing overconfident logit-margin growth reduces singular-value concentration.

Spectral clipping or singular-value regularization is a direct mechanism probe. It tests whether capping high-gain modes prevents feature collapse and stabilizes high-learning-rate SGD.

Spherical or hyperbolic representation learning is a geometry probe. It tests whether changing representation geometry allows nested extra features to use more independent directions.

## 11. Intervention Predictions

If label smoothing reduces top singular values and improves extra-feature effective rank while preserving continuation accuracy, then margin pressure is part of the cause. If it improves calibration but destroys fine-grained extra-feature information, then it is a partial solution with an information-erasure tradeoff.

If column-wise normalization reduces output-head gradient imbalance and stabilizes high-learning-rate SGD, then token-frequency imbalance is part of the cause. If it stabilizes training without changing frequency-gradient correlation or spectral concentration, then the current hypothesis is incomplete.

If spectral clipping directly reduces instability while preserving prediction accuracy, then high-gain singular modes are causally implicated. If instability remains after high-gain modes are controlled, then the source may be layer-wise curvature, attention dynamics, or non-spectral noise.

If hyperbolic or spherical representation increases $\operatorname{erank}(E)$ and reduces top-$k$ extra-feature mass, then the low-rank collapse problem is partly geometric. If it only changes norms without improving extra-feature rank, then manifold geometry is not solving the central mechanism.

## 12. Falsification Criteria

The frequency-driven singular-amplification hypothesis is false or incomplete if any of the following occur.

First, high-learning-rate SGD instability occurs before measurable singular concentration or token-column imbalance appears.

Second, frequent-token gradient dominance does not correlate with output-head spectral growth.

Third, rare or later extra features do not project into the common high-gain subspace.

Fourth, capping or regularizing top singular values does not reduce instability or improve extra-feature rank.

Fifth, column-wise normalization improves training but does not change token-column imbalance, spectral-band update mass, or long-tail behavior.

Sixth, Adam's advantage is not associated with stronger movement in weak, undertrained, or low-frequency directions.

## 13. Expected Diagnostic Outcomes

If the hypothesis is correct, the baseline cross-entropy run should show a temporal order: common-feature accuracy improves first, top singular values grow next, later extra features increasingly project into those top modes, and high-learning-rate SGD instability is localized to the same modes or token columns.

If the hypothesis is partially correct, frequency may explain the output head but not the attention matrices. In that case the paper should separate output-head anisotropy from attention-routing anisotropy.

If the hypothesis is wrong, instability will not be localized in high-frequency or high-singular-value directions. Then the project should shift toward curvature, LayerNorm dynamics, residual amplification, or batch-noise effects.

## 14. Visualization Contract

The viewer should support causal diagnosis rather than optimizer ranking.

Panel 1 should show the temporal sequence of common-feature accuracy, rare-feature accuracy, and top singular-value growth. Its purpose is to test whether singular concentration follows common-feature learning.

Panel 2 should show token frequency rank versus output-head gradient column norm. Its purpose is to test whether frequency becomes gradient geometry.

Panel 3 should show gradient and update mass across top, middle, and tail singular bands. Its purpose is to test whether instability is spectrally localized.

Panel 4 should show extra-feature effective rank and top-$k$ mass over training. Its purpose is to test whether later compositional features occupy independent directions or collapse into a shared subspace.

Panel 5 should mark instability events and decompose each event by token column, layer, and spectral band. Its purpose is to identify the immediate cause of high-learning-rate SGD failure.

## 15. Claim Boundary

This design does not claim that Adam works only because of singular-value effects. It does not claim that SGD-LL or SCALE are final solutions. It does not claim that all common high-gain directions are harmful. Common high-gain directions may be useful for efficient learning and generalization.

The narrow claim is:

**$\text{LLM pretraining may create frequency-shaped anisotropic gradient geometry through common-feature singular amplification, and this geometry may be a causal source of high-learning-rate SGD instability.}$**

The stronger claim remains open:

**$\text{long-tail feature failure in large language models is mainly caused by excessive reuse of common high-gain spectral directions.}$**

This stronger claim requires additional experiments on real text and larger models.

## 16. Next Concrete Step

Create one controlled experiment in `docs/experiment_design.md`. The experiment should train a small one-layer causal transformer on a Zipfian nested-prefix next-token task. It should compare standard cross-entropy AdamW, safe-learning-rate SGD, high-learning-rate SGD, SGD-LL-style clipping, SCALE-like column normalization, label smoothing, and spectral clipping.

The primary output is not final validation loss. The primary output is a causal profile: singular-value growth, token-column gradient imbalance, spectral-band update mass, extra-feature effective rank, top-$k$ extra-feature mass, and instability localization.

Only after the causal profile is clear should the project move to larger models or real-text probes.
