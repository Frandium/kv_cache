# End-to-End Report: Long-Horizon Features and Multi-Token Training

Created: 2026-07-03  
Revised: 2026-07-03
Theory-extended: 2026-07-03


## Plain-Language Summary

The main lesson is simple: next-token training can learn to continue a sequence one step at a time, but that does not mean an earlier hidden state contains the information needed to predict a far-future suffix. In our tests, next-token models could roll out the right sequence, but failed when we asked one prefix state to predict the whole future suffix directly.

Multi-token training changed this. When the model was trained to predict several future tokens from the same prefix state, that state became useful for long-horizon prediction. On the harder branch dataset, the model had to remember an earlier branch token and use it after a shared prefix. Multi-token models did this; fixed next-token models did not.

The spectrum analysis suggests the reason: successful models do not just memorize the suffix. They align branch-sensitive directions in the hidden state, or decoder slot states, with the output directions for the correct future suffix tokens. In plain terms, the model learns to point the representation in the direction of the right future answer.

The current evidence supports multi-token training, but it does not yet prove that curriculum is better than training all future positions from the beginning. Both curriculum and full-horizon training worked. The next step is to repeat the branch experiment across more seeds and then make the branch rule more compositional.

## 1. Executive Summary

The original hypothesis was that rare features are hard to learn. The experiments refine this into a more precise statement: the difficult object is not rarity by itself, and it is not composition by itself. The difficult object is a long-horizon feature. A long-horizon feature is information that appears early in the sequence, must be preserved through intervening tokens, and must later be used to select the correct continuation when the local context alone is ambiguous.

This distinction matters because ordinary next-token training can succeed on sequence rollout while still failing to make an earlier hidden state contain the information needed for a far-future decision. A model can learn local transitions such as $D \to E$, $E \to F$, and $F \to G$ without representing, at the hidden state after $D$, the whole continuation $E\,F\,G\,H\,I\,J\,.$. This is not a contradiction. It means that next-token loss can teach a model to walk through a trajectory without necessarily teaching it to forecast the trajectory from an earlier state.

The experiments support this interpretation. On the simple contiguous dataset, standard next-token training passes one-step rollout but fails the single-prefix future test. Multi-token training passes the single-prefix future test. On the long-horizon branch dataset, the result is stronger: fixed next-token training again passes local rollout, but it fails when asked to choose the branch-specific suffix from the hidden state after the shared prefix $A\,B\,C\,D$. Multi-token training succeeds and passes the branch-swap intervention, which means that the model has not merely memorized the shared local prefix; it has retained the earlier branch token and used it to choose the later suffix.

The added theoretical analysis proves the precise separation behind these experiments. Low next-token loss does not identify whether an earlier hidden state is a decodable forecast state; a branch-blind representation has an unavoidable loss lower bound on branch-specific suffix prediction; and multi-token loss supplies a direct gradient from future suffix errors to the earlier prefix representation.

The geometric analysis gives a plausible mechanism. The successful models do not merely have sharper spectra. More importantly, the branch-sensitive hidden-state differences become aligned with the output directions of the correct far-future suffix tokens. In the direct-head models, this appears as strong alignment between branch direction and suffix output direction. In the latent-decoder models, this appears as branch separation in future slots and strong alignment between slot differences and suffix output directions. The current evidence therefore supports the following working conjecture: multi-token training creates direct gradient pressure for a single prefix representation to become predictive of multiple future tokens, whereas next-token training only creates local pressure for the next token.

## 2. The Central Question

The central question is whether ordinary next-token training makes one hidden state carry information about many future tokens. More concretely, after the model reads a prefix, can the hidden state at the final prefix position predict a later suffix, or does it only support the next local transition?

Let the token sequence be $x_1, x_2, \ldots, x_T$. At position $i$, the model produces a hidden state $h_i$. Ordinary next-token training optimizes the prediction of $x_{i+1}$ from $h_i$. Multi-token training asks the same $h_i$ to predict several future tokens, such as $x_{i+1}$, $x_{i+2}$, $\ldots$, $x_{i+H}$.

The ordinary objective can be written as **$\mathcal{L}_{\mathrm{NT}} = \sum_i \mathrm{CE}(p_\theta(\cdot \mid x_{\le i}), x_{i+1})$**. Here, $\mathcal{L}_{\mathrm{NT}}$ is the next-token loss, $\mathrm{CE}$ is cross-entropy, $p_\theta$ is the model distribution with parameters $\theta$, $x_{\le i}$ is the prefix up to position $i$, and $x_{i+1}$ is the next token. This equation is a definition of the standard language-model training objective.

The multi-token objective can be written as **$\mathcal{L}_{\mathrm{MT}} = \sum_i \sum_{k=1}^{H} w_k\,\mathrm{CE}(p_{\theta,k}(\cdot \mid x_{\le i}), x_{i+k})$**. Here, $H$ is the maximum prediction horizon, $k$ is the future offset, $w_k$ is the loss weight for offset $k$, and $p_{\theta,k}$ is the model's prediction head or decoding slot for the token $k$ steps into the future. This equation is a definition of the multi-token auxiliary objective used to make one prefix representation predictive of several future positions.

The key experimental question is not whether the model can eventually generate the correct sequence by autoregressive rollout. The stricter question is whether the hidden state at an earlier prefix position already contains enough information to predict the later suffix. These are different tests. Autoregressive rollout allows the model to update its hidden state after every generated token. The single-prefix future test does not. It asks whether one earlier representation contains the future-relevant information before the suffix has begun.

## 3. Why Next-Token Training Can Be Insufficient

Next-token training is powerful because it decomposes sequence modeling into many local prediction problems. However, this decomposition can hide a failure mode. If a sequence is locally deterministic, the model can learn the transition graph without learning a global forecast representation.

For example, in the sequence $A\,B\,C\,D\,E\,F\,G\,H\,I\,J\,.$, the model can succeed during rollout by learning local transitions. Once it predicts $E$ after $D$, the new context contains $E$, and then predicting $F$ is easier. Once it predicts $F$, the new context contains $F$, and then predicting $G$ is easier. In this regime, the model does not need the hidden state after $D$ to encode the entire future suffix. It only needs enough information to predict $E$.

This creates an identifiability gap in the training signal. Next-token loss does not distinguish between a model that represents the future suffix at $D$ and a model that only knows the next local step at $D$. Both can have low next-token loss. Therefore, passing the autoregressive rollout test is not sufficient evidence that the earlier hidden state contains the long-horizon feature.

The long-horizon branch dataset is designed to close this loophole. In that dataset, the same local prefix $A\,B\,C\,D$ appears under multiple branches. The correct suffix depends on an earlier branch token, not on $A\,B\,C\,D$ alone. Therefore, a model that only learns local transitions from $D$ cannot know which suffix should follow. To pass the single-prefix suffix test and the branch-swap test, the model must preserve branch information across the filler and the shared prefix.

This is the core conceptual refinement. Rarity can make a feature hard because there are fewer gradient updates. Composition can make a feature hard because several pieces of information must be combined. Long horizon is different: the information is available, but the loss must force the model to keep it until it becomes useful. If the training objective only rewards the next local token, it may not create enough pressure for the earlier hidden state to retain and expose that information.


## 4. Theoretical Analysis: What the Experiments Can and Cannot Prove

The empirical result becomes stronger if we separate three mathematical claims. The first claim is an identifiability claim: low next-token loss does not identify whether an earlier hidden state is already predictive of a far-future suffix. The second claim is an impossibility claim: if the representation at the shared-prefix decision point discards the branch variable, then no direct future head can reliably predict the branch-specific suffix from that representation. The third claim is an optimization claim: multi-token loss creates a direct gradient from far-future suffix errors to the earlier prefix representation, while next-token-only loss does not.

These claims do not require a large-language-model assumption. They follow from the supervised objectives and from the construction of the branch dataset. The Transformer architecture matters for how the model may implement a solution, but the separation between local rollout and single-prefix future prediction is already visible at the level of the training objective.

### 4.1 Formal Setup

Let the branch variable be $B \in \{1,\ldots,m\}$. In the current branch dataset, $m=3$. For each branch $b$, the sequence has an early branch token $\mathrm{BR}_b$, a filler block, a shared prefix $U=A\,B\,C\,D$, another filler block, and a branch-specific suffix $S_b=(s_{b,1},\ldots,s_{b,H})$. The shared prefix $U$ is identical for all branches, while the suffixes $S_1,\ldots,S_m$ are different.

Let $i_D$ denote the position after the model has read the shared prefix ending in $D$. Let $h_D^{(b)}$ be the hidden state at that position under branch $b$. A single-prefix future evaluator uses $h_D^{(b)}$ to predict future suffix tokens. For a direct-head model, the offset-$k$ prediction has the form $q_k(y\mid h_D^{(b)})$. For a decoder model, $q_k$ is produced by a learned future slot conditioned on $h_D^{(b)}$. In both cases, the essential question is whether the information in $h_D^{(b)}$ is sufficient for predicting $s_{b,k}$.

The branch dataset is constructed so that the local suffix decision is branch-dependent. Formally, for at least one future offset $k$, the target token $s_{b,k}$ is not constant in $b$. In the strongest clean case, the tokens $\{s_{1,k},\ldots,s_{m,k}\}$ are all distinct. This is exactly the setting in which the branch-swap test is meaningful: changing only $\mathrm{BR}_b$ should change the future prediction, even though the later visible shared prefix remains fixed.

### 4.2 Proposition 1: Autoregressive Rollout Does Not Imply Single-Prefix Decodability

Proposition 1 states that successful autoregressive rollout is not sufficient evidence that $h_D^{(b)}$ contains linearly or decodably accessible information about the whole future suffix.

**$\mathcal{L}_{\mathrm{NT}}=\sum_t \mathrm{CE}(p_\theta(\cdot\mid x_{\le t}),x_{t+1})$**.

Here, $\mathcal{L}_{\mathrm{NT}}$ is the next-token objective, $t$ ranges over token positions, $p_\theta(\cdot\mid x_{\le t})$ is the model's next-token distribution after prefix $x_{\le t}$, and $x_{t+1}$ is the next token. This equation is the definition of the objective optimized by standard autoregressive training.

The key point is that $\mathcal{L}_{\mathrm{NT}}$ supervises the conditional distribution at each position after the model has already consumed all earlier tokens up to that position. During rollout, the model is allowed to update its internal state after every generated token. Therefore, the hidden state used to predict a late suffix token is not $h_D^{(b)}$; it is a later hidden state computed after additional common gap tokens and possibly after earlier suffix tokens.

By contrast, the single-prefix future test evaluates distributions of the form $q_k(\cdot\mid h_D^{(b)})$. These distributions ask whether the earlier state after $D$ already exposes information about $s_{b,k}$. This is a different conditional object from the ordinary next-token distribution at the later position.

The formal separation is immediate. The next-token objective constrains $p_\theta(\cdot\mid x_{\le t})$ at positions $t$, but it does not directly constrain $q_k(\cdot\mid h_D^{(b)})$ for $k>1$ unless the model is trained with an auxiliary future-offset loss. Therefore, there can be two models with the same next-token behavior and different single-prefix future behavior. One model may organize its computation so that $h_D^{(b)}$ is already a forecast state. Another model may defer the branch-dependent computation until later positions, for example by attending back to $\mathrm{BR}_b$ when the suffix decision becomes locally necessary. Both can have low next-token loss, but only the first must pass the single-prefix future test.

This proves the logical claim: autoregressive success does not imply single-prefix decodability. It does not prove that every next-token-trained model will fail the single-prefix test. Rather, it proves that next-token success alone is insufficient evidence for long-horizon representation at the earlier state. This is exactly why the single-prefix suffix test is needed.

### 4.3 Proposition 2: Branch-Blind Representations Have an Unavoidable Error Lower Bound

Proposition 2 states that if the hidden state after the shared prefix discards the branch variable, then no direct future predictor can reliably choose branch-specific suffixes.

Call a representation branch-blind at the decision point if $h_D^{(1)}=h_D^{(2)}=\cdots=h_D^{(m)}$. More generally, the same argument applies if the future predictor cannot distinguish the branches, meaning $q_k(\cdot\mid h_D^{(1)})=\cdots=q_k(\cdot\mid h_D^{(m)})$ for the relevant offset $k$.

Assume the branch variable is uniformly distributed and that, at offset $k$, the target tokens $s_{1,k},\ldots,s_{m,k}$ are distinct. If the representation is branch-blind, then the predictor must use one common distribution $q_k(\cdot)$ for all branches. The expected cross-entropy at this offset is **$\mathcal{L}_{k}^{\mathrm{blind}}=\frac{1}{m}\sum_{b=1}^{m}-\log q_k(s_{b,k})$**. Here, $\mathcal{L}_{k}^{\mathrm{blind}}$ is the expected offset-$k$ loss under a branch-blind representation, $m$ is the number of branches, and $q_k(s_{b,k})$ is the probability assigned to the branch-$b$ target token.

The best branch-blind distribution puts probability $1/m$ on each of the $m$ distinct target tokens. Therefore the minimum possible loss is **$\min_{q_k}\mathcal{L}_{k}^{\mathrm{blind}}=\log m$**. Here, the minimization ranges over all probability distributions $q_k$ over the vocabulary, and $\log m$ is the entropy of a uniform distribution over the $m$ branch-specific targets.

The proof follows from Gibbs' inequality. The target distribution induced by a uniformly random branch is uniform over the distinct tokens $\{s_{1,k},\ldots,s_{m,k}\}$. Cross-entropy is minimized when the prediction distribution equals the target distribution. The entropy of this target distribution is $\log m$. Thus no branch-blind predictor can drive the offset-$k$ cross-entropy below $\log m$.

There is also an accuracy version. Because one common prediction distribution must serve all branches, the best deterministic top-1 predictor can choose at most one of the $m$ distinct branch-specific tokens. Thus its expected top-1 accuracy is at most $1/m$. For the current three-branch dataset, this upper bound is $1/3$ at any offset where the branch-specific targets are distinct.

This proposition explains why the branch-swap test is theoretically diagnostic. If changing only $\mathrm{BR}_b$ changes the correct suffix, then a representation that fails to carry branch information cannot pass the test. A successful model must either make $h_D^{(b)}$ branch-dependent or make the decoder computation conditioned on $h_D^{(b)}$ branch-dependent. In either case, branch information must be available to the single-prefix future predictor.

### 4.4 Proposition 3: Multi-Token Loss Creates a Direct Gradient From Future Tokens to the Prefix State

Proposition 3 states that multi-token training changes the credit-assignment path. It attaches future suffix errors directly to the prefix state $h_i$.

For a direct-head model, let the offset-$k$ logits be $z_{i,k}=W_k h_i+b_k$, and let $p_{i,k}=\mathrm{softmax}(z_{i,k})$. The multi-token loss at position $i$ is **$\mathcal{L}_{i}^{\mathrm{MT}}=\sum_{k=1}^{H}w_k\,\mathrm{CE}(p_{i,k},x_{i+k})$**. Here, $W_k$ is the offset-$k$ output matrix, $b_k$ is the offset-$k$ bias, $p_{i,k}$ is the predicted distribution for token $x_{i+k}$, and $w_k$ is the loss weight for that future offset.

The gradient with respect to the prefix hidden state is **$\nabla_{h_i}\mathcal{L}_{i}^{\mathrm{MT}}=\sum_{k=1}^{H}w_k\,W_k^\top(p_{i,k}-e_{x_{i+k}})$**. Here, $e_{x_{i+k}}$ is the one-hot vector for the true future token $x_{i+k}$, and $W_k^\top(p_{i,k}-e_{x_{i+k}})$ is the error signal from offset $k$ mapped back into hidden-state space.

This equation follows directly from differentiating the softmax cross-entropy loss through the linear head $z_{i,k}=W_k h_i+b_k$. It proves that every supervised future offset with $w_k>0$ sends a direct gradient to the same prefix representation $h_i$. Therefore, if a far-future suffix token is branch-dependent, its prediction error directly pushes $h_i$ to preserve and expose branch information.

For next-token-only training, the corresponding gradient is **$\nabla_{h_i}\mathcal{L}_{i}^{\mathrm{NT}}=W_1^\top(p_{i,1}-e_{x_{i+1}})$**. Here, only the offset-$1$ prediction error at position $i$ contributes to the gradient on $h_i$. If $x_{i+1}$ is a shared or locally determined token, this gradient can be nearly identical across branches. In that case, it gives little or no direct pressure for $h_i$ to encode which branch-specific suffix will appear many steps later.

This proves the optimization distinction. Multi-token training does not merely add more parameters. It changes which future errors are allowed to shape the earlier hidden state. This is why the fixed next-token baselines are critical: they have the architecture but not the multi-offset gradient terms, and they fail the single-prefix suffix test.

### 4.5 Proposition 4: Alignment With Output Directions Is a Sufficient Condition for Correct Branch Selection

The geometric measurements in the report can also be connected to a simple margin condition. Suppose the model must choose between branch-$a$ suffix token $y_a$ and branch-$b$ suffix token $y_b$ at some future offset. For a linear readout, the logit difference is **$\ell_{y_a}(h)-\ell_{y_b}(h)=(w_{y_a}-w_{y_b})^\top h+(b_{y_a}-b_{y_b})$**. Here, $\ell_y(h)$ is the logit for token $y$, $w_y$ is the token output vector, $h$ is the hidden state or decoder slot state, and $b_y$ is the token bias.

Now compare the hidden states for two branches using $\Delta h_{a,b}=h^{(a)}-h^{(b)}$. If moving from branch $b$ to branch $a$ increases the correct logit margin, then the branch direction is useful for prediction. The margin increase is **$\Delta M_{a,b}=(w_{y_a}-w_{y_b})^\top \Delta h_{a,b}$**. Here, $\Delta M_{a,b}$ is the change in the logit margin favoring $y_a$ over $y_b$ caused by the branch-sensitive representation difference.

If $\Delta M_{a,b}>0$, the branch direction pushes the representation toward the correct token for branch $a$ relative to branch $b$. If the margin is large enough to overcome biases and competing-token margins, the top-1 prediction is correct. Thus positive alignment between $\Delta h_{a,b}$ and $w_{y_a}-w_{y_b}$ is not merely a descriptive statistic; it is a sufficient geometric mechanism for branch-sensitive suffix selection.

This proposition justifies the alignment analysis. A model can store branch information in a way that is not useful to the output head. Such information may exist in hidden space but fail to affect logits. The useful quantity is readout-effective branch information: the component of the branch-sensitive direction that projects onto the output direction separating the correct and incorrect suffix tokens.

### 4.6 Theoretical Conclusion

The theoretical result is not that next-token training must always fail. The result is more precise: next-token loss does not by itself force the hidden state at an earlier shared-prefix position to be a decodable forecast state for far-future branch-specific suffixes. Multi-token loss does impose that pressure, because future-token errors are attached directly to the earlier representation.

The branch dataset turns this distinction into a falsifiable test. If a model passes autoregressive rollout but fails single-prefix suffix prediction and branch-swap consistency, then it has learned a local rollout policy without exposing the long-horizon feature at $h_D$. If a model passes both, then the earlier representation or the decoder slots conditioned on it must contain branch-relevant information that is usable for future-token prediction. The empirical tables should therefore be read as tests of these theoretical claims, not merely as benchmark scores.


## 5. Model Variants and What Each Variant Tests

The experiments compare three model families. These models are not merely implementation alternatives; each one isolates a different part of the causal story.

The standard language model is the ordinary next-token baseline. It maps a hidden state to one LM head and predicts only the next token. This model is useful because it tells us whether the dataset can be learned by ordinary autoregressive training. However, it does not naturally expose separate predictions for offsets $2, 3, \ldots, H$ from the same hidden state.

The direct multi-offset head model is the cleanest representation test. It takes the same hidden state $h_i$ and attaches one prediction head per future offset. Head 1 predicts $x_{i+1}$, head 2 predicts $x_{i+2}$, and so on. Because each future prediction is made directly from the same hidden state, success means that the hidden state itself contains information useful for the future suffix. This architecture gives the most direct evidence about whether multi-token training changes the information content and geometry of the prefix representation.

The transformer latent decoder is a more flexible version of the same idea. It takes $h_i$, combines it with learned future slots, and uses a small decoder to predict future tokens. The decoder receives no ground-truth future tokens. Slot 1 predicts the first future token, slot 2 predicts the second future token, and so on. This model tests whether a small learned computation can unpack future information from the prefix hidden state. If the decoder succeeds only under multi-token loss, then the result suggests that the base representation and the future slots are jointly shaped by the auxiliary objective.

A critical baseline is the fixed next-token baseline inside the multi-token architectures. In this condition, the architecture has the direct heads or decoder machinery, but training uses only offset 1. This controls for model capacity. If the fixed next-token versions fail while the curriculum and full-horizon versions succeed, then the success cannot be attributed simply to adding heads or adding a decoder. It must come from the multi-offset training signal.

## 6. Dataset 1: Simple Contiguous Sequence

The first dataset contains the deterministic pattern $A\,B\,C\,D\,E\,F\,G\,H\,I\,J\,.$ mixed with random filler tokens. The main test is whether the model can take the prefix $A\,B\,C\,D$ and predict the future suffix $E\,F\,G\,H\,I\,J\,.$ from the same prefix position.

This dataset is useful because it separates one-step rollout from single-prefix future prediction in the simplest possible setting. It is also intentionally limited. Since the continuation after $D$ is locally deterministic, a next-token model can learn the sequence as a chain of local transitions. Therefore, failure on the single-prefix future test is informative, but success on rollout is not strong evidence of long-horizon memory.

Three seeds were run: 951, 952, and 953. The remote result folder is `/home/zhicheng/representation-space/research-problems/sgd_adam_gap_representation_geometry/results/deterministic_curriculum_v2_multiseed`.

| Method | One-step test | Single-prefix future test | Worst future-token probability |
|---|---:|---:|---:|
| standard next-token LM | passed 3/3 | failed 0/3 | $0.000018$ |
| direct heads, fixed next-token | passed 3/3 | failed 0/3 | $0.0078$ |
| direct heads, curriculum | passed 3/3 | passed 3/3 | $0.9944$ |
| direct heads, full horizon | passed 3/3 | passed 3/3 | $0.9934$ |
| decoder, fixed next-token | passed 3/3 | failed 0/3 | $0.0071$ |
| decoder, curriculum | passed 3/3 | passed 3/3 | $0.9938$ |
| decoder, full horizon | passed 3/3 | passed 3/3 | $0.9908$ |

The result is clean. Every method can pass the one-step test, but only the multi-token methods pass the single-prefix future test. This supports the claim that next-token training can learn the local chain without making the hidden state after $D$ directly predictive of the whole future suffix. Multi-token training changes the representation so that the same prefix state supports multiple future predictions.

The correct interpretation is limited but important. Dataset 1 does not prove that the model has learned branch-sensitive memory, because there is no branch ambiguity. It proves a narrower claim: even in a deterministic contiguous sequence, ordinary next-token learning and single-prefix future prediction are empirically separable.

## 7. Dataset 2: Long-Horizon Branch Dataset

The second dataset is the stronger test. Each example begins with a branch token, then filler, then a shared prefix, then more filler, and finally a branch-specific suffix. The examples have the following structure:

$\mathrm{BR}_1\,\mathrm{F00}\,\ldots\,\mathrm{F07}\,A\,B\,C\,D\,\mathrm{R00}\,\mathrm{R01}\,\mathrm{R02}\,\mathrm{R03}\,E\,F\,G\,H\,I\,J\,K\,L\,.$

$\mathrm{BR}_2\,\mathrm{F00}\,\ldots\,\mathrm{F07}\,A\,B\,C\,D\,\mathrm{R00}\,\mathrm{R01}\,\mathrm{R02}\,\mathrm{R03}\,M\,N\,O\,P\,Q\,S\,T\,U\,.$

$\mathrm{BR}_3\,\mathrm{F00}\,\ldots\,\mathrm{F07}\,A\,B\,C\,D\,\mathrm{R00}\,\mathrm{R01}\,\mathrm{R02}\,\mathrm{R03}\,V\,W\,Y\,Z\,AA\,AB\,AC\,AD\,.$

The decisive property is that $A\,B\,C\,D$ is shared across branches. The local prefix after $D$ is therefore ambiguous. To know which suffix should follow, the model must retain information from the earlier branch token $\mathrm{BR}_k$.

The main test stops after $D$. The prompt is $\mathrm{BR}_k\,\mathrm{F00}\,\ldots\,\mathrm{F07}\,A\,B\,C\,D$, and the model is asked to predict the branch-specific suffix from that same position. The branch-swap test then changes only the branch token while keeping the filler and shared prefix fixed. Good behavior means that the predicted suffix changes when $\mathrm{BR}_1$ is replaced by $\mathrm{BR}_2$ or $\mathrm{BR}_3$.

One full seed was run: 971. The remote result folder is `/home/zhicheng/representation-space/research-problems/sgd_adam_gap_representation_geometry/results/long_horizon_branch_v1_seed971`.

| Method | One-step rollout | Single-prefix suffix test | Worst suffix probability | Branch-swap test |
|---|---:|---:|---:|---:|
| standard next-token LM | passed | N/A | N/A | N/A |
| direct heads, fixed next-token | passed | failed | $0.0023$ | failed |
| direct heads, curriculum | passed | passed | $0.9890$ | passed |
| direct heads, full horizon | passed | passed | $0.9966$ | passed |
| decoder, fixed next-token | passed | failed | $3.6\times 10^{-10}$ | failed |
| decoder, curriculum | passed | passed | $0.9896$ | passed |
| decoder, full horizon | passed | passed | $0.9863$ | passed |

This result is more meaningful than Dataset 1 because the correct suffix is not determined by the local prefix. The fixed next-token models can learn local rollout, but they fail when the suffix must be selected from the earlier branch token. The multi-token models pass both the single-prefix suffix test and the branch-swap test. This implies that the hidden representation after the shared prefix contains branch-relevant information and exposes it in a form useful for predicting future suffix tokens.

The branch-swap test is especially important because it is an intervention, not just an accuracy measurement. If changing only $\mathrm{BR}_k$ changes the predicted suffix while all later visible tokens remain fixed, then the model's prediction is causally sensitive to the earlier branch token. This is precisely the behavior expected from a long-horizon representation.

## 8. What the Experiments Actually Falsify

The experiments falsify a simple but tempting explanation: that adding extra heads or adding a decoder is enough. This explanation is not consistent with the fixed next-token baselines. Those models have the extra architectural capacity, but they fail the single-prefix future tests. Therefore, the decisive factor is not merely capacity; it is the loss applied to multiple future offsets.

The experiments also weaken the claim that autoregressive success is sufficient evidence of long-horizon representation. The fixed next-token models pass one-step rollout but fail the single-prefix suffix tests. Therefore, one-step rollout and single-prefix future prediction are empirically different behaviors.

The experiments do not yet prove that curriculum training is better than full-horizon training. Both curriculum and full-horizon training work well in the current evidence. The clean conclusion is that multi-token supervision matters. Curriculum is a viable implementation, but the current results do not establish it as superior.

The experiments also do not yet prove that the same mechanism governs all natural-language long-range reasoning. The datasets are synthetic and the rules are simple. The value of the experiment is that it isolates a controlled mechanism: when future-relevant information appears early and local later context is ambiguous, next-token training may not force the earlier hidden state to preserve and expose that information, while multi-token training can.

## 9. Geometry: Why Multi-Token Training Works

The geometric result suggests that success is not just a matter of making hidden states different. The useful difference must point in the right output-facing directions.

For a linear output head, a token logit can be approximated as **$\ell_y(h) = w_y^\top h + b_y$**. Here, $\ell_y(h)$ is the logit for token $y$, $w_y$ is the output vector for token $y$, $h$ is the hidden state, and $b_y$ is the token bias. This equation is a definition of a linear readout. It implies that a hidden-state difference matters for token $y$ only to the extent that it has projection on $w_y$.

For two branches $a$ and $b$, the relevant hidden-state difference can be written as **$\Delta h_{a,b} = h^{(a)}_{D} - h^{(b)}_{D}$**. Here, $h^{(a)}_{D}$ is the hidden state after the shared prefix under branch $a$, and $h^{(b)}_{D}$ is the corresponding hidden state under branch $b$. This equation is a definition of the branch-sensitive direction at the decision point.

The branch difference is useful for choosing suffix token $y_a$ over suffix token $y_b$ when **$(w_{y_a} - w_{y_b})^\top \Delta h_{a,b} > 0$**. Here, $w_{y_a} - w_{y_b}$ is the output direction that separates the two candidate suffix tokens, and $\Delta h_{a,b}$ is the branch-sensitive hidden-state difference. This inequality is a direct consequence of the linear logit definition: it says that the branch difference increases the correct token logit relative to the incorrect token logit.

This gives a precise interpretation of the alignment measurements. It is not enough for the model to remember the branch token in some arbitrary subspace. The remembered information must be aligned with the output directions that select the correct future suffix. Multi-token training creates this pressure because far-future suffix losses backpropagate to the prefix state. Fixed next-token training lacks this direct pressure for offsets beyond 1.

### 9.1 Direct-Head Geometry

For direct heads, the result is very clear.

| Method | Single-prefix suffix test | Alignment between branch direction and suffix output direction | Far-future head size |
|---|---:|---:|---:|
| fixed next-token | failed | $0.037$ | $1.46$ |
| curriculum | passed | $0.863$ | $6.61$ |
| full horizon | passed | $0.903$ | $8.34$ |

The fixed next-token model has almost no alignment between branch direction and far-future suffix output direction. This explains why it fails. It may encode some information about the branch, but that information is not expressed in a way that the far-future output heads can use.

The curriculum and full-horizon models have strong alignment. Their branch-sensitive hidden-state differences point toward the correct suffix output directions. This is the expected geometry if multi-token loss has forced the hidden state after $D$ to become a usable forecast representation.

The larger far-future head size is also meaningful, but it should not be overinterpreted alone. A large head norm without alignment would not solve the task. The decisive quantity is readout-effective branch information: branch differences must project onto the output directions that separate the suffix tokens.

### 9.2 Latent-Decoder Geometry

For the transformer latent decoder, the relevant object is not only the base hidden state but also the future slot representation produced by the decoder. The analysis therefore measures branch separation in suffix slots and alignment between slot differences and suffix output directions.

| Method | Single-prefix suffix test | Alignment between slot difference and suffix output direction | Branch separation in suffix slots |
|---|---:|---:|---:|
| fixed next-token | failed | $0.130$ | $0.89$ |
| curriculum | passed | $0.758$ | $8.96$ |
| full horizon | passed | $0.879$ | $7.55$ |

The fixed next-token decoder has weak branch separation in suffix slots and weak alignment with suffix output directions. The decoder architecture exists, but the loss does not force future slots to become branch-specific. The curriculum and full-horizon decoders show strong branch separation and strong alignment, which is consistent with the decoder learning to unpack branch information from the prefix hidden state into future-specific slots.

This is an important result because it shows that multi-token training can act through more than one architectural pathway. In the direct-head model, the hidden state itself must linearly support future predictions. In the decoder model, the hidden state can seed a structured future representation. In both cases, the successful models align branch-sensitive information with future-token readout directions.

## 10. Updated Causal Story

The current causal story is as follows. In the branch dataset, the early branch token determines the later suffix, but the shared prefix $A\,B\,C\,D$ erases local distinguishability. Ordinary next-token training rewards the model for predicting the immediate next token at each position. This is enough to learn local rollout, especially after the suffix has already started. However, it does not directly reward the hidden state after $D$ for predicting suffix tokens several steps ahead.

Multi-token training changes the credit assignment path. The far-future suffix losses are attached directly to the hidden state after the prefix. During training, gradients from suffix tokens flow to the same prefix representation. This encourages the model to keep branch information through the filler and shared prefix, and to orient that branch information toward output directions that distinguish the correct suffix from competing suffixes.

The mechanism can be summarized as a falsifiable conjecture: multi-token training improves long-horizon feature learning because it converts delayed relevance into immediate supervision at the prefix state. The hidden state is no longer rewarded only for the next token; it is rewarded for being a compact forecast state for a future suffix.

This conjecture predicts that the benefit should be largest when three conditions hold simultaneously. First, the decisive information appears early. Second, there is a long gap or distracting filler before the information is used. Third, the local context near the decision point is ambiguous, so the correct future cannot be inferred from local transitions alone. If any of these conditions is removed, the advantage of multi-token training should shrink.

## 11. What We Can Claim Now

The current evidence supports the following claim: on these synthetic tasks, multi-token training makes one prefix position much better at predicting long future suffixes than next-token-only training.

The current evidence also supports a stronger branch-specific claim: on the long-horizon branch dataset, multi-token training makes the model use the earlier branch token to choose the correct later suffix from the hidden state after a shared ambiguous prefix.

The current evidence supports a geometric claim: successful multi-token models align branch-sensitive representation differences with future-token output directions, while fixed next-token baselines do not.

The current evidence does not yet support the claim that curriculum is better than full-horizon training. Full-horizon training performs at least as well in the reported runs. The clean statement is that both curriculum and full-horizon multi-token training succeed, while fixed next-token training fails.

The current evidence does not yet support a broad claim about natural language. The synthetic tasks isolate a mechanism. They do not prove that the same objective will solve long-range reasoning, discourse consistency, tool-use planning, or factual recall in large-scale language models. Those are plausible extensions, not established conclusions.

## 12. Limitations

The most important limitation is that the branch dataset currently has only one full seed. The result is strong, but it must be repeated across seeds before it becomes a stable empirical claim. At minimum, seeds 971, 972, and 973 should be run with identical evaluation metrics.

The second limitation is that the branch rule is a simple lookup rule. The model only needs to preserve one branch token and retrieve one corresponding suffix. This is a useful controlled test, but it is not yet a test of rich compositional reasoning. A stronger dataset would require the suffix to be computed from multiple earlier keys, for example where one key chooses a base suffix and another key transforms it.

The third limitation is that the models are small and the datasets are synthetic. This is appropriate for mechanism isolation, but it means that the current result should be interpreted as a controlled representation-learning finding, not as a direct claim about production-scale language models.

The fourth limitation is that the spectrum analysis is suggestive rather than complete. The strongest evidence is alignment, not merely singular-value growth or head norm. Future analysis should measure how much suffix logit advantage is explained by the top singular subspace, by the residual subspace, and by branch-sensitive directions directly.

## 13. Next Experiments

The immediate next experiment should repeat the long-horizon branch dataset across seeds 971, 972, and 973. The falsification criterion is simple: if fixed next-token baselines frequently pass the single-prefix suffix test, or if multi-token models frequently fail it, then the current conclusion is not stable. The expected pattern is that fixed next-token models pass local rollout but fail single-prefix suffix prediction, while curriculum and full-horizon multi-token models pass both.

The second experiment should increase the horizon length and filler entropy. The conjecture predicts that next-token-only training should become increasingly fragile as the gap between branch token and suffix decision grows, especially when filler tokens are diverse enough to prevent trivial position-based shortcuts. Multi-token training should degrade more slowly if it truly creates forecast-oriented prefix states.

The third experiment should introduce compositional branch rules. Instead of $\mathrm{suffix}=\mathrm{lookup}(\mathrm{branch})$, use a rule such as $\mathrm{suffix}=f(\mathrm{branch\_key}_1,\mathrm{branch\_key}_2)$. For example, one branch key could choose a symbol family, and another key could choose a permutation, reversal, or offset transformation. This would test whether the model can combine multiple earlier pieces of information rather than merely remember one branch identifier.

The fourth experiment should perform a causal representation intervention. After training, compute branch-sensitive directions at the shared-prefix position. Then add or subtract these directions from hidden states and measure whether the predicted suffix changes accordingly. If the branch direction is causally responsible, moving a hidden state from the $\mathrm{BR}_1$ region toward the $\mathrm{BR}_2$ region should increase the probability of the $\mathrm{BR}_2$ suffix and decrease the probability of the $\mathrm{BR}_1$ suffix.

The fifth experiment should profile readout-effective rank. For each model, decompose the future-token logit advantage into contributions from increasing fractions of the singular spectrum. This should be reported as a cumulative curve rather than a single top-$k$ number. The key question is whether successful multi-token training uses a broader set of dimensions or simply amplifies a small number of output-facing directions.

## 14. Recommended Metrics for the Next Report

The next report should keep the current accuracy tables, but it should add a small number of mechanism-focused metrics. The first metric is single-prefix suffix accuracy, which is the central behavioral test. The second metric is worst suffix-token probability, because it detects brittle failures hidden by average accuracy. The third metric is branch-swap consistency, because it tests causal dependence on the early branch token. The fourth metric is branch-direction/readout alignment, because it connects behavior to geometry. The fifth metric is cumulative singular-subspace contribution, because it tests whether the future suffix logits are controlled by a narrow dominant subspace or distributed across many dimensions.

For a suffix token $y$, the cumulative top-$r$ contribution can be defined as **$C_r(y,h)=\frac{\lVert P_r w_y\rVert\,\lVert P_r h\rVert}{\lVert w_y\rVert\,\lVert h\rVert}$**. Here, $P_r$ is the projection onto the top $r$ singular directions of the relevant output-facing matrix, $w_y$ is the output vector for token $y$, and $h$ is the hidden state. This metric is a diagnostic definition rather than a theorem. It measures whether readout-relevant token information is concentrated in the dominant singular subspace.

A more direct logit-decomposition metric is **$D_r(y,h)=w_y^\top P_r h$**. Here, $D_r(y,h)$ is the portion of the token logit explained by the top-$r$ projected hidden state, $w_y$ is the token output vector, $P_r$ is the projection onto the selected singular subspace, and $h$ is the hidden state. This metric follows directly from the linear readout definition and is easier to interpret as a contribution to the actual logit.

## 15. Final Interpretation

The best current interpretation is that multi-token training changes the role of the prefix hidden state. Under next-token training, the prefix hidden state only has to be good enough for the immediate next-token decision. Under multi-token training, the same hidden state must become predictive of a structured future. This changes the geometry of the representation: branch information must survive across irrelevant tokens and become aligned with output directions for future suffix tokens.

This is why the phrase "long-horizon feature" is more precise than "rare feature" or "compositional feature." A rare feature is about frequency. A compositional feature is about combining parts. A long-horizon feature is about delayed relevance under local ambiguity. The branch dataset directly tests delayed relevance: the branch token appears early, the shared prefix creates ambiguity, and the correct suffix can only be chosen if earlier information is preserved.

The current results therefore justify a focused next step rather than a broad expansion. First, verify the branch result across seeds. Then increase horizon length and filler entropy. Then replace lookup branches with compositional branch rules. At each stage, the decisive question should remain the same: does the hidden state at the shared-prefix decision point contain branch-relevant information that is aligned with future-token output directions?
