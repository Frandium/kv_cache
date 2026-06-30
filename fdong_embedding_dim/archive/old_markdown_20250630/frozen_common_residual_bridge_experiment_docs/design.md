# Frozen-Common Residual-Bridge Experiment

## Objective

Test whether a model can keep using a learned common hidden direction as input while new optimization is restricted to outputs orthogonal to that direction.

The falsifiable claim is:

> After the common direction has stabilized, a trainable mapping of the form
> \(P_R A P_C\) can reduce the remaining next-token error without changing the
> frozen common channel.

This is an existence test on a controlled sequence task. It is not yet a claim about full language models.

## Data

There are four equal-probability cyclic patterns:

\[
A_0,A_1,K,A_2;\quad
B_0,B_1,K,B_2;\quad
C_0,C_1,K,C_2;\quad
D_0,D_1,K,D_2.
\]

Each cycle produces four two-token-context next-token examples:

\[
G_0,G_1\to K,\quad
G_1,K\to G_2,\quad
K,G_2\to G_0,\quad
G_2,G_0\to G_1.
\]

The 16 examples have equal probability. Token \(K\) is therefore 25% of all targets, while each group-specific token is 6.25%.

## Base model

The model has:

- one tied input/output embedding \(E\in\mathbb{R}^{13\times d}\);
- one single-head attention-only layer;
- \(W_Q,W_K,W_V,W_O\in\mathbb{R}^{d\times d}\);
- a residual connection from the second context token;
- hidden dimension \(d\in\{2,3\}\).

For context \((x_1,x_2)\):

\[
q=W_QE[x_2],\quad k_i=W_KE[x_i],\quad v_i=W_VE[x_i],
\]

\[
\alpha=\operatorname{softmax}([q^\top k_1,q^\top k_2]/\sqrt d),
\]

\[
h_{\rm base}=W_O(\alpha_1v_1+\alpha_2v_2)+E[x_2],
\]

\[
\operatorname{logits}=E h_{\rm base}.
\]

## Common-direction definition

At a checkpoint, form the effective column-form attention Value map:

\[
B_{VO}=W_OW_V=U\Sigma V^\top.
\]

The common input and output directions are:

\[
v_C=V[:,1],\qquad u_C=U[:,1].
\]

The corresponding projectors are:

\[
P_C^{\rm in}=v_Cv_C^\top,\qquad
P_R^{\rm out}=I-u_Cu_C^\top.
\]

This definition is deliberately narrow. Diagnostics also report alignment between \(u_C\) and the current \(K\) embedding. If that alignment is low, the top effective-Value output direction cannot be interpreted as the K/common direction.

## Common-direction convergence rule

Every `check_interval` steps, compare both current top singular directions and the singular value with the preceding check. The recorded angle is the maximum of the input-side and output-side principal-angle changes:

\[
\theta_t=\max\{\cos^{-1}(|v_t^\top v_{t-\Delta}|),
\cos^{-1}(|u_t^\top u_{t-\Delta}|)\},
\]

\[
r_t=\frac{|\sigma_{1,t}-\sigma_{1,t-\Delta}|}{\max(\sigma_{1,t-\Delta},10^{-12})}.
\]

The common direction is declared stable only when:

- `step >= min_pretrain_steps`;
- `to_K_accuracy == 1`;
- either \(\theta_t\leq\)`angle_tol_deg` or
  \(r_t\leq\)`sigma_rel_tol` under the default `angle_or_sigma` rule;
- all conditions hold for `stable_checks` consecutive checks.

The default uses `angle_or_sigma` because cross-entropy can keep increasing the
singular-value magnitude after its direction has stabilized. The stricter
`angle_and_sigma` rule remains available as a command-line control.

If the rule never fires, the run uses `max_pretrain_steps` and records `forced_max_steps` rather than silently claiming convergence.

## Stage-two mappings

At the switch, clone the identical base checkpoint for all variants. The frozen directions and projectors are buffers, not trainable parameters.

The adapter runs in parallel with \(B_{VO}\). It reads the attention-weighted input embeddings

\[
z=\alpha_1E[x_1]+\alpha_2E[x_2]
\]

and adds:

\[
h=h_{\rm base}+M(z).
\]

Variants:

1. `baseline_continue`: no adapter; all original parameters continue training.
2. `frozen_no_bridge`: the base is frozen and no trainable mapping is added.
3. `unconstrained_bridge`: \(M(z)=Az\).
4. `common_to_residual`: \(M(z)=P_R^{\rm out}AP_C^{\rm in}z\).
5. `residual_to_residual`: \(M(z)=P_R^{\rm out}AP_R^{\rm in}z\). This deliberately blocks common input and is the negative geometric control.

For `common_to_residual`, automatic differentiation gives:

\[
\nabla_A L=P_R^{\rm out}\delta z^\top P_C^{\rm in}.
\]

The common input remains on the right side of the outer product, while the output update is restricted to the residual space.

## Claim boundary

A successful `common_to_residual` run shows that this constrained mapping exists for the toy and chosen checkpoint. It does not show that the top embedding singular direction is the unique common subspace, that the method scales to deep Transformers, or that it is better than continued end-to-end training.
