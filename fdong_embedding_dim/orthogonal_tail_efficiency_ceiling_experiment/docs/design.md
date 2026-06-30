# Can a spectral-tail parameterization learn tail patterns faster?

## Answer first

Yes, such a setting exists in this controlled oracle class, but the advantage is conditional rather than universal.

With hidden size 16, a rank-2 branch forced into the bottom two left-singular directions of the pretrained value map reaches stable tail convergence in a median of 65 steps, versus 80 steps for an exactly parameter-matched branch writing into the top two directions. The paired held-out mean improvement is 12 steps over 45 seeds. With hidden size 8, both medians are 65 steps and there is no detectable benefit.

This result rejects the strongest negative claim—“using spectral-tail directions can never be more efficient.” It does **not** show that ordinary SGD can discover the branch, that every residual direction helps, or that spectral flattening by itself solves Zipf learning.

## Question being isolated

The previous frequency experiment established an easier claim: equalizing optimizer-visible frequency can accelerate tail learning without changing the learned spectral organization. That demonstrated one efficient route through the common space, but did not exclude another route that places new information into previously weak directions.

The present experiment asks a narrower existence question:

> After a common pattern has produced a highly singular parameter map, can a new tail pattern be learned faster when an oracle forces an otherwise matched update into the literal bottom of that parameter spectrum?

This is intentionally not a Zipf experiment. Stage 2 uses an exact, uniform full-batch tail objective so that frequency does not confound the direction comparison.

## Controlled intervention

After stage 1, decompose the effective value map

\[
B_{vo}=W_oW_v=U\Sigma V^\top.
\]

For rank \(r=2\), define

\[
U_C=U_{[:,1:r]},\qquad V_C=V_{[:,1:r]},\qquad U_T=U_{[:,-r:]}.
\]

Both stage-2 branches receive the same common input coordinates \(zV_C\), use the same trainable matrix \(A\in\mathbb{R}^{r\times r}\), and have the same contextual tied-embedding coefficients \(Z\). They differ only in the output basis:

\[
\Delta h_C=(zV_C)A^\top U_C^\top,
\]

\[
\Delta h_T=(zV_C)A^\top U_T^\top.
\]

The common branch therefore reuses the largest parameter directions. The spectral-tail branch must write into the smallest parameter directions.

## Why this is an oracle ceiling

The treatment receives three pieces of free information that a practical optimizer does not have:

1. the exact switch time after common-pattern pretraining;
2. the exact top and bottom singular bases of \(B_{vo}\);
3. a context gate that activates the new tied-embedding delta only on tail sequences, including a tail-specific delta for the shared token `moon`.

These choices deliberately remove routing, interference, and subspace-discovery costs. This makes the test favorable to the spectral-tail hypothesis. A positive result establishes existence inside this class; a negative result would have ruled out a broad but still limited class.

## Current interpretation

The hidden-size dependence matters. At dimension 8, the bottom-r parameterization is not faster. At dimension 16, it is faster under a separately tuned larger learning rate. This is consistent with—though it does not yet prove—the mechanism that extra weak directions provide an isolated, better-conditioned workspace once width leaves enough unused capacity.

The result does not overturn the frequency mechanism. The oracle branch is tested after removing frequency imbalance. A practical method must still supply enough optimizer-visible tail gradient and discover or maintain the appropriate subspace.

## Strongest defensible claim

There exists a width-16, rank-2, attention-only tied-embedding setting in which forcing tail learning into the pretrained parameter spectrum's bottom directions is more sample-step efficient than a parameter-matched common-direction branch. The effect is absent at width 8, and the experiment does not establish a general theorem or a practical optimizer.

## Next decisive experiment

Remove oracle components one at a time:

1. replace the tail-context gate with a learned router;
2. estimate the bottom subspace online rather than freezing a known basis;
3. reintroduce the 6:6:1:1:1 Zipf objective;
4. compare common and spectral-tail branches under equal token exposure and equal wall-clock/FLOP budgets;
5. test whether Muon or projected Adam discovers the same width-dependent advantage.

