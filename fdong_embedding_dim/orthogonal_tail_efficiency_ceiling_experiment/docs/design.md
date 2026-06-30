# Can a spectral-tail parameterization learn tail patterns faster?

## Answer first

The unrestricted gated branch is the correct ceiling, and it changes the interpretation.

With hidden size 8, allowing the isolated tail branch to write into all eight directions converges in a median of 50 steps, versus 65 for bottom-2 spectral tail and 70 for top-2 common. With hidden size 16, unrestricted full output and bottom-2 spectral tail both converge in a median of 65 steps, versus 80 for top-2 common.

Therefore perfect routing and parameter isolation are sufficient to obtain the fastest observed branch learning. A spectral-tail restriction is not necessary. Its remaining value is parameter efficiency at dimension 16: 14 trainable parameters match the 65-step median of the 112-parameter unrestricted branch.

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

All stage-2 branches receive the same common input coordinates \(zV_C\) and the same perfect tail gate. The two restricted branches use \(A\in\mathbb{R}^{r\times r}\):

\[
\Delta h_C=(zV_C)A^\top U_C^\top,
\]

\[
\Delta h_T=(zV_C)A^\top U_T^\top.
\]

The common branch therefore reuses the largest parameter directions. The spectral-tail branch must write into the smallest parameter directions.

The unrestricted ceiling instead learns

\[
\Delta h_F=(zV_C)M,\qquad M\in\mathbb{R}^{r\times d},
\]

and full-dimensional contextual tied-embedding deltas. It can write into any direction. This increases trainable parameters from 14 to 56 at dimension 8 and to 112 at dimension 16, while hidden width remains unchanged.

## Why this is an oracle ceiling

The treatment receives three pieces of free information that a practical optimizer does not have:

1. the exact switch time after common-pattern pretraining;
2. the exact top and bottom singular bases of \(B_{vo}\);
3. a context gate that activates the new tied-embedding delta only on tail sequences, including a tail-specific delta for the shared token `moon`.

These choices deliberately remove routing, interference, and subspace-discovery costs. This makes the test favorable to the spectral-tail hypothesis. A positive result establishes existence inside this class; a negative result would have ruled out a broad but still limited class.

## Current interpretation

The dominant causal factor is a clean tail-specific parameter path protected from common-data updates. At dimension 8, forcing that path into only two directions costs 15 steps relative to unrestricted output. At dimension 16, bottom-2 is as fast in median as unrestricted output despite using one eighth as many trainable parameters, so a weak spectral subspace can be a compact workspace but is not a faster workspace.

The result does not overturn the frequency mechanism. The oracle branch is tested after removing frequency imbalance. It also remains slower than the earlier full-model uniform/reweight result of roughly 40 stable steps.

## Strongest defensible claim

Perfect routing plus parameter isolation makes tail learning faster than an isolated top-2 common branch. Unrestricted output is fastest at width 8 and tied with bottom-2 at width 16. The experiment does not show that spectral-tail restriction is required, does not beat full-model uniform/reweight, and does not provide a practical router.

## Next decisive experiment

Remove oracle components one at a time:

1. replace the tail-context gate with a learned router;
2. estimate the bottom subspace online rather than freezing a known basis;
3. reintroduce the 6:6:1:1:1 Zipf objective;
4. compare common and spectral-tail branches under equal token exposure and equal wall-clock/FLOP budgets;
5. test whether Muon or projected Adam discovers the same width-dependent advantage.
