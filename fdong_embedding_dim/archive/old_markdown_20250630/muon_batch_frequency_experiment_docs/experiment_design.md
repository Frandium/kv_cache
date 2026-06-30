# Experiment design

## Falsifiable conjectures

### C1: global-batch existence

On the 90/10 distribution with an exact population update, Muon or exact
inverse-frequency reweighting removes most of the tail delay relative to the
uniform AdamW control.

Pass condition:

- all-feature stable convergence is reached;
- `tail_stable_step / common_stable_step <= 1.2`; and
- total stable step is at most 1.25 times the matched uniform control.

Fail condition: any of these conditions fails after optimizer-specific learning
rate selection.

### C2: mini-batch degradation

On the 90/10 distribution, reducing batch coverage increases tail delay more
than common delay for Muon and inverse-frequency reweighting.

Pass condition:

- tail stable step or final tail loss worsens monotonically from population to
  batch 64 to batch 16 for a majority of seeds; and
- the degradation is larger under 90/10 than under uniform sampling.

Fail condition: mini-batch Muon matches the exact-population result within 10
percent and shows no larger tail degradation under 90/10.

### C3: noise versus bias

Known-global loss reweighting has low raw-gradient bias but high variance,
whereas Muon's transformed batch update has an additional nonlinear bias.

Pass condition:

- Monte Carlo raw-gradient relative bias decreases toward zero with more
  samples;
- raw-gradient RMS error increases as batch size decreases;
- Muon transformed-update relative bias is materially larger than raw-gradient
  bias for at least one 90/10 mini-batch condition.

## Parameters

- `num_features=16`: large enough to create feature absence, small enough for
  repeated NS5 diagnostics on CPU.
- `num_common=4`: separates a reusable common set from twelve tail features.
- `common_mass=0.90`: matches the proposed strong frequency imbalance.
- batch regimes: exact population, 64, 16.
- seeds: 0, 1, 2 for the first controlled run.
- steps: 400; stable window: 10.
- Muon: momentum 0.95, Nesterov on, five NS steps, no weight decay.
- AdamW: betas 0.9/0.999, no weight decay.
- learning-rate candidates are selected separately for AdamW and Muon using
  exact-population uniform runs. Stable convergence step is the primary score
  and final macro loss breaks ties. The selected value is then held fixed
  across distributions and batch regimes.

Too few features would make every batch cover the tail. Too many features would
turn this into a capacity experiment. A batch larger than 64 weakens the absence
test; a batch much smaller than 16 makes failure nearly guaranteed and less
diagnostic.

## Stage-level evidence

1. Algorithm check: singular values of an anisotropic test matrix must become
   substantially flatter after NS5 while zero singular directions stay zero.
2. Objective check: the expected globally reweighted gradient must equal the
   uniform-feature gradient numerically.
3. Estimator check: save raw and transformed bias/variance at initialization.
4. Training check: save common and tail accuracy/loss every step.
5. Geometry check: save top-1 energy and effective rank of both the applied
   update and the learned parameter matrix.

## Failure interpretation

- If full-batch reweighting fails, the implementation or learning-rate scale is
  wrong because the balanced objective is exact in this separable toy.
- If full-batch Muon fails while reweighting passes, update-spectrum flattening
  is not equivalent to balancing feature mass.
- If mini-batch reweighting has high variance but no asymptotic bias, the issue
  is delayed/unstable coverage rather than a wrong expected objective.
- If mini-batch Muon has additional bias, the nonlinear polar transform is part
  of the failure mechanism.
- If all methods pass equally, fixed orthogonal features are still too simple
  and a learned shared hidden representation is needed next.
