# Meeting Brief: 05_02 Feature Assumption Relaxation

## One-Sentence Conclusion

P1/P3 weaken the claim that ordinary top-1 routing naturally produces meaningful feature-level specialization under relaxed assumptions: target prediction succeeds, but routing either collapses or forms only weak, seed-unstable structure.

## Main Question

The question is not whether the synthetic target can be learned. It is whether the route assignment remains non-collapsed and aligned with the intended feature relation after relaxing baseline assumptions.

## P1 Result

P1 tests compositional token routing:

- C1: target depends on $S1$.
- C2: target depends on $S2$.
- C3: target depends on $(S1,S2)$.

Primary metric:
归一化互信息（normalized mutual information, NMI）between route and each candidate axis.

Result:

- target accuracy is 1.0 for all conditions;
- C1/C2 mostly collapse, so they cannot support target-rule-dependent routing;
- C3-B0 is the only relatively interpretable case: active experts = 3.0, max load = 0.686, NMI(route,$(S1,S2)$)=0.528.

Interpretation:
Compositional target structure can leave a route-axis clue in C3, but ordinary top-1 does not robustly follow target-relevant factors across C1/C2/C3.

## P3 Result

P3 tests same-expert feature relation:

- P3a: input family structure only, target is $Y_{gk}$.
- P3b: input family plus shared target utility, target is $Y_g$.

Primary metric:
family purity over load-matched random baseline, $\Delta_{\mathrm{family}}$.

Result:

- target accuracy is 1.0 for all conditions;
- P3a-B0 delta = 0.055;
- P3a-I1 delta = 0.047;
- P3b-B0 delta = 0.087;
- P3b-I1 delta = 0.090;
- P3b gains are seed-unstable and partly tied to stronger concentration.

Interpretation:
Input family alone does not reliably induce same-expert family grouping. Shared family-level target utility also does not reliably induce grouping under ordinary top-1.

## Current Claim

Can claim:
ordinary top-1 learned the targets but did not robustly produce meaningful, non-collapsed routing under P1/P3 assumption relaxation.

Cannot claim:
expert utility specialization, real-data generalization, Zipfian robustness, or that no router can learn the family grouping.

## Next Decision

Decide whether to test an explicit non-collapse / utility-binding intervention on P3b. If not, stop treating ordinary top-1 same-expert assignment as structured evidence and park P2.
