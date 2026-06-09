# 05_04_02 Dot-Product Common-Logit Causality Anchor

## 0. Tiny Summary

父问题：为什么普通 sparse top-1 MoE router 不能稳定形成 feature-to-expert specialization？

当前子问题：dot-product router 的 collapse 是否由 common-logit dominance 的时序和因果作用驱动？

核心结论：common logit 在 step 0 已经强于 slot margin，step 10 前继续放大，并且 common-logit cancellation 显著提高 slot-level specialization。

最小测试：审计 common logit 的 step-0/early timing、slot-init basin、common-logit cancellation 和 common-source ablation。

当前边界：只研究 no-position uniform slot toy setting；不声称真实 LM、multi-B 或 expert computation causality。

下一步决策：设计 label-free anti-common / anti-lockin router，并保留 B-source audit 作为边界条件。

## 1. Problem Definition

Parent problem:
Why does sparse top-1 MoE gating fail to stably form feature-to-expert specialization?

Sharper subproblem:
Round 1 with a dot-product router showed that the minimal no-position moving-block baseline collapses even though `h_B` contains slot information. Round 2 resolves the next mechanism question: common-logit dominance is not only a diagnostic pattern; in this toy setting it is early, predictive, and intervention-relevant for collapse.

Decision question:
In the dot-product no-position uniform slot task, does common-logit dominance cause random-init top-1 routing to enter a collapse basin rather than a slot-specialized basin?

Not in scope:

- Do not claim expert computation is causally slot-specialized; P4 forced-routing utility is not the object here.
- Do not claim transfer to real language models, longer contexts, or multi-B exact-feature routing.
- Do not use soft routing, top-k routing, or probability-weighted expert mixtures.

## 2. Physical Prior

P1: Dot-product routing makes hidden-component logit decomposition exact.
Meaning: for `score_e(h)=w_e^T h`, the contribution of common, slot, position, and residual hidden components adds linearly to the router score.
Could be wrong if: reconstruction error is not negligible relative to the margin scale, or the measured decomposition is numerically unstable.

P2: Early top-1 routing can lock the model into a common-dominated basin.
Meaning: if common margin selects the same dominant expert before slot margin becomes decisive, sparse top-1 updates reinforce that early assignment.
Could be wrong if: common dominance appears only after routing has already locked, or early common ranking does not predict final dominant experts.

P3: Removing the common-logit term should help only if common dominance is causal.
Meaning: subtracting `w_e^T c_t` from every expert score should improve slot NMI/purity while preserving accuracy if common logit drives collapse.
Could be wrong if: cancellation improves load balance without slot NMI, harms accuracy, or collapse persists through another non-slot component.

## 3. Hypothesis

H1: Random-init dot-product top-1 routing collapses because common-logit advantage is present at initialization or grows before step 10 and predicts the final dominant expert.

Supported if:

- step-0 or early common margin predicts final dominant expert before global lock-in;
- common margin dominates before or by step 10;
- slot margin is not the early decisive routing component.

Weakened if:

- common dominance appears only after lock-in;
- early common ranking does not predict final experts;
- slot margin dominates early while routing still collapses.

H2: Slot initialization succeeds because it moves the router across a basin threshold where initial slot margin exceeds common margin.

Supported if:

- interpolation from random to slot-init shows a threshold alpha where final slot NMI and max-load switch from collapsed to slot-specialized;
- successful runs have positive initial `slot_margin_minus_common_margin`.

Weakened if:

- final success does not correlate with initial slot/common margin geometry;
- no threshold behavior appears across alpha.

H3: Common-logit cancellation is a causal intervention for collapse.

Supported if:

- cancellation increases slot NMI/purity and reduces collapse while keeping target accuracy high;
- improvements are not merely load balancing without slot alignment.

Weakened if:

- cancellation removes common score but routing still collapses with low slot NMI;
- accuracy drops enough that routing evidence is not comparable.

## 4. Minimal Mathematical Model

Objects:

- `h_{s,p,t}`: routed hidden state at `B_CONST` for slot `s`, block start `p`, checkpoint `t`
- `c_t`: common hidden component
- `r_{s,t}`: slot component
- `u_{p,t}`: block-position component
- `w_{e,t}`: dot-product router weight for expert `e`
- `e*`: selected top-1 expert

Core decomposition:

$$
m_{s,p,t}=c_t+r_{s,t}+u_{p,t}+residual_{s,p,t}
$$

Mechanism relation:

$$
\Delta score
=\Delta w^T c_t
+\Delta w^T r_{s,t}
+\Delta w^T u_{p,t}
+\Delta w^T residual_{s,p,t}
$$

where `\Delta w = w_{e*,t}-w_{e2,t}` for the top-1 expert and runner-up.

Observable metrics:

- reconstruction error for the additive margin;
- common / slot / position / residual margin contribution;
- `common_predicts_final_expert_at_step_t`;
- `slot_to_expert_NMI`, per-slot purity, `max_load`, target accuracy;
- `slot_margin_minus_common_margin`;
- route switch time and global lock step.

Falsifier:
If common margin is not early, not predictive, and cancellation does not improve slot NMI while preserving accuracy, then common-logit causality is weakened.

## 5. Minimal Tests

Experiment: R2-P1 Common-Logit Timing Audit
Question: Is common dominance already present at step 0, or does it grow before lock-in?
Intervention: none; track common/slot margins at steps `0,1,2,5,10,20,50,final`.
Primary metric: early common ranking predicts final dominant expert before global lock step.
Supports: common margin is early and predictive.
Weakens: common margin appears only after routing has locked.

Experiment: R2-P2 Slot-Init Basin Audit
Question: Does slot-init success reflect a basin threshold?
Intervention: interpolate router weights `W(alpha)` from random to slot-initialized directions.
Primary metric: final slot NMI and initial `slot_margin_minus_common_margin` across alpha.
Supports: threshold alpha separates collapse from slot specialization.
Weakens: success is not related to initial margin geometry.

Experiment: R2-P3 Common-Logit Cancellation
Question: Does removing common logit improve slot specialization?
Intervention: route with `score'_e(h)=w_e^T h-w_e^T c_t`, keeping sparse top-1 forward.
Primary metric: slot NMI with accuracy guard.
Supports: slot NMI/purity improves while accuracy remains high.
Weakens: only load balance improves, or collapse persists.

Supporting checks:
R2-P0 verifies that dot-product decomposition is numerically valid. R2-P4 audits the likely source of the common component; it informs the next intervention but is not the core causal test.

## 6. Falsifiable Outcomes

Supported:

- common margin is exact, early, predictive of final expert, and cancellation improves slot NMI/purity with high accuracy;
- slot-init interpolation shows a basin threshold tied to slot-vs-common margin.

Weakened:

- common dominance is post-lock-in or non-predictive;
- cancellation fails to improve slot NMI, or only balances load;
- slot-init success is not explained by slot/common margin threshold.

Insufficient evidence:

- cancellation changes routing but accuracy drops;
- source ablation changes common norm but also changes the task;
- only aggregate load improves without slot-aligned routing.

## 7. Decision Synthesis

Conclusion:
In the no-position uniform slot task with a dot-product sparse top-1 router, collapse is best explained as a common-logit-driven basin problem. The model has slot information in the routed hidden state, but random initialization lets a shared B-position/common component dominate the router before slot-aligned routing can stabilize.

Why this conclusion is supported:

- Dot-product decomposition is reliable enough to separate common, slot, position, and residual logit contributions.
- Common-logit advantage appears at initialization, grows before the early lock-in window closes, and predicts the final collapsed expert.
- Slot-derived initialization moves the model into a different basin where slot-level routing becomes reachable.
- Removing the common-logit contribution during routing sharply improves slot-level specialization while preserving task accuracy.
- The source audit points away from filler/template identity and toward fixed `B_CONST` plus the routed B-position representation as the strongest common-source candidate.

Rival explanations weakened:

- Missing slot information is weakened because `h_B` remains slot-decodable.
- Pure load-balancing failure is weakened because common cancellation improves slot alignment, not only expert load.
- Pure position embedding leakage is weakened because position embeddings are disabled and the B-position source still matters.

Still unresolved:

- Whether a label-free intervention can reproduce the common-cancellation benefit without supervised slot information.
- Whether the fixed-B common source is token-identity-specific, residual-stream-mean-specific, or caused by optimizer/top-1 feedback.
- Whether expert computation, not only routing assignment, becomes causally slot-specialized.

Evidence location:
Full result tables, seed-level metrics, plots, and implementation details are in:

- `round2_summary.md`
- `round2_detailed.md`

## 8. Claim Boundary And Next Decision

Can claim:

- random-init dot-product sparse top-1 routing collapses in this minimal no-position setting;
- common-logit dominance is early, predictive, and intervention-relevant;
- common-logit cancellation supports a causal role for common dominance in routing collapse;
- slot-init success is best interpreted as entering a different routing basin;
- fixed `B_CONST` / B-position routed representation is the leading common-source candidate.

Cannot claim:

- common source is fully identified;
- cancellation is a deployable solution;
- expert computation is causally slot-specialized;
- the result transfers to real language models, longer contexts, cosine routers, or broader MoE settings.

Next decision:
Design one label-free anti-common / anti-lockin router intervention that keeps sparse top-1 routing and tests whether random-init training can reach high slot NMI without supervised slot initialization.
