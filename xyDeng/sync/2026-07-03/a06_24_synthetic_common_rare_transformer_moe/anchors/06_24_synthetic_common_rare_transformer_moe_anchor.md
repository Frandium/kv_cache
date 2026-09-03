# A06_24 Synthetic Common/Rare Transformer-MoE Routing Audit

## 0. researcher judgment record

**Phenomenon:** Earlier A06 evidence shows that simple common subtraction can reduce expert concentration but does not by itself split features across experts. The previous A06_24_toy vector audit is useful as a mechanism toy, but it is not reliable enough for the mainline because it does not train a one-layer Transformer plus one-layer MoE.

**Mechanism guess:** Under an imbalanced common/rare feature distribution, ordinary top-1 routing will spend routing capacity on the high-frequency common feature or on non-route states. Common subtraction may improve load, but rare-feature separation requires a route-relevant feature center or a preservation control after a valid initialization.

**Key variables:** feature frequency, feature slot length, route-position hidden state, all-position hidden state, common vector, router rows, rare-feature route assignment, rare margin, sign-flip rate, slot-start nuisance agreement.

**Causal relation:** If the failure is mainly common concentration, then common-subtracted routing should improve rare-feature separation. If the failure is route-relevance / center-selection mismatch, common subtraction should improve load or common/rare balance but rare features should still merge unless route-relevant or oracle centers are used.

**Observable metric:** The primary metric is rare-feature routing consistency, measured as normalized mutual information between rare feature id and routed expert on a balanced held-out route-position evaluation set. The required guard is joint feature score, which also checks common-vs-rare separation.

**Rival Explanations:** apparent improvement may come from load balancing, position leakage, common-vs-rare binary separation without rare-rare separation, oracle label leakage, or a toy hidden-space construction that does not survive Transformer-MoE training.

**Decision:** Use a no-position one-layer Transformer plus one-layer MoE synthetic task to decide whether common subtraction is only a concentration repair or can actually create rare feature-level expert separation under feature-frequency imbalance.

## 1. Problem Definition

**Parent problem:** A06 studies route-relevant proxy discovery, initialization, and early preservation for feature-level expert specialization.

**Sharper subproblem:** Determine whether common subtraction can separate common and rare features in a real trained synthetic Transformer-MoE surface, or whether it only improves concentration while rare-feature separation still requires valid route-relevant centers and preservation controls.

**Terminology / Definitions:**

| Term | Plain meaning | Concrete object or computation | Unit / formula | Why it matters for the current decision | What it cannot prove |
| --- | --- | --- | --- | --- | --- |
| Common feature | High-frequency feature in the synthetic task | Feature id 0 sampled much more often during calibration and training | Probability mass | Tests whether routing capacity follows frequency | Natural-language common semantics |
| Rare feature | Low-frequency feature that should remain distinguishable | Feature ids 1--3 | Feature ids | Tests rare-rare separation, not only load | Real rare semantic feature |
| Route position | Token position where the router is audited and the target is predicted | Last token of the repeated feature slot | Sequence index | Keeps evaluation tied to the task-relevant state | A real language route-relevance detector |
| No-position model | Transformer without learned or sinusoidal position embeddings | Token embeddings plus causal attention only | Architecture setting | Guards against position shortcuts | Complete removal of causal context-length effects |
| Rare-feature NMI | Agreement between rare feature id and expert route | NMI(feature id, routed expert) restricted to rare examples | 0--1 | Primary metric for rare separation | Expert utility or semantics |
| Rare margin | Matched rare expert score minus strongest competitor | $z_{i,m(f_i)}-\max_{e\ne m(f_i)}z_{i,e}$ for rare examples | Logit difference | Tests whether rare routes are inside a stable basin | Training usefulness by itself |
| Joint feature score | Combined common/rare and rare-feature separation | `rare_feature_NMI * common_rare_NMI` | Dimensionless score | Prevents overclaim when rare features separate but common still mixes with rare | Expert utility |
| Slot-start NMI | Agreement between route assignment and slot start | NMI(slot_start, routed expert) | 0--1 | Position leakage guard | Absence of all positional effects |

**Decision question:** In a no-position one-layer Transformer plus one-layer MoE synthetic common/rare task, does common subtraction create rare-feature expert separation, or is it only a load/concentration control while rare separation requires route-relevant centers and preservation controls?

**Not in scope:** real DCLM claims, semantic experts, production router methods, theoretical optimal learning rates, or proof that all common-removal methods fail.

## 2. Physical Priors

**P1: Frequency imbalance can make load repair look like specialization.**  
Meaning: A high-frequency common feature can dominate router load, so a method may reduce max load without separating rare features.  
Could be wrong if: common subtraction raises rare-feature NMI and rare margins while slot-start NMI stays low.

**P2: Route relevance is not supplied by global common subtraction.**  
Meaning: subtracting one global common vector does not tell the router which hidden states are task-relevant route states.  
Could be wrong if: all-position common-subtracted centers match route-position or oracle centers on rare-feature metrics.

**P3: Valid initialization and preservation are separate.**  
Meaning: route-relevant centers may separate rare features at step 0, but early training can still erase that separation unless residual input or row projection protects the margin.  
Could be wrong if: ordinary raw training preserves rare separation as well as residual controls.

## 3. Falsifiable Hypotheses

**H1: Common subtraction is a load/concentration repair, not a rare-feature separator.**  
Supported if: common-subtracted random or all-position conditions reduce max load or common/rare concentration but do not improve rare-feature NMI/margin over raw baselines.  
Weakened if: common subtraction alone reliably raises rare-feature NMI and rare margins across slot lengths.

**H2: Route-position centers are needed for rare-feature separation under imbalance.**  
Supported if: route-position k-means or oracle feature centroids outperform all-position common-subtracted centers on rare-feature NMI and margin.  
Weakened if: all-position common-subtracted centers match route-position centers without position leakage.

**H3: Preservation controls matter after valid initialization.**  
Supported if: residual router input or router-row projection improves final rare margin and reduces sign flips relative to raw ordinary training.  
Weakened if: ordinary training preserves rare separation equally well.

## 4. Mathematical Model

**Objects:** hidden state $h_i$, common vector $c$, residual hidden state $r_i=h_i-c$, router row $w_e$, feature id $f_i$, route score $z_{i,e}=w_e^\top h_i$, matched expert $m(f)$.

**Core decomposition:**

$$
h_i = c + r_i,\qquad z_{i,e}=w_e^\top c + w_e^\top r_i.
$$

**Mechanism relation:** If the common term controls only concentration, removing $c$ should reduce common bias or load imbalance but should not create a one-to-one rare-feature mapping unless $r_i$ is clustered in a route-relevant population.

**Observable metrics:** rare-feature NMI, common/rare binary NMI, joint feature score, rare margin, sign-flip rate, max load, effective expert count, slot-start NMI, task loss and target accuracy.

**Falsifier:** common-subtracted all-position routing achieves high rare-feature NMI and positive rare margins across slot lengths while slot-start NMI remains low and task loss is not worse.

## 5. Computational Realization

**Input objects:** synthetic sequences with one high-frequency common feature, three low-frequency rare features, repeated feature slots, random neutral background tokens, and feature-specific target tokens.

**Computed variables:** route-position hidden states, all-position hidden states, common vectors, k-means centers, oracle feature centroids, router assignments, rare margins, position nuisance metrics, training trajectories.

**Algorithm stages:**

1. Build a one-layer causal Transformer plus one-layer top-1 weighted MoE without position embeddings.
2. Extract calibration hidden states before training.
3. Initialize router rows from random rows, all-position centers, route-position centers, or oracle feature centers.
4. Evaluate step-0 rare separation on a balanced held-out route-position set.
5. Train with an imbalanced common/rare objective and evaluate preservation.
6. Compare raw routing, common-subtracted routing, residual router input, and router-row projection.

**Stage-local evidence:** step-0 rare-feature NMI, final rare-feature NMI, rare margin, sign flips, load, target accuracy, slot-start NMI.

**Expected artifacts:** `protocol.md`, `summary.md`, `detailed.md`, CSV tables, PNG figures, logs, and ACP submission record.

## 6. Minimal Falsification Tests

| Test | Question | Intervention / comparison | Primary metric | Pass / fail / insufficient | Why it decides | Failure means |
| --- | --- | --- | --- | --- | --- | --- |
| Step-0 common subtraction audit | Does common subtraction itself separate rare features? | random/raw and all-position/raw versus common-subtracted variants | rare-feature NMI | Pass for H1 if load improves but rare NMI does not; fail H1 if rare NMI rises across slot lengths | Separates load repair from feature separation | Falsifies common-subtraction-as-separator operationalization if it fails |
| Route-relevant center audit | Does knowing the route-relevant pool matter under imbalance? | all-position k-means versus route-position k-means versus oracle centroids | rare-feature NMI and rare margin | Pass H2 if route/oracle centers win; fail H2 if all-position common centers match them | Tests sample-pool mismatch under common/rare skew | Falsifies route-pool necessity in this synthetic surface |
| Preservation audit | Does valid rare separation survive training better with common control? | route/oracle init with raw training versus residual input or row projection | final rare margin and sign-flip rate | Pass H3 if controls preserve margin better without loss regression; fail if raw matches controls | Separates initialization from early training preservation | Falsifies the tested preservation controls, not all preservation methods |

## 7. Current Evidence

**Observation:** A06_07 showed common-centering can reduce load while leaving feature NMI nearly unchanged; A06_17 addendum showed training-time common subtraction does not rescue an all-position feature-merge basin; A06_20 showed a routing-aware common estimator does not improve feature recovery over raw all-position clustering.

**A06_24_synthetic result:** The full 4-GPU run `pt-hb9swzcm` completed 32 seed/slot cells. At step 0, all-position common-subtracted centers had rare-feature NMI `0.690`, joint feature score `0.405`, and rare margin p05 `-2.759`, while route-position residual centers and oracle centers reached rare-feature NMI `1.000`, joint score `0.637`, and positive rare margin p05 about `11.6`. After training, all conditions reached target accuracy `1.0`, but all-position common-subtracted routing remained weaker (`joint=0.432`, rare margin p05 `-5.427`) than route-position raw (`joint=0.620`, rare margin p05 `5.227`) and oracle row-projected routing (`joint=0.636`, rare margin p05 `8.646`). Slot-start NMI stayed low, with step-0 maximum mean `0.024`.

**Interpretation:** Existing evidence already weakened simple common subtraction as a feature-separation method, and A06_24_synthetic supports the same boundary on a trained no-position Transformer-MoE surface. A refined preservation boundary appears: residual input preserves rare-rare separation and margin, but row projection better preserves the full common/rare partition.

**Boundary:** The previous A06_24_toy vector audit supports a possible residual-control mechanism but is not reliable enough as mainline evidence because it bypasses Transformer hidden-state formation and training.

**Evidence links:**

- `Projects/from-attention-to-search/main/experiments/A06/A06_07_label_free_common_residual_control_router/summary.md`
- `Projects/from-attention-to-search/main/experiments/A06/A06_17_all_position_route_relevant_feature_discovery/addendum_common_subtraction_rescue/summary.md`
- `Projects/from-attention-to-search/main/experiments/A06/A06_20_route_logit_common_estimator_random_init_feature_recovery/summary.md`
- `Projects/from-attention-to-search/main/experiments/A06/A06_24_toy_common_rare_residual_proxy_synthetic/summary.md`
- `Projects/from-attention-to-search/main/experiments/A06/A06_24_synthetic_common_rare_transformer_moe/summary.md`
- `Projects/from-attention-to-search/main/experiments/A06/A06_24_synthetic_common_rare_transformer_moe/detailed.md`

## 8. Claim Boundary And Next Decision

**Can claim:** In this no-position Transformer-MoE synthetic surface, simple global common subtraction is not a reliable feature separator. Rare-feature expert separation under imbalance is much cleaner with route-relevant centers or oracle centers. Target accuracy does not prove specialization. Row projection is a stronger preservation candidate than residual input when the desired claim includes common-vs-rare separation as well as rare-rare separation.

**Cannot claim:** real language transfer, semantic expert formation, deployable gating, optimal optimizer design, or impossibility of all label-free route-relevance methods.

**Next decision:** Move to a method anchor that tests a task-aware route-relevant state selector or a row-projected margin-preserving update on a harder bridge; do not spend the next iteration on simple global common subtraction as the main method.
