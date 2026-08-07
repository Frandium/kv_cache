---
experiment_id: A15_01_09_E01_qwen3_mlp_parameter_commonality_attention_write_semantic_location
anchor_id: 15_01_09_mlp_parameter_commonality_attention_write_semantic_location
status: COMPLETED_LOCATION_WITHOUT_COMMONALITY
canonical_language: en
approved: 2026-08-07
---

# Protocol: Qwen3-8B MLP Parameter Commonality And Attention-Write Semantic Location

## 0. Approval Snapshot

- Researcher authorization: the 2026-08-07 instruction confirms the new Anchor and Protocol, treats the review as complete, and authorizes implementation, smoke, multiple single-node eight-5090 full runs, monitoring, analysis, figures, canonical result records, a self-contained report, and `daily_research_reports/0807/focus.md`.
- A--G audit blocks: `CONFIRMED`; cross-block consistency: `PASS`.
- Execution state: `COMPLETED_LOCATION_WITHOUT_COMMONALITY`; all approved smoke/P/S/R jobs succeeded with zero retries.
- No new semantic-data construction is authorized or needed. Model training, Router construction, graph editing, root sync, commit, and push remain outside scope.

## A. Question And Claim Boundary — CONFIRMED

### A1. Decision question

Does frozen Qwen3-8B have a cross-layer shared high-gain MLP input subspace, and does fine-relative-to-coarse semantic variance move toward lower-gain local parameter ranks with depth in the isolated attention write and remain there after coordinate matching, parameter gain, and actual MLP response?

### A2. Typed outcomes

- `joint_effective_pass`: parameter head commonality, the primary gain-weighted input-increment shift, and directional nonlinear-response confirmation all pass.
- `geometry_only`: raw post-`o_proj` write shifts later, but coordinate-matched, gain-weighted, input-increment, or nonlinear response does not.
- `commonality_only`: head commonality passes but semantic location does not.
- `location_without_commonality`: semantic location passes while the proposed shared-head mechanism does not.
- `joint_fail`: mandatory guards pass and neither registered clause holds.
- `insufficient_<guard>`: a mandatory provenance, numerical, replay, reliability, or sensitivity guard fails.

### A3. Cannot claim

No outcome proves a token-frequency mechanism, rare or high-level knowledge storage, where knowledge was learned, that attention is causally necessary, cross-layer identity of an individual singular vector, SAE feature identity, Router utility, or training benefit.

## B. Data Construction And Provenance — CONFIRMED

### B1. Frozen semantic cube

Reuse exactly the A15_01_05 dataset:

- 8 coarse parent domains;
- 8 fine children per parent;
- 4 label-free templates per child;
- 2 fact bundles per template;
- 512 total records, balanced in every cell;
- design templates 1--2 and confirmation templates 3--4;
- shared final `Classification:` colon readout;
- dataset identifier `cb440b98d81bac3f9813344f85e6efdbd994b7b988d8009ba64e207e64a11859` from the frozen experiment manifest.

The source file itself and its text serialization checksum are also recorded because the manifest identifier is a payload hash rather than the file byte hash. Cached post-`o_proj` states from run `a15-01-05-e04-post-o-proj-20260806T172600Z` are the primary write states.

### B2. Fair coarse/fine construction

For record state $z_{pcet}$ with parent $p$, child $c$, expression/template $e$, and fact bundle $t$:

$$
B^{coarse}=\operatorname{Cov}_{p}\!\left(\mathbb E_{c,e,t}z_{pcet}\right),
$$

$$
B^{fine}=\mathbb E_p\operatorname{Cov}_{c\mid p}\!\left(\mathbb E_{e,t}z_{pcet}\right).
$$

Both are population-weighted over balanced cells. They are not inflated by a different number of examples or labels. The common expression-noise covariance

$$
W^{expr}=\mathbb E_{p,c}\operatorname{Cov}_{e,t}(z_{pcet})
$$

is used only for reliability. The primary spectral-location comparison normalizes each role's between-class variance over the complete spectrum and therefore has no role-specific denominator.

### B3. No new data code

Implementation may copy or link the frozen data file and read the existing cache. It may add extraction code only for missing representation sites $x_\ell$, $\Delta n_\ell$, and MLP response. Any change to records, prompt text, tokenization, split, readout, or labels invalidates this Protocol.

## C. Diagnostic Task And Supervision — CONFIRMED

The model is frozen. There is no learned probe in the primary analysis. Coarse and conditional-fine class means define semantic between-class covariance. Reliability is accepted only if both role traces exceed parent-preserving hierarchical label-permutation q95 and split-template covariance overlap has a bootstrap lower bound above zero. Failed reliability yields `insufficient_semantic_reliability`, not a negative spectral claim.

Uncertainty uses 2,000 paired hierarchical bootstraps resampling parent, child within parent, and the eight expressions within child. Any sampling approximation must use a fixed seed and be recorded.

## D. Model, Parameters, And Representation Sites — CONFIRMED

### D1. Model

- `/data/share/Qwen3-8B`, frozen;
- 36 decoder blocks, hidden size 4096, 32 attention heads, MLP width 12,288;
- bfloat16 forward, FP32/FP64 analysis as specified below;
- all blocks analyzed; block 36 displayed separately and excluded from the registered early/late contrast.

### D2. Native and normalization-folded parameter operators

The native MLP input Gram operator is

$$
K_\ell=W_{gate,\ell}^{\top}W_{gate,\ell}+W_{up,\ell}^{\top}W_{up,\ell}
=V_\ell\Gamma_\ell V_\ell^{\top}.
$$

`down_proj` is excluded because it maps MLP hidden units back to the residual stream and does not define input-side directions. The normalization-folded control is

$$
K_\ell^{eff}=D_\ell K_\ell D_\ell,
$$

where $D_\ell$ contains the post-attention RMSNorm weights. It controls the coordinate mismatch between a residual-space write and the native post-RMSNorm MLP input coordinates; the sample-dependent radial RMS factor is handled exactly by $\Delta n$.

### D3. Representation sites

1. **Requested isolated write:** $a_\ell=$ direct `self_attn.o_proj` output before residual addition.
2. **Actual attention-induced MLP input increment:**

$$
\Delta n_\ell=RMSNorm_\ell(x_\ell+a_\ell)-RMSNorm_\ell(x_\ell).
$$

3. **Full nonlinear response:**

$$
\Delta m_\ell=MLP_\ell(n_{old,\ell}+\Delta n_\ell)-MLP_\ell(n_{old,\ell}).
$$

The pre-`o_proj` concatenated multi-head representation is not rerun because it was frozen in the preceding audit and is not in the MLP input coordinate. It may appear only as linked context, not as evidence for this Protocol.

## E. Objective And Optimization — CONFIRMED

No model optimization occurs. Parameter eigendecomposition uses FP32 Gram accumulation and symmetric eigensolvers. Semantic covariance and bootstrap aggregation use FP64 on CPU unless a numerically audited FP32 GPU equivalent is recorded. Fixed seeds: eigensolver deterministic path where available; bootstrap 1701; Haar controls 1702; replay 0.

## F. Comparison, Metrics, And Decision Evidence — CONFIRMED

### F1. Cross-layer parameter commonality

For projector $P_{\ell,k}$ of rank $r$:

$$
O_{\ell m k}=\frac{\operatorname{tr}(P_{\ell,k}P_{m,k})}{r}.
$$

The null-normalized value is $O_{\ell m k}/(r/4096)$. Primary width is $r=256$; sensitivity widths are 128 and 512. Report all-pair and adjacent-layer summaries, the 36×36 layer-pair heatmaps, same-band 16×16 summaries, and wrong-band overlap. H1 passes only if F1 exceeds matched middle/tail summaries and Haar expectation with paired layer-pair/bootstrap lower bounds above zero, and the ordering survives native/folded operators and registered widths.

### F2. Raw and gain-weighted semantic band variance

For each layer, split local sorted parameter rank into 16 bands of 256 directions: F1=head, F2--F8=middle, F9--F16=tail. For representation $z$ and semantic role $g$:

$$
E^{raw}_{\ell g k}=\operatorname{tr}(P_{\ell,k}B_{\ell,g})/256,
$$

$$
E^{gain}_{\ell g k}=\operatorname{tr}(V_{\ell,k}\Gamma_{\ell,k}V_{\ell,k}^{\top}B_{\ell,g})/256.
$$

Report absolute per-direction values, within-role shares $p_{\ell g k}$, and fine/coarse relative location

$$
L_{\ell k}=\log\frac{p_{\ell,fine,k}+\epsilon}{p_{\ell,coarse,k}+\epsilon}.
$$

No $D_{fine}/D_{coarse}$ with unequal within-class definitions is used.

### F3. Local-rank centroid and primary metric

With band center $c_k=(k-0.5)/16$,

$$
C_{\ell,g}=\sum_k p_{\ell,g,k}c_k,\qquad \Delta C_\ell=C_{\ell,fine}-C_{\ell,coarse}.
$$

The primary metric is gain-weighted $\Delta n$:

$$
T_{eff}=\operatorname{median}_{25:35}\Delta C_\ell-
\operatorname{median}_{1:12}\Delta C_\ell.
$$

H2/H3 require $T_{eff}>0$ with hierarchical-bootstrap 95% lower bound above zero, positive design and confirmation contrasts, at least 7/8 positive leave-one-parent-out contrasts, and no sign reversal under rank-128/512 regrouping. Raw $a_\ell$ centroid is a supporting geometry metric, not the primary metric.

### F4. Nonlinear broad-band response

Using native $K_\ell$ broad bands H=F1, M=F2--F8, T=F9--F16:

$$
\Delta m_{\ell,k}=MLP_\ell(n_{old,\ell}+P_{\ell,k}\Delta n_\ell)-MLP_\ell(n_{old,\ell}).
$$

Report each role's semantic variance share across H/M/T, the fine/coarse relative response, and

$$
N_\ell=\frac{\|\Delta m_\ell-(\Delta m_{\ell,H}+\Delta m_{\ell,M}+\Delta m_{\ell,T})\|_F}
{\|\Delta m_\ell\|_F+\epsilon}.
$$

Because MLP is nonlinear, H/M/T responses are interventions and need not add. Directional confirmation requires the late-minus-early fine-relative non-head response to be positive with a bootstrap lower bound above zero; large $N_\ell$ restricts interpretation to band interventions rather than additive decomposition.

### F5. Required controls

1. Native vs RMSNorm-folded parameter operator.
2. Direct cached $a_\ell$ vs rerun $a_\ell$ exact replay.
3. Actual $\Delta n$ vs raw $a_\ell$.
4. Raw vs gain-weighted location.
5. Same-band vs wrong-band cross-layer overlap.
6. 128/256/512 rank width.
7. Design vs confirmation templates and leave-one-parent-out.
8. Block 36 boundary view.

### F6. Figure contracts

1. **Parameter commonality heatmaps.** Native and folded layer×layer F1/M/T overlap normalized by Haar expectation. Allowed conclusion: subspace commonality, not individual direction identity or semantics.
2. **Layer×16 semantic-location heatmaps.** $L_{\ell k}$ for raw write, gain-weighted write, raw/gain $\Delta n$. Diverging color scale centered at zero and shared within metric family. Allowed conclusion: fine-relative-to-coarse local-rank allocation.
3. **Layerwise moving curves.** One line per representation/weighting for $\Delta C_\ell$, with early/late windows and block 36 boundary. Optional selected-layer rank curves use one line per layer; x is local sorted rank, never a shared direction.
4. **Actual MLP response.** H/M/T fine/coarse response shares and non-additivity by layer.
5. **Decisive composite.** Minimum panels needed to distinguish joint effective, geometry-only, commonality-only, and joint-fail explanations. It is embedded in `focus.md` and visually audited at full resolution.

## G. Execution And Reproducibility Boundary — CONFIRMED

### G1. Parallel job split

Use up to three independent single-node jobs so no multi-node dependency is introduced:

- `P`: parameter eigenspaces, native/folded cross-layer overlap, sensitivities;
- `S`: cached post-`o_proj` semantic raw/gain parameter-location analysis;
- `R`: identical-text replay for $x$, $a$, $\Delta n$, full and H/M/T MLP response.

Jobs may recompute parameter eigenspaces locally to avoid blocking dependencies. They must produce a common artifact manifest and compare eigenspectrum hashes/tolerances before consolidation.

### G2. Frozen resource profile

- execution surface: SCO ACP;
- profile: `5090-8-spot`;
- workspace: `share-space`;
- AEC2: `computing-cluster-5090-01g`;
- worker spec: `n12lp.nn.i10a.8`;
- one worker node per job;
- spot quota, normal priority;
- 8 visible GPUs and 8 local processes;
- image: `registry.cn-sh-01.sensecore.cn/lepton-trainingjob/ngc-pytorch:25.06-cu12.9-py3.12-ubuntu24.04`;
- storage mount: `01995892-d478-76d8-aec7-13fd8284477e:/data`.

The maintained ACP compatibility wrapper is permitted because the installed client cannot parse the current workspace metadata. Every job records the exact command, job ID, runtime preflight, GPU inventory, logs, exit status, retry/fault-tolerance state, and artifact checksums.

### G3. Run order and stop rules

1. Local contract validation and unit tests.
2. One-job smoke on reduced layers/records; verify shapes, cache replay, eigensolver, projection energy, and MLP intervention identity.
3. Submit P/S/R full jobs only after smoke passes; researcher has pre-approved this transition.
4. Monitor to terminal state; inspect logs and output manifests.
5. Consolidate and run all registered controls before plotting.
6. Open every central figure and audit labels, scales, missing cells, block 36, and claim boundary.
7. Write `summary.md`, `detailed.md`, self-contained report, then `focus.md`; use report evidence to polish focus once more.

No failed guard may be bypassed by filtering layers, parents, bands, or templates. A code repair that preserves the frozen estimator is allowed and recorded; a scientific definition change reopens Protocol approval.

## 1. Cross-Block Consistency Audit

- Question ↔ metric: `PASS`; $T_{eff}$ directly measures the registered depth change in fine-relative-to-coarse local parameter rank on the actual MLP input increment.
- Mechanism ↔ guard: `PASS`; cross-layer projector overlap, not rank labels, adjudicates parameter commonality.
- Representation ↔ parameter coordinate: `PASS`; raw requested diagnostic, norm-folded control, and exact $\Delta n$ are separated.
- Data ↔ fairness: `PASS`; both roles use the same balanced 512 records and population weighting.
- Claim ↔ evidence: `PASS`; geometry, gain, and nonlinear response have typed outcomes and cannot be silently merged.
- Execution ↔ authorization: `PASS`; implementation, smoke, three single-node 8×5090 jobs, monitoring, documentation, and figures are explicitly authorized.

This file is the executed contract. The observed outcome is `location_without_commonality`: head-specific commonality failed because tail overlap exceeded head; raw write fine-specific location failed; gain-weighted actual $\Delta n$ passed; nonlinear MLP response failed its confidence-interval gate. Any retrospective alteration to data, primary metric, parameter operator, early/late windows, representation sites, or pass rule invalidates this adjudication.
