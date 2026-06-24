# 0624 Feature Specialization Meeting Sync

## Scope

This package syncs the June 24 meeting material from the active
`from-attention-to-search` work branch into the `kv_cache` specialization
mainline reading surface.

It is a curated handoff package. It includes meeting documents, theory-audit
prompts, and copied source summaries only. It does not include raw logs,
checkpoints, datasets, full result directories, or debug-only artifacts.

## Reading Order

1. `meeting/meeting_brief_talk_version.md`  
   Short talk-facing version: question, verdict, mechanism, evidence, boundary,
   and next ask.
2. `meeting/meeting_brief.md`  
   Longer audit-facing meeting brief with source index and rival explanations.
3. `meeting/theory_ai_master_prompt.md`  
   Prompt for asking a theory AI to audit the mechanism and next anchor.
4. `meeting/theory_ai_prompts.md`  
   Two mechanism-specific theory prompts.
5. `source_summaries/`  
   Copied source summaries supporting the meeting claims.

## Current Claim

Uniform feature frequency is not enough to produce stable feature-level expert
specialization from random top-1 gating.

The current supported story is:

```text
route-position feature geometry exists
-> feature-center initialization is reachable in controlled settings
-> random gating and all-position clustering use the wrong geometry object
-> real DCLM proxy routing is reachable at step 0
-> ordinary real DCLM training overwrites that proxy partition by step 5/10
```

## Claim Boundary

This package does not claim:

- real-language semantic experts;
- deployable common-subtraction gating;
- reliable all-position feature discovery;
- solved real-DCLM training preservation;
- real-checkpoint expert utility.

The A07 utility evidence is controlled synthetic evidence, not a real-DCLM
checkpoint result.

## Next Decision

The recommended next decision is whether to open a real-text early
preservation / anti-feedback anchor:

```text
Can the step-0 proxy feature partition survive the step 5/10 real-DCLM
training window without materially hurting LM loss?
```

## Included Source Summaries

- `A05_04_02_round2_dotproduct_common_logit`
- `A05_04_03_real_text_common_logit_audit`
- `A06_08_label_free_feature_discovery_initialization`
- `A06_09_training_basin_preservation`
- `A06_10_real_dclm_proxy_feature_operationalization`
- `A06_11_real_dclm_proxy_center_router_initialization`
- `A06_12_real_dclm_proxy_init_training_preservation`
- `A06_13_real_dclm_proxy_init_failure_decomposition`
- `A06_16_synthetic_to_realistic_proxy_bridge`
- `A06_17_all_position_route_relevant_feature_discovery`
- `A07_01_common_rare_conflict_metric_audit`
- `A07_02_common_controlled_rare_interference`
- `A07_03_route_function_binding`
