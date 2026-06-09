# Round-2 Summary

## Conclusion

Common-logit dominance is already visible at step 0, grows sharply before step 10, predicts the final dominant expert, and common-logit cancellation substantially improves slot-level specialization while preserving accuracy. This supports common-logit dominance as a causal contributor to random-init collapse in this toy dot-product top-1 setting.

## Key Evidence

- Dot-product reconstruction: mean error 1.073e-07, max error 5.849e-07; common-dominant cells 2238/2560.
- Timing: step-0 common margin 0.2993 vs slot margin 0.0523; step-10 common margin 1.7087 vs slot margin 0.2993.
- Prediction: common argmax predicts the final dominant expert at rate 0.844 at step 0 and 0.969 at step 10.
- Slot-init basin: best alpha 0.2 gives mean final NMI 0.930; first alpha with mean NMI >= 0.90 is 0.2.
- R2P3_cancel_0_10: final slot NMI 0.896, max_load 0.434, accuracy 1.000
- R2P3_cancel_0_final: final slot NMI 0.963, max_load 0.309, accuracy 1.000
- baseline_random_init_dot_product: final slot NMI 0.080, max_load 0.969, accuracy 1.000

## Decision Answers

1. Common-logit timing: step-0 common margin is 0.2993; step-10 common margin is 1.7087. Interpret growth only from the timing table, not from final collapse alone.
2. Early prediction: common argmax predicts final expert at 0.844 at step 0 and 0.969 at step 10.
3. Slot-init basin: threshold alpha = 0.2; see alpha sweep.
4. Common cancellation causal test: supports the causal-common claim under the strict NMI-plus-accuracy criterion.
5. Likely common source: fixed `B_CONST` / B-token identity is a likely major contributor; routing at the B position itself also contributes, because the route-at-slot diagnostic reduces the common margin; filler/template variation is not the main supported source. This is still a toy source audit, not a full source identification.
6. Next decision: Design a minimal anti-common or anti-lock-in router that preserves sparse top-1 while making slot-specialization reachable from random initialization.

## Claim Boundary

These results apply only to this toy uniform no-position synthetic task, dot-product router, sparse top-1 routing, and B-position loss/metrics. They do not show transfer to real language models or causal slot-specialized expert computation.
