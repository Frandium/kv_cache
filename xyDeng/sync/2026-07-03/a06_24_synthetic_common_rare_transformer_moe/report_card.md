# Report Card: A06_24 Synthetic Common/Rare Routing Audit

## Decision Question

Can global common subtraction by itself create common/rare and rare-feature
expert separation in a trained synthetic Transformer-MoE surface?

## Result

Supported negative boundary: simple global common subtraction is not a reliable
feature separator. Route-position centers and oracle centers separate features
cleanly; all-position common-subtracted centers remain weaker and can keep a
negative rare-margin lower tail.

## Key Evidence

- Full 4-GPU run: `pt-hb9swzcm`, `32` seed/slot cells, `1280` training rows.
- Step 0 all-position common-subtracted: rare-feature NMI `0.690`, joint
  feature score `0.405`, rare margin p05 `-2.759`.
- Step 0 route-position residual: rare-feature NMI `1.000`, joint feature
  score `0.637`, rare margin p05 `11.657`.
- Final all-position common-subtracted: joint feature score `0.432`, rare
  margin p05 `-5.427`, target accuracy `1.0`.
- Final oracle row-projected: joint feature score `0.636`, rare margin p05
  `8.646`, target accuracy `1.0`.
- Position guard: step-0 slot-start NMI maximum mean `0.024`.

## Mechanism Reading

The route score contains a common term and a feature residual term. Removing
the common term can reduce shared bias, but it does not identify the
route-relevant hidden-state population. A method must either find that
population without labels or preserve a valid center by constraining the router
update space.

## Claim Boundary

The result is synthetic and no-position. It does not prove real-DCLM transfer,
semantic expert utility, or optimality of row projection.

## Next Action

Use A06_24_synthetic as the boundary-setting package and open the next method
anchor for row-projected preservation or label-free route-relevant state
selection.
