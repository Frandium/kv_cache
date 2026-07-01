# Report Card: Feature Route Margin And Common Interference

## Decision Question

Can the current feature-routing line be explained by a margin-preservation
mechanism, and what is the next test needed before claiming common-subtraction
as a routing method?

## Result

Supported in the controlled setting: clustering/center initialization can
produce balanced feature routing and preserve it through training when the
initial route margin is thick enough.

Not yet supported in real DCLM: common-component removal is not validated,
because A06_22 shows that common/residual dominance depends on the projector
used to define common structure.

## Key Evidence

- A06_08: route-position residual clustering reaches feature NMI `1.0` and
  load $L=0.0$ in the controlled uniform-feature setting.
- A06_17_02: preserved routing is not explained by active router-center
  tracking; movement alignment is negative while feature NMI stays `1.0`.
- A06_21: preserved center initialization keeps positive margins, zero sign
  flips, and matched preserve fraction `1.0`.
- A06_23: reducing the initial row gap to $q=0.02$ starts correct but fails;
  final matched preserve fraction is `0.375`, final margin is `-0.588`, and
  sign-flip rate is `0.159`.
- A06_22: real-DCLM proxy routing collapses, but common dominance is not
  projector-robust.

## Claim Boundary

The current result supports a controlled margin mechanism and motivates a
common/rare interference audit. It does not prove real semantic experts,
expert utility, or a working common-subtraction gate.

## Next Action

Run a bounded common-subtracted routing audit: define $P_C$, compare raw
routing on $h$ with residual routing on $(I-P_C)h$, and report rare-feature
margin, matched preserve fraction, sign-flip rate, language-model loss, and
projector robustness.
