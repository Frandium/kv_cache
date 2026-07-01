# Report Card: Feature Center Margin Mechanism

## Question

Can feature-to-expert specialization form, and why can it remain stable after
training begins?

## Verdict

Controlled synthetic evidence supports:

```text
label-free route-position feature-center initialization can form a clean
feature-to-expert partition;
the observed preservation is better explained by positive margin buffer than
by active router-center tracking.
```

## Why This Is Nontrivial

Positive margin at a single step is just the definition of top-1 routing.

The nontrivial part is that feature-center initialization creates a large
feature-specific margin, and the observed training trajectory does not consume
that margin. Random initialization can learn the task but does not reliably
create the same feature-specific partition.

## Key Evidence

| Result | Update |
|---|---|
| A06_16 | Corrected no-position bridge passes C0-C3: step-0 and final feature NMI are `1.000`. |
| A06_17 | Route-position states recover feature centers; all-position states often merge features. |
| A06_17_02 | Center init preserves NMI `1.000`, but router movement is negatively aligned with feature-center movement. |
| A06_17_03 | Successful center init stays inside the initial boundary buffer; dynamic switch rate is `0.000`. |
| A06_17_04 | Margin is a real boundary: stress lowers margin but does not cross it; forced crossing breaks the matched region. |
| A06_17_05 | Exact centers are not required; perturbations through `rho=0.70` preserve while pure random collapses. |
| A06_18 | Generic representation learning does not replace route-relevant state selection. |

## Boundary

This is controlled synthetic evidence. It does not claim real-DCLM semantic
experts, expert utility, or solved preservation under real next-token training.

## Next Todo

Run a spectral-band margin decomposition:

```text
m_f = m_f^C + m_f^{perp C}
```

where `C` is the high-gain common spectral band. The decision is whether
preservation is supported by feature residual margin or mostly by common-band
margin.

