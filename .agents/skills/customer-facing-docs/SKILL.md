---
name: customer-facing-docs
description: "Draft or revise concise, conclusion-first technical documents for external customers or partners. Use for customer reports, weekly or milestone updates, executive summaries, and proposal text. Do not use for internal research records, raw experiment logs, or code implementation."
---

# Customer-Facing Docs

## Default Stance

Write for an external customer, not for an internal lab notebook.

The document should show that we understand the problem, have a clear technical position, and know the next step. Do not expose the messy path taken to reach the conclusion unless the user explicitly asks for internal analysis.

## Core Principles

- Lead with the conclusion, then give evidence.
- State what the result means for the customer's objective.
- State decision-relevant uncertainty and claim boundaries concisely; never
  omit a limitation that would change the customer's decision.
- Omit irrelevant internal trial history, but preserve material negative
  results, failed assumptions, and caveats.
- Do not narrate every failed attempt unless the customer needs that history;
  state any decision-relevant failure and its measured consequence directly.
- Name an optimization direction only when the evidence supports it, and keep
  it separate from the direct result.
- Use exact numbers when available, but frame them as evidence for a decision.
- Avoid overexplaining implementation details that do not affect the customer's decision.
- Keep section titles plain and outcome-oriented.

## Summary And Conclusion Sections

A good executive summary or stage conclusion is short and clearly segmented.

Use this shape:

```text
1. What we established:
   one sentence, then 2-3 bullets.

2. What the experiment shows:
   one sentence, then 2 bullets for the key metrics or effects.

3. What we do next:
   one sentence, then 2-4 concrete next actions.
```

Good summary paragraphs have these properties:

- One center sentence per paragraph.
- Bullets carry parallel information, not hidden essays.
- Each bullet has a technical meaning and supports the customer-facing conclusion.
- The summary can be moved to the top of the document without sounding incomplete.
- It avoids repeating the whole document.

Avoid this pattern:

```text
Long paragraph explaining background
-> many caveats
-> historical attempts
-> conclusion hidden at the end
```

Prefer this pattern:

```text
This stage establishes <main conclusion>.
- Evidence A.
- Evidence B.
- Implication for the customer.

Next we will <concrete next step>.
- Action 1.
- Action 2.
```

## Framing Experimental Gaps

When a metric is not yet ideal, state the measured gap first, then its claim
boundary and the evidence-backed optimization direction:

- "This measured gap is consistent with overlapping expert functions; a
  specialization test is the next discriminator."
- "The routing distribution passes the predefined balance guard, but this does
  not establish expert specialization."
- "The current version preserves the measured target behavior under the tested
  conditions; capacity optimization remains untested."

Avoid vague or self-deprecating phrases like:

- "we failed"
- "we tried many variants"
- "this did not work"
- "we hide/exclude this task"
- "the result is not beautiful"
- "blind test"

Use `insufficient evidence for <specific claim>` when that is the actual
verdict; do not replace it with optimistic wording.

## Tables And Metrics

- Choose one metric story and keep it consistent.
- Disclose every metric-selection or filtering rule that could materially
  change the interpretation.
- If using a subset, define it neutrally and explain why that subset answers
  the stated decision.
- Put metric interpretation immediately after the table.
- Explain why the number matters, not just whether it is higher or lower.

Example:

```text
The core-task average remains in the same performance band, with a 1.1% absolute gap.
This gap indicates that routing specialization should be paired with expert-space constraints,
because nominal active capacity does not guarantee non-overlapping expert functions.
```

## Final Self-Check

Before delivering or editing a customer-facing document, check:

- Does the first paragraph make the main conclusion obvious?
- Does every section answer "so what" for the customer?
- Are caveats placed after the claim, not inside the headline claim?
- Are bullets parallel and short?
- Did we avoid internal-process language?
- Did we preserve every material negative result and claim boundary?
- Are metric selection and filtering rules disclosed when decision-relevant?
- Can this paragraph be sent directly to the customer without apology or extra explanation?
