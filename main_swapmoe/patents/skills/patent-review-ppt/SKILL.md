---
name: patent-review-ppt
description: Create or revise Chinese patent-review PowerPoint decks for customer and committee presentations, especially when a reference patent deck defines the required page structures and rhetorical format.
---

# Patent Review PPT

Use this skill when a patent proposal must be understandable to a customer or review committee and a reference PPT defines how each page should be written.

## Treat The Reference As A Contract

- Inspect every reference slide before editing.
- For each output slide, record its source slide, communication job, fixed headings, item count, ordering, typography pattern, and visual composition.
- Preserve requested headings verbatim. Replace the placeholder explanation beneath each heading with patent-specific content.
- Match the reference's rhetorical structure, not only its colors and title position. If the reference uses four bold protection-point names followed by normal-weight explanations, use exactly that structure.
- When the user identifies a slide as an example, do not substitute a personally preferred structure.

## Write For A First-Time Reviewer

- Assume the audience has not read the Markdown proposal and does not know project-defined commands, abbreviations, or internal terminology.
- Explain an implementation as observable data and computation flow: what is computed, where data is found, what happens on a hit, what happens on a miss, and what becomes ready for the next token.
- Use ordinary technical Chinese. A custom term may appear only after its meaning has been stated in the same slide; omit command-like labels when plain language is clearer.
- Each slide must answer one reviewer question. Remove name inventories that do not help the audience understand the causal chain.

## Use The Right Evidence Form

- Architecture slides must compare baseline and invention with the same visual grammar and system boundary. Show containing hardware, compute block, storage blocks, and data movement explicitly.
- Background slides should use recognizable, sourced product or architecture images when concrete hardware families are discussed.
- Result slides must separate: headline comparison, source parameters, model/formula, and calculated result. Do not replace derivation with decorative bars.
- State what is measured, what is sourced, what is assumed, and what is simulated. Put formulas and numerical substitutions on the slide when they are needed to justify the conclusion.

## Required Patent Page Patterns

### Protection Points

- Use four primary points unless the reference or user specifies another count.
- Each point contains a bold protection-point name and a normal-weight paragraph explaining the protected structure, interaction, timing, and technical effect.
- Prefer four complete boundaries over seven one-sentence feature fragments.

### Patent And Commercial Value

- Keep every reference subheading unchanged.
- Under each subheading, explain the value of this specific patent in complete sentences; do not rename or merge categories.

### Discoverability

- Follow the reference categories and provide detailed observable evidence for each: public materials and patent text, hardware or driver interfaces, runtime/profiling behavior, and strongest evidence combinations.
- Describe where the evidence appears, what behavior would be observed, and how it maps to a claimed feature.

### Avoidance And Upper-Level Protection

- First list three or four avoidance routes as one group.
- Then list the corresponding upper-level protection proposals as a second group in the same order.
- Do not interleave an avoidance route with its answer.

### Patent Search

- Use patent-database Boolean search expressions with quoted phrases, parentheses, `AND`, `OR`, proximity operators when supported, and bilingual variants.
- Organize searches by concept intersection, such as MoE routing prediction, expert-object addressing, tiered memory, and prefetch or preactivation.
- Do not present slash-separated keyword inventories as search formulas.

## Final Compliance Check

Before delivery, compare every user comment against the rendered slide, one by one. Fail the draft if any explicitly requested item count, heading, grouping order, diagram boundary, evidence derivation, image requirement, or terminology restriction is missing.

Render every slide and inspect it at full size. Confirm that a reviewer can understand the deck without reading another document.
