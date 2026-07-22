# A14 Probabilistic Sparse-Activation Tree Handoff

## Scope

This package syncs the current A14 self-contained theory of conditional
low-dimensional computation in a globally high-dimensional tree. It contains
the English and Chinese stories, the formal Chinese proof, and the smallest
curated evidence set needed to audit the static theory and real-language
boundary.

## Reading Order

1. [English story](story.md)
2. [Chinese story](story_cn.md)
3. [Formal Chinese proof](theory/probabilistic_sparse_activation_tree_theory_proof_cn.md)
4. [E07 reachable-space implementation audit](evidence/E07/summary.md)
5. [E06 real-language boundary audit](evidence/E06/summary.md)

## Included Material

- self-contained English and Chinese probability-tree stories;
- the formal proof for fixed-mode and high-probability reachable-dimension
  bounds;
- E02 and E04 summaries for historical geometry and controlled-training
  context;
- E06 summary and central figure for the direction-balanced real-language
  boundary;
- E07 protocol, summary, detailed record, and central figures for exact
  reachable-space accounting.

## Excluded Material

Daily focus notes, private meeting material, raw logs, checkpoints, datasets,
code, complete output directories, and unrelated research lines are excluded.

## Current Claim

For a fixed activation mode, if each active local source has dimension at most
$r$ and at most $s$ sources are active, the root reachable dimension is at
most $sr$. If that source-count event holds with probability at least
$1-\delta$, the same probability lower bound applies to the mode-conditioned
dimension event.

## Claim Boundary

The package does not establish that natural language is a tree, that real
Transformers use the assumed fixed linear path operators, that activation
modes are sparse and reusable, or that a low-dimensional active subtree
preserves task utility. A low-dimensional mode does not by itself imply MoE,
Top-1/Top-k routing, or functional expert specialization. The previous global
shared-linear propagation claim remains closed.

## Next Action

Decide whether to promote “task-conditioned low-dimensional active subtrees
that preserve held-out task utility” into a new Thinking Card. Router
architecture design remains parked until that decision.
