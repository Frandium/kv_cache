# Shared-K Trigram Muon Design

Objective:

Test the minibatch-frequency hypothesis in the user-aligned synthetic sequence setting rather than in the earlier matrix-classification toy.

Data contract:

- shared token `K`
- four groups `A/B/C/D`
- each group has tokens `G0,G1,G2`
- each group follows:
  - `G0,G1 -> K`
  - `G1,K -> G2`
  - `K,G2 -> G0`
  - `G2,G0 -> G1`
- `withK_uniform`: `A/B/C/D = 0.25/0.25/0.25/0.25`
- `withK_zipf`: `A/B/C/D = 0.70/0.10/0.10/0.10`

Model contract:

- tied input/output embedding `E`
- one attention-only layer
- one head
- causal two-token context
- residual connection `final_h = attn_out + h2`
- logits `final_h @ E^T`

Optimizer contract:

- `adam`: Adam-style update on all trainable matrices
- `muon`: canonical hybrid
  - `Wq/Wk/Wv/Wo` use Muon
  - tied embedding `E` uses Adam-style update

This is intentional. It matches standard Muon usage more closely than applying Muon to the embedding table.

Primary hypothesis:

If global population gradients expose all shared-K and tail patterns every step, Muon may flatten hidden-matrix updates enough to reduce the Zipf common-tail convergence gap. If minibatches hide rare tail patterns on many steps, the Muon-transformed update should still be dominated by whichever hidden directions are present in the batch.
