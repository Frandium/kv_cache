# Visualization and results

No full experiment has been run yet.

`plot_results.py` produces two paired bar plots for every backbone and hidden
size: frozen linear-probe suffix accuracy and centered prefix effective rank.
Each bar aggregates seeds separately for MTP=1 and MTP=3.

The behavioral plot answers whether a fresh readout can extract the suffix from
the early prefix state. The spectrum plot only shows whether representation
energy is distributed across more directions. A flatter spectrum does not prove
that those directions carry useful suffix information; that interpretation is
allowed only when probe accuracy changes in the same condition.

After a run, this document should record the observed seed-level values,
optimization failures, representative learning curves, and the resulting claim
boundary.

