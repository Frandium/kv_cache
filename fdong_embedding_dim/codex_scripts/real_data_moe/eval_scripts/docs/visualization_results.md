# Visualization results

No remote plots are generated. The remote suite deliberately returns compact
CSV evidence so plots can be produced locally without transferring hidden
states, route traces, predictor weights, or profiler data.

Planned local views are:

- layer × expert activation-share heatmaps;
- cache capacity versus loads and measured PPU latency;
- same-token upper-triangle recall@1/2/4 heatmaps;
- next-token lower-triangle recall@1/2/4 heatmaps;
- log training FLOPs versus downstream accuracy and loss.

Observed results and conclusions remain incomplete until the remote
`final_results.tar.gz` is returned and inspected.
