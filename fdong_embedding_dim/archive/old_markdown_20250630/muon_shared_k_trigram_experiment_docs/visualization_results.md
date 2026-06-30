# Visualization Results

This file records only supported claims after runs complete.

Expected artifacts:

- `results/summary.csv`
- `results/history.csv`
- `results/lr_selection.csv`
- `results/estimator_diagnostics.csv`
- `results/learning_curves.png`
- `results/batch_gap.png`

Reading guide:

- `learning_curves.png`
  - compare `common_accuracy`, `tail_accuracy`, and `internal_accuracy`
  - the internal pattern is the hardest pure tail-family check
- `batch_gap.png`
  - compare the median `first_stable_all_groups_full_accuracy_step` under `population`, `64`, and `16`
  - if the hypothesis is right, shrinking batch coverage should hurt `withK_zipf`, especially for tail/internal learning

Claim boundary template:

- supported:
  - exact population Muon does or does not reduce the Zipf convergence gap in this attention toy
  - minibatch Muon does or does not keep that benefit
  - sqrt reweight does or does not recover part of the minibatch gap
- not supported:
  - any direct claim about full-scale real-data Transformers
  - any claim that minibatching is the only failure source

Current run summary:

- run:
  - `steps=300`
  - `seeds=0,1,2`
  - `batch in {population, 64, 16}`
  - LR selected on `withK_uniform + population + raw`
  - selected LR:
    - `adam = 0.03`
    - `muon = 0.03`

Supported by the current run:

- Exact population does help.
  - `withK_zipf + adam + raw`: median stable all-group step `79`
  - `withK_zipf + adam + raw + batch16`: `156`
  - `withK_zipf + muon + raw`: median stable all-group step `92`
  - `withK_zipf + muon + raw + batch16`: `199`
  - So shrinking batch coverage clearly slows learning in this attention toy for both optimizers.

- Canonical Muon does not fully solve the Zipf gap in this setting.
  - Under exact population, `withK_uniform + muon + raw` median stable all-group step is `76`
  - Under exact population, `withK_zipf + muon + raw` is `92`
  - Muon remains slower on Zipf than on uniform.

- Muon does flatten the hidden routing spectrum.
  - `withK_zipf + adam + raw + population`: final `Bqk top1 energy` mean `0.9639`, `effective rank` mean `1.1752`
  - `withK_zipf + muon + raw + population`: final `Bqk top1 energy` mean `0.7581`, `effective rank` mean `1.8047`
  - So the hidden routing matrix is materially less rank-1 under Muon.

- In this toy, flatter hidden spectrum is not enough to guarantee faster convergence.
  - Despite flatter `Bqk`, `withK_zipf + muon + raw + population` is slower than `adam + raw + population`
  - Therefore "flatten spectrum" and "learn tail faster" are not equivalent in this tied-embedding attention task.

- `sqrt_reweight` interacts differently with Adam and Muon.
  - For Adam, `sqrt_reweight` helps the internal family under `batch16`:
    - raw internal median `133`
    - sqrt reweight internal median `78`
  - But it worsens Adam's all-group stable step:
    - raw `156`
    - sqrt reweight `211`
  - For Muon, `sqrt_reweight` helps the internal family strongly:
    - population internal median `77 -> 52`
    - batch64 internal median `132 -> 36`
  - But under `batch16`, Muon + `sqrt_reweight` did not reach stable all-group-full accuracy within `300` steps for any seed.

- Minibatch noise changes Muon's hidden update more than it changes the raw hidden gradient.
  - `withK_zipf + raw + batch16`:
    - hidden gradient relative bias `0.0455`
    - Muon-transformed hidden update relative bias `0.4498`
  - `withK_zipf + sqrt_reweight + batch16`:
    - hidden gradient relative bias `0.0600`
    - Muon-transformed hidden update relative bias `0.4148`
  - This supports the statement that `E[Muon(g_batch)] != Muon(E[g_batch])` matters in the sequence-attention toy too.

Not supported by the current run:

- That global population Muon can completely remove the common-tail gap.
- That `sqrt_reweight` is uniformly beneficial for all convergence metrics.
- That minibatching is the only reason Muon is incomplete on real data.
