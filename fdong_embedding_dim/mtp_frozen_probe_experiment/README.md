# Frozen-backbone NTP vs MTP experiment

Controlled sequences have the form:

```text
P_i B_j_0 B_j_1 S_i
```

Every `(prefix, suffix) x bone` pair is generated. Under the repository
convention, `MTP=1` is ordinary next-token prediction and `MTP=3` predicts all
three future offsets. The central diagnostic freezes a trained backbone,
discards its original heads, and trains the same fresh probe to predict `S_i`
from `h(P_i)`.

Run only the structural smoke test:

```bash
python3 -m fdong_embedding_dim.mtp_frozen_probe_experiment.smoke_test
```

Run the configured sweep in the background:

```bash
nohup bash fdong_embedding_dim/mtp_frozen_probe_experiment/run_experiment.sh \
  > fdong_embedding_dim/mtp_frozen_probe_experiment/run.log 2>&1 &
```

The sweep writes `results.json`, `summary.csv`, and `summary.png` under
`fdong_embedding_dim/outputs/mtp_frozen_probe_v1` by default.

## Compositional v2

The harder dataset uses unseen prefix compositions:

```text
X_a Y_b B_j_0 B_j_1 S_((a+b) mod M)
```

The probe reads `h(X_a,Y_b)` and predicts the suffix three offsets away. Entire
`(a,b)` pairs are held out, while every factor and suffix class remains present
in both splits. Probe inputs are layer-normalized to remove hidden-norm scale as
an explanation for learning-speed differences.

Run the focused two-layer-MLP experiment:

```bash
nohup bash fdong_embedding_dim/mtp_frozen_probe_experiment/run_compositional_mlp.sh \
  > fdong_embedding_dim/mtp_frozen_probe_experiment/run_compositional_mlp.log 2>&1 &
```
