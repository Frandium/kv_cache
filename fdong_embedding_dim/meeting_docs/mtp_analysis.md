# MTP analysis: current question, experiments, results, and artifacts

Date: 2026-07-12

This note records the current MTP experiment state under `fdong_embedding_dim`.  It is meant to be a restart point for the next analysis session: what question we are trying to answer, what synthetic data we built, what result we already observed, where the code is, and where the saved checkpoints are.

## 1. Question we are trying to answer

We are studying MTP as a training objective, not speculative decoding.

Convention in this experiment:

```text
MTP=1 means ordinary next-token prediction.
MTP=3 means one shared backbone state predicts offsets 1, 2, and 3.
```

The main question is:

```text
Why can MTP make the model's offset-1 / next-token loss drop faster in early-to-mid training, even though MTP does not add new information to the dataset?
```

We separate three claims:

1. MTP may help training speed.
2. MTP may help small models more than large models.
3. MTP does not necessarily improve the final converged capability of large, sufficiently trained models.

The current experiment only addresses the first two claims in a tiny controlled setting.  It does not prove a large-language-model downstream improvement.

## 2. Working hypothesis

MTP does not add new information.  The sequence distribution is the same.  The possible benefit is an optimization and representation effect.

For MTP=3, a hidden state receives gradients from:

```text
offset-1: predict x_{t+1}
offset-2: predict x_{t+2}
offset-3: predict x_{t+3}
```

The hypothesis is:

```text
If the extra future-token losses provide useful auxiliary supervision, then the shared backbone may learn a representation that helps offset-1 prediction earlier.
```

The failure mode is also clear:

```text
If offset-2 and offset-3 are high-entropy, noisy, or only aligned by an artificial fixed offset, MTP total loss can be misleading.
```

Therefore the key diagnostic is not only MTP total loss.  We must compare:

```text
NTP offset-1 CE
vs
MTP offset-1 CE
```

This tells us whether MTP actually helps the ordinary next-token task.

## 3. Synthetic data designs

### 3.1 Fixed-bone lookup data

Initial data:

```text
P_i B_j0 B_j1 S_i
```

The data is a Cartesian product over prefix `i` and bone `j`.

From the prefix position `P_i`, the MTP targets are:

```text
offset-1: B_j0
offset-2: B_j1
offset-3: S_i
```

This dataset has a strong artifact:

```text
S_i is always at offset 3 from P_i.
```

So MTP=3 can quickly learn:

```text
P_i -> S_i
```

This makes MTP total loss drop fast, but that does not prove that MTP helps next-token prediction.

Observed result:

```text
MTP total CE drops faster.
MTP offset-3 CE quickly approaches zero.
MTP offset-1 CE is not faster early.
```

Interpretation:

```text
The fixed-bone data mainly shows a fixed-offset shortcut.
```

### 3.2 Variable-bone lookup data

Improved data:

```text
P_i B_{j,0} ... B_{j,L-1} S_i
```

with:

```text
L in {1, 2, 3, 4}
```

Sequences are padded to length 6:

```text
P + up to 4 bone tokens + S
```

PAD tokens are masked out of the loss.

For MTP=3, the suffix offset from `P_i` is:

| bone length L | suffix offset from P_i | inside MTP=3 horizon |
| --- | ---: | --- |
| 1 | 2 | yes |
| 2 | 3 | yes |
| 3 | 4 | no |
| 4 | 5 | no |

This removes the fixed offset-3 shortcut.  If MTP still improves offset-1 loss here, that is stronger evidence for a representation / optimization effect.

## 4. Current main result

The three-seed variable-bone experiment shows:

```text
MTP improves offset-1 / next-token CE consistently across seeds.
MTP total CE is not consistently better across seeds.
MTP offset-3 no longer quickly goes to zero.
```

Final train offset-1 CE:

| seed | NTP offset-1 CE | MTP offset-1 CE | result |
| ---: | ---: | ---: | --- |
| 971 | 1.1394 | 0.9464 | MTP better |
| 972 | 1.1893 | 0.9926 | MTP better |
| 973 | 1.1178 | 0.9965 | MTP better |

Final test offset-1 CE:

| seed | NTP offset-1 CE | MTP offset-1 CE | result |
| ---: | ---: | ---: | --- |
| 971 | 3.1466 | 1.7851 | MTP better |
| 972 | 4.2285 | 1.2705 | MTP better |
| 973 | 2.6590 | 1.4885 | MTP better |

Final train total CE:

| seed | NTP total CE | MTP total CE | result |
| ---: | ---: | ---: | --- |
| 971 | 1.1394 | 1.0800 | MTP better |
| 972 | 1.1893 | 1.1223 | MTP better |
| 973 | 1.1178 | 1.1276 | MTP slightly worse |

The total CE result is less stable because MTP total CE averages offset-1, offset-2, and offset-3:

```text
MTP total CE = (CE_offset1 + CE_offset2 + CE_offset3) / 3
```

In the variable-bone data, offset-2 and offset-3 are mixed targets and do not always represent the suffix.  They can therefore drag down the total objective even when offset-1 improves.

## 5. Claim boundary after current experiments

Supported:

```text
In this small controlled variable-bone setting, MTP training consistently improves the learned model's offset-1 / next-token CE across three seeds.
```

Also supported:

```text
The fixed-bone MTP total-loss advantage was partly an artifact of suffix being fixed at offset 3.
```

Not supported yet:

```text
MTP always improves total training objective.
MTP always improves final downstream performance.
MTP necessarily helps large models after sufficient training.
This synthetic data is close to real language.
```

Better wording:

```text
The experiment reproduces a qualitative small-model MTP behavior reported by large labs: multi-token auxiliary supervision can improve early-to-mid learning and can be more helpful under limited capacity.  The likely mechanism is optimization / representation shaping, not new information.
```

## 6. Code locations

Main experiment directory:

```text
fdong_embedding_dim/mtp_frozen_probe_experiment/
```

Important files:

```text
fdong_embedding_dim/mtp_frozen_probe_experiment/data.py
```

Defines:

```text
make_cartesian_patterns
make_compositional_patterns
make_variable_lookup_patterns
```

The current main dataset is `make_variable_lookup_patterns`.

```text
fdong_embedding_dim/mtp_frozen_probe_experiment/model.py
```

Defines:

```text
CausalBackbone
MultiTokenModel
Probe
```

The current main model is:

```text
backbone = mlp
hidden_size = 4
mtp = 1 or 3
```

```text
fdong_embedding_dim/mtp_frozen_probe_experiment/experiment.py
```

Defines masked MTP loss and evaluation helpers:

```text
multi_token_loss
evaluate_offsets
evaluate_next_token_positions
train_frozen_probe
```

For variable-length data, PAD is excluded through `loss_mask`.

```text
fdong_embedding_dim/mtp_frozen_probe_experiment/learning_curve.py
```

Runs the learning-curve experiment and writes:

```text
learning_curve.csv
learning_curve_config.json
vocabulary.json
optional checkpoints
```

It supports:

```text
--dataset variable_lookup
--checkpoint-steps
--checkpoint-dir
```

```text
fdong_embedding_dim/mtp_frozen_probe_experiment/plot_learning_curve.py
```

Plots:

```text
NTP offset-1 CE vs MTP offset-1 CE
MTP total CE and per-offset CE
```

Start scripts:

```text
fdong_embedding_dim/mtp_frozen_probe_experiment/run_learning_curve.sh
fdong_embedding_dim/mtp_frozen_probe_experiment/run_variable_learning_curve_mps.sh
fdong_embedding_dim/mtp_frozen_probe_experiment/run_variable_learning_curve_checkpoints_mps.sh
```

## 7. Result locations

Fixed-bone learning curve:

```text
fdong_embedding_dim/outputs/mtp_learning_curve_v1/
```

Contains:

```text
learning_curve.csv
learning_curve.png
learning_curve_config.json
vocabulary.json
```

Variable-bone single-seed learning curve:

```text
fdong_embedding_dim/outputs/mtp_variable_learning_curve_v1/
```

Variable-bone three-seed learning curve:

```text
fdong_embedding_dim/outputs/mtp_variable_learning_curve_seeds_v1/
```

Contains the result used for the three-seed conclusion:

```text
learning_curve.csv
learning_curve.png
learning_curve_config.json
vocabulary.json
```

Variable-bone checkpoint run:

```text
fdong_embedding_dim/outputs/mtp_variable_learning_curve_checkpoints_seed971_v1/
```

Despite the directory name, the current contents include seeds:

```text
971, 972, 973
```

and MTP settings:

```text
mtp=1, mtp=3
```

The checkpoint steps are:

```text
0, 20, 60, 100, 140, 200, 500, 1000, 3000
```

Checkpoint directory:

```text
fdong_embedding_dim/outputs/mtp_variable_learning_curve_checkpoints_seed971_v1/checkpoints/
```

Example checkpoint files:

```text
mlp_d4_seed971_mtp1_step000140.pt
mlp_d4_seed971_mtp3_step000140.pt
mlp_d4_seed971_mtp1_step003000.pt
mlp_d4_seed971_mtp3_step003000.pt
```

Each checkpoint contains:

```text
run_name
step
seed
model_state_dict
optimizer_state_dict
metrics
config
model_metadata
```

## 8. Re-run commands

Run the three-seed variable-bone learning curve without checkpoints:

```bash
cd /Users/bytedance/kv_cache

nohup bash -lc '
python3 -m fdong_embedding_dim.mtp_frozen_probe_experiment.learning_curve \
  --output-dir fdong_embedding_dim/outputs/mtp_variable_learning_curve_seeds_v1 \
  --dataset variable_lookup \
  --backbones mlp \
  --hidden-sizes 4 \
  --seeds 971,972,973 \
  --mtps 1,3 \
  --num-prefixes 8 \
  --num-bones 8 \
  --min-bone-length 1 \
  --max-bone-length 4 \
  --holdout-stride 4 \
  --train-steps 3000 \
  --learning-rate 3e-2 \
  --log-every 20 \
  --device mps

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig_mtp_variable_learning_curve \
python3 -m fdong_embedding_dim.mtp_frozen_probe_experiment.plot_learning_curve \
  --input fdong_embedding_dim/outputs/mtp_variable_learning_curve_seeds_v1/learning_curve.csv \
  --output fdong_embedding_dim/outputs/mtp_variable_learning_curve_seeds_v1/learning_curve.png
' > fdong_embedding_dim/mtp_frozen_probe_experiment/run_variable_learning_curve_seeds.log 2>&1 &
```

Run the checkpoint version:

```bash
cd /Users/bytedance/kv_cache

nohup bash fdong_embedding_dim/mtp_frozen_probe_experiment/run_variable_learning_curve_checkpoints_mps.sh \
  > fdong_embedding_dim/mtp_frozen_probe_experiment/run_variable_learning_curve_checkpoints_mps.log 2>&1 &
```

If only one seed is desired, update the script or call `learning_curve.py` directly with:

```text
--seeds 971
```

## 9. Next analysis plan using checkpoints

The next goal is to explain why MTP improves offset-1 in this small model.

Planned analyses:

### 9.1 Representation decodability over time

For each checkpoint, freeze the backbone and train or fit probes from hidden states:

```text
h(P_i) -> prefix i
h(P_i) -> suffix S_i
h(P_i) -> bone length L
h(B position) -> suffix S_i
```

Question:

```text
Does MTP make useful prefix/suffix information linearly readable earlier than NTP?
```

### 9.2 Hidden-state geometry

For each checkpoint, compute hidden states for all examples and inspect:

```text
P_i state clustering by prefix i
P_i state clustering by suffix S_i
state spectrum / effective rank
pairwise distances within same prefix vs different prefix
```

Question:

```text
Does MTP organize the small hidden space more efficiently?
```

### 9.3 Gradient alignment

From each checkpoint, recompute gradients on the same batch:

```text
gradient from offset-1 loss
gradient from offset-2 loss
gradient from offset-3 loss
```

Measure:

```text
gradient norm
cosine(offset1, offset2)
cosine(offset1, offset3)
cosine(offset2, offset3)
```

Question:

```text
Are the auxiliary offset gradients aligned with offset-1 during the period where MTP starts to outperform NTP?
```

### 9.4 Head vs backbone separation

Compare:

```text
trained MTP offset-1 head
fresh offset-1 probe on frozen MTP backbone
fresh offset-1 probe on frozen NTP backbone
```

Question:

```text
Is the MTP gain mostly in the backbone representation or mostly in the trained head?
```

## 10. Current caveats

This experiment is still synthetic and small.

Important caveats:

```text
The data has symbolic tokens, not natural-language syntax.
The prefix-to-suffix mapping is deterministic.
The sequence length is at most 6.
The model is extremely small.
The current backbone is a causal-mean MLP, not a full Transformer.
```

Therefore the result should be framed as:

```text
a controlled mechanism test
```

not:

```text
a direct proof that MTP improves real LLM downstream performance.
```

