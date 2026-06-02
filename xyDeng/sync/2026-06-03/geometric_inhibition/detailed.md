# Detailed Record: H0603a Geometric Inhibition

## 0. Quick Recap

目的：

用组会可读的最小实验判断：在 uniform multi-B synthetic 中，slot-stable router initialization 加 geometric inhibition 是否稳定 top-1 MoE routing。

假设：

slot centroid prototype 中有可用的 slot 几何信息；如果 router row 从该 prototype 初始化，再用 token-level margin 和 router-center separation 约束训练，routing 应该更稳定地对齐 slot。

实验思路：

只跑 6 个条件：C0/C1/C2 是 dot router 的 random、slot-init、slot-init+geo；C3/C4/C5 是 cosine router 的对应版本。去掉 expert warmup，不跑 Zipfian，不做超参 sweep。所有 router 使用 `bias=False`，不存在可训练 $b_e$。

结论：

Q1 支持，slot-stable init 在 step 0 有效；Q2 支持，geometric inhibition 在 init 之外提供强额外稳定性；Q3 rival 被削弱，因为 confidence 上升伴随 route-slot NMI 大幅上升；Q4 只在 init-only 下部分支持 cosine；Q5 没有 accuracy tradeoff。

证据：

C1 vs C0 的 step-0 route-slot NMI 是 0.242 vs 0.009；C4 vs C3 是 0.242 vs 0.008。C2 vs C1 的 final route-slot NMI 是 1.000 vs 0.446；C5 vs C4 是 1.000 vs 0.566。所有条件 target accuracy 都是 1.000。

## 1. Run Identity

Job:

```text
pt-1dqucc3y
display name: ats-h0603a-geo-inhib-4gpu
quota: RESERVED
status: SUCCEEDED
```

Run name:

```text
h0603a_geometric_inhibition_standard_4gpu_20260603
```

Code workspace:

```text
Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/
```

Runner:

```text
scripts/run_h0603a_geometric_inhibition.py
```

Config:

```text
configs/h0603a_geometric_inhibition.json
```

Runtime log:

```text
logs/acp/h0603a_geometric_inhibition_standard_4gpu_20260603_runtime_20260602_163915.log
```

Raw result dir:

```text
results/h0603a_geometric_inhibition/h0603a_geometric_inhibition_standard_4gpu_20260603/
```

Raw figure dir:

```text
figures/h0603a_geometric_inhibition/h0603a_geometric_inhibition_standard_4gpu_20260603/
```

Curated result dir:

```text
Projects/from-attention-to-search/main/experiments/05_01_geometric_inhibition_anchor/
```

## 2. Data / Model / Training

Data:

```text
uniform multi-B synthetic
slots = 4
B identities = 256
sequence = [r_start, C_s, B_i, Y_{s,i}, r_end]
primary routing position = B_i position
positive assignment = a(s,i)=s
```

Splits:

| Split / Use | Per Slot | Total |
| --- | ---: | ---: |
| train | 5000 | 20000 |
| eval | 1600 | 6400 |
| calibration | 512 | 2048 |
| trajectory eval | 256 | 1024 |

Model:

```text
SparseTinyTransformer
d_model = 128
n_heads = 4
ffn_dim = 256
num_experts = 4
dropout = 0.0
router bias = False
```

Training:

```text
steps = 1600
batch_size = 384
seeds = 20260521, 20260522, 20260523
lr = 0.0008
weight_decay = 0.01
grad_clip = 1.0
```

Geometric loss:

```text
L_geo = 0.1 * L_tok + 0.02 * L_sep
token_margin = 1.0
center_max_cosine = 0.0
```

## 3. Conditions

| Condition | Router | Init | Geometric inhibition |
| --- | --- | --- | --- |
| C0 | dot | random | no |
| C1 | dot | slot-stable | no |
| C2 | dot | slot-stable | yes |
| C3 | cosine | random | no |
| C4 | cosine | slot-stable | no |
| C5 | cosine | slot-stable | yes |

Smoke passed before full run:

1. router gradient nonzero for all C0-C5;
2. geometric router gradient nonzero for C2/C5;
3. selected-gate sparse top-1 path preserved;
4. no router bias used.

## 4. Q1 Step-0 Initialization

Pairs:

```text
C1 vs C0
C4 vs C3
```

Metric:

```text
step-0 route-slot NMI
```

Observation:

| Pair | Step-0 NMI A | Step-0 NMI B | Delta |
| --- | ---: | ---: | ---: |
| C1 vs C0 | 0.242 | 0.009 | +0.234 |
| C4 vs C3 | 0.242 | 0.008 | +0.233 |

Interpretation:

slot-stable init 确实把 slot 信息注入到了 router 初始方向里。Prototype 构造有信息。

Boundary:

Step-0 NMI 只有约 0.24，不是完美 routing；它说明 prototype useful，不说明单靠 init 足够。

## 5. Q2 Geometric Inhibition Extra Contribution

Pairs:

```text
C2 vs C1
C5 vs C4
```

Metrics:

```text
final route-slot NMI
seed stability
route drift / trajectory
```

Observation:

| Pair | Final NMI A | Final NMI B | Delta | Final NMI std A | Final NMI std B |
| --- | ---: | ---: | ---: | ---: | ---: |
| C2 vs C1 | 1.000 | 0.446 | +0.554 | 0.000 | 0.320 |
| C5 vs C4 | 1.000 | 0.566 | +0.434 | 0.000 | 0.491 |

Interpretation:

在已经有 slot-stable init 的前提下，geometric inhibition 仍然提供强额外稳定性。最重要的是 seed stability：C2/C5 final NMI std 为 0。

Boundary:

这里的 positive assignment 是 $a(s,i)=s$，所以结论是 supervised slot-assignment stabilization，不是 label-free discovery。

## 6. Q3 Confidence-Only Rival

Rival:

```text
selected gate confidence rises, but route-slot NMI does not change
```

Observation:

| Pair | Delta confidence | Delta final route-slot NMI | Confidence-only rival |
| --- | ---: | ---: | --- |
| C2 vs C1 | +0.028 | +0.554 | false |
| C5 vs C4 | +0.029 | +0.434 | false |

Interpretation:

geometric inhibition does sharpen gates, but it also changes routing alignment. Therefore the main rival "只是 gate sharper" is weakened.

Boundary:

This does not prove task utility is caused by confidence. It only shows confidence increase is not the only observed change.

## 7. Q4 Cosine Router

Pairs:

```text
C4 vs C1
C5 vs C2
```

Observation:

| Pair | Final route-slot NMI A | Final route-slot NMI B | Delta |
| --- | ---: | ---: | ---: |
| C4 vs C1 | 0.566 | 0.446 | +0.121 |
| C5 vs C2 | 1.000 | 1.000 | 0.000 |

Interpretation:

cosine 在 init-only 下更稳定一些，但一旦加入 geometric inhibition，dot 和 cosine 都达到 final NMI 1.000。Cosine 不是必要条件。

## 8. Q5 Accuracy Tradeoff

Metric:

```text
target accuracy
```

Observation:

All conditions have target accuracy 1.000.

Interpretation:

没有出现 NMI 上升但 accuracy 下降的 tradeoff。可以说 routing 更整齐且没有损害任务学习。

Boundary:

Accuracy 在这个 synthetic 中太容易达到 1.000，因此不能作为 specialization 的主证据。

## 9. Key Figures

Route-slot NMI trajectory:

![Route-slot NMI trajectory](figures/route_slot_nmi_trajectory.png)

Selected gate confidence trajectory:

![Selected gate confidence trajectory](figures/selected_gate_confidence_trajectory.png)

Route heatmap:

![Route-slot heatmap step0 final](figures/route_slot_heatmap_step0_final.png)

Router center geometry:

![Router center offdiag cosine trajectory](figures/router_center_offdiag_cosine_trajectory.png)

## 10. Artifact Map

Curated tables:

```text
tables/h0603a_decision_metrics_compact.csv
tables/h0603a_question_pair_effects.csv
tables/h0603a_confidence_only_rival_check.csv
tables/h0603a_summary_by_condition.csv
tables/h0603a_summary_by_seed.csv
```

Curated figures:

```text
figures/route_slot_nmi_trajectory.png
figures/selected_gate_confidence_trajectory.png
figures/route_slot_heatmap_step0_final.png
figures/router_center_offdiag_cosine_trajectory.png
figures/router_weight_to_slot_center_cosine_trajectory.png
figures/router_weight_pca_step0_final.png
figures/assignment_utility_trajectory.png
figures/forced_expert_loss_heatmap_final.png
```

Repro command:

```text
python scripts/run_h0603a_geometric_inhibition.py \
  --config configs/h0603a_geometric_inhibition.json \
  --run-name h0603a_geometric_inhibition_standard_4gpu_20260603 \
  --run-stage full \
  --parallel \
  --max-parallel 4 \
  --trajectory
```

## 11. Current Claim

Can claim:

Given external slot assignment $a(s,i)=s$, geometric inhibition stabilizes slot-aligned routing beyond slot-stable initialization in uniform multi-B synthetic.

Cannot claim:

This does not prove unsupervised specialization, full-LM behavior, Zipfian robustness, or that expert utility is fully solved.

Next decision:

For group meeting, use this as a clean mechanism slide. For research next step, test whether the same stabilization survives Zipfian imbalance or less-oracle prototype assignment.
