# Protocol: A06_24_synthetic_common_rare_transformer_moe

## 0. 审批快照

审批状态：用户本轮明确要求完成 anchor、protocol、实验、结果报告、0703 meeting brief，并提交 4 卡 full run。

目的：在无位置编码的一层 Transformer + 一层 MoE common/rare 合成任务中，检验 common subtraction 是 rare-feature separator，还是只修复 concentration/load。

主 anchor：`../../../problem_anchors/06_geometry_proxy_preservation/06_24_synthetic_common_rare_transformer_moe_anchor.md`

Anchor 决策问题：common subtraction 能否产生 rare-feature expert separation，还是只是一种 load/concentration control，rare separation 仍需要 route-relevant centers 和 preservation controls？

主物理先验：全局 common subtraction 可以移除共享偏置，但不提供 route relevance，也不保证 rare-rare separation。

核心模型项：$h_i=c+r_i$，检查路由读 $r_i$ 是否真的比 load repair 更能提高 rare-feature NMI。

证伪条件：common-subtracted all-position routing 在多个 slot lengths 上达到接近 route-position/oracle 的 rare-feature NMI 和 rare margin，同时 slot-start NMI 低、loss 不恶化。

实验角色：root-cause audit + method-readiness gate。

主指标：balanced held-out route-position set 上的 rare-feature NMI。

最小设置：无位置编码的一层 causal Transformer，一层 weighted top-1 MoE，一个高频 common feature，三个低频 rare features，repeated feature slots，neutral background，feature-specific targets。

基础配置：seeds 0--7，slot lengths 1/2/4/8，4 features，4 experts，balanced held-out eval，imbalanced calibration/training，full checkpoints 0/10/40/80/160。

运行条件：random raw、random common-subtracted、all-position k-means raw、all-position common-subtracted k-means、route-position k-means raw、route-position residual-input k-means、oracle raw、oracle residual input、oracle row projection。

通过：common-subtracted all-position 不能产生 rare-feature separation，而 route-position/oracle centers 可以；preservation controls 提高 final rare margin 或降低 sign flip，且不损害 task loss。

失败：common-subtracted all-position 在 rare-feature NMI/margin 上跨 slot lengths 匹配 route-position/oracle，且无位置泄漏。

不充分：target accuracy 失败、slot-start NMI 高、route-position positive control 失败、或 full cells 完成不足。

不可声称：真实 DCLM、自然语言语义专家、可部署路由、优化器最优性。

审批决定：执行 full synthetic run。

## 1. 术语解释

| 术语 | 中文含义 | 具体对象或计算方式 | 单位或公式 | 为什么影响判断 | 不能证明 |
| --- | --- | --- | --- | --- | --- |
| common feature | 高频合成特征 | feature id 0，训练/校准概率 0.70 | 概率质量 | 检查频率主导路由 | 真实 common 语义 |
| rare feature | 低频合成特征 | feature ids 1--3，总概率 0.30 | feature id | 检查 rare-rare separation | 真实 rare 语义 |
| route position | 任务相关审计位置 | feature slot 最后一个 token，用来预测 target token | 序列位置 | 绑定任务相关状态 | 真实 route selector |
| no-position model | 不加显式位置编码的模型 | token embedding + causal attention | 架构 | 排除位置 shortcut | 所有上下文长度效应 |
| rare-feature NMI | rare feature 和 expert assignment 的一致性 | rare examples 上 NMI(feature id, route) | 0--1 | 主指标 | expert utility |
| rare margin | 匹配 rare expert 的分数优势 | $z_{i,m(f_i)}-\max_{e\ne m(f_i)}z_{i,e}$ | logit 差 | 检查 basin 厚度 | utility |
| joint feature score | common/rare 和 rare-feature 分离的联合 guard | `rare_feature_NMI * common_rare_NMI` | 无量纲分数 | 防止只通过一个分离轴时过度声称 | expert utility |
| sign flip | margin 跨过 0 | step-0 positive rare margin later non-positive | 比例 | 检查 preservation | 不能说明原因 |
| slot-start NMI | 位置干扰一致性 | NMI(slot_start, route) | 0--1 | 位置泄漏 guard | 完全无位置效应 |

## 2. Anchor 对齐

决策问题：common subtraction 是否产生 rare-feature separation，还是只降低 concentration？

物理先验：common subtraction 能去共享偏置，但不能识别 route-relevant feature centers。

核心模型项：router score 中 $w_e^\top c$ 与 $w_e^\top r_i$。

证伪条件：common-subtracted all-position centers 匹配 route-position/oracle centers 的 rare-feature NMI 和 rare margin。

结论边界：仅限 synthetic no-position Transformer-MoE。

## 3. Tested Hypothesis

主假设 H1：common subtraction 是 load/concentration repair，不是 rare-feature separator。H2/H3 是 guard：route-position centers 应强于 all-position centers；valid init 后再检查 residual input 或 row projection 是否更能保持。

## 4. Rival Explanations

- load-only improvement：max load 改善但 rare-feature NMI 不变。
- position leakage：路由追踪 slot start，不是 feature。
- binary common/rare separation：common 和 rare 分开，但 rare features 彼此合并。
- oracle leakage：只有 label-based centroids 通过。
- training failure：模型没学会合成目标，导致路由方法看起来失败。

## 5. Data / Model / Algorithm / Objective

数据：序列包含 neutral background tokens、repeated feature slot 和 slot 后 target token。Feature 0 是 common；features 1--3 是 rare。Slot starts 在合法位置上平衡；模型没有显式位置编码。

模型：一层 causal self-attention + 一层 weighted top-1 MoE + LM head。selected expert output 乘以 selected softmax gate probability，使 router rows 有梯度。

目标：route position 的 cross-entropy，预测 feature-specific target token。

Discovery：训练前提取 hidden states，在 all positions 或 route positions 上拟合 centers，必要时减 common mean，再用 equal-norm centers 初始化 router rows。

Training：各条件用 imbalanced feature distribution 训练，在 balanced held-out route-position set 上评估。

## 6. Conditions, Seeds, And Checkpoints

Full：seeds 0--7，slot lengths 1/2/4/8，checkpoints 0/10/40/80/160。

主要条件：`random_raw`、`random_common_subtract`、`allpos_kmeans_raw`、`allpos_kmeans_common_subtract`、`route_kmeans_raw`、`route_kmeans_residual_input`、`oracle_raw`、`oracle_residual_input`、`oracle_row_projected`。

主表：`step0_discovery.csv`、`training_trajectory.csv`。

## 7. Primary Metric

Rare-feature NMI 决定判断，因为当前问题是 rare features 彼此是否分开。整体 load 或整体 feature NMI 会被高频 common feature 掩盖。Joint feature score 是必要 guard，因为 rare features 可能彼此分开，但 common 仍和 rare 共享 expert。

## 8. Secondary Metrics

Overall feature NMI、common/rare binary NMI、joint feature score、rare margin mean/p05、sign-flip rate、max load、effective experts、target accuracy、route-position loss、slot-start NMI。

## 9. Known Good / Known Bad / Known Confusing Cases

Known good：oracle feature centers 应在 step 0 获得高 rare-feature NMI。

Known bad：random raw 不应稳定分开 rare features。

Known confusing：common subtraction 可能降低 max load 或改善 common/rare 粗分，但 rare features 仍合并。

## 10. Stage-Level Profiling Plan

检查 data audit、step-0 discovery、training target accuracy、preservation margin/sign flip、position guard。

## 11. Algorithm Specification

生成 calibration/eval batches；初始化无位置模型；提取 hidden；拟合 all-position/route-position centers；构造 router init；step-0 评估；训练；checkpoint 评估；聚合表和图。

## 12. Success / Failure / Insufficient Evidence

成功：common subtraction 改善 concentration 但不改善 rare separation；route/oracle centers 分开 rare；residual controls 不损害 loss 并改善或保持 margin。

失败：common-subtracted all-position centers 跨 slot lengths 匹配 route/oracle。

不充分：positive controls 或 target accuracy 失败、slot-start NMI 高、full cells 不够。

## 13. What This Cannot Claim

不能声称真实 DCLM transfer、自然语言 semantic expert、最终方法、optimizer optimality。

## 14. Review Notes

旧 vector-only A06_26 已改为 A06_24_toy；本 protocol 是主线 A06_24_synthetic，因为它使用一层 Transformer + 一层 MoE，并显式去掉位置编码。
