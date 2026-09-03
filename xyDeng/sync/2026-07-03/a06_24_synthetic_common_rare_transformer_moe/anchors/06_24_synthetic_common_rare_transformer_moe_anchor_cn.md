# A06_24 Synthetic：一层 Transformer-MoE 中 common/rare 路由审计

## 0. researcher judgment record

**现象：** 旧 A06 证据已经显示，简单减去 common 可以缓解 expert 集中，但不能自动把不同 feature 分到不同 expert。刚做的 A06_24_toy 向量实验有机制启发，但它不是一层 Transformer + 一层 MoE 的训练表面，所以不能作为主线可靠证据。

**机制猜想：** 在 common/rare 不均匀分布下，普通最高分唯一专家路由会优先使用高频 common feature 或非 route 状态。减 common 可能改善负载，但 rare feature 彼此分开需要 route-relevant center，或者需要在正确初始化后保护 margin。

**关键变量：** feature 频率、slot 长度、route-position hidden state、all-position hidden state、common vector、router row、rare-feature 路由一致性、rare margin、sign flip、slot-start 位置干扰。

**因果关系：** 如果失败主要来自 common 集中，那么减 common 应该提高 rare-feature 分离；如果失败来自 route-relevant center selection，那么减 common 只会改善负载或 common/rare 粗分，而 rare features 仍会合并。

**可观察指标：** 主指标是 rare-feature 路由一致性，即只在 rare features 上计算 feature id 和 routed expert 的归一化互信息（rare-feature NMI）。必要 guard 是 joint feature score，用来同时检查 common-vs-rare 分离。

**替代解释：** 改善可能只是负载更均匀、位置泄漏、common/rare 二分类而非 rare-rare 分离、oracle label 泄漏，或 toy hidden 构造不能迁移到 Transformer-MoE。

**决策：** 用无位置编码的一层 Transformer + 一层 MoE 合成任务判断：common subtraction 是只能修 concentration，还是能真正建立 rare feature-level expert separation。

## 1. Problem Definition

**父问题：** A06 研究 route-relevant proxy discovery、initialization 和 early preservation。

**当前子问题：** 判断 common subtraction 在真实训练的 synthetic Transformer-MoE 表面中，是否能分开 common/rare features；还是只改善集中，而 rare separation 仍需要 route-relevant centers 与 preservation controls。

**术语解释：**

| 术语 | 中文含义 | 具体对象或计算方式 | 单位或公式 | 为什么影响当前判断 | 不能证明什么 |
| --- | --- | --- | --- | --- | --- |
| common feature | 高频公共特征 | feature id 0 在校准和训练中高频出现 | 概率质量 | 检查路由是否被频率主导 | 自然语言 common 语义 |
| rare feature | 低频稀有特征 | feature id 1--3 | feature id | 检查 rare 彼此是否分开 | 真实稀有语义特征 |
| route position | 路由审计和预测目标所在位置 | repeated feature slot 的最后一个 token | 序列下标 | 把审计绑定到任务相关状态 | 真实语言中的 route relevance detector |
| no-position model | 不加显式位置编码的模型 | token embedding + causal attention，无 learned/sinusoidal position embedding | 架构设定 | 排除显式位置 shortcut | 不能消除所有因果上下文长度效应 |
| rare-feature NMI | rare feature id 与 expert route 的一致性 | 只在 rare examples 上算 NMI(feature id, routed expert) | 0--1 | 当前主指标 | expert utility 或语义 |
| rare margin | rare feature 匹配 expert 相对竞争 expert 的分数优势 | $z_{i,m(f_i)}-\max_{e\ne m(f_i)}z_{i,e}$ | logit difference | 检查 rare route 是否有稳定 basin | 不能单独证明训练有用 |
| joint feature score | common/rare 分离和 rare-feature 分离的联合 guard | `rare_feature_NMI * common_rare_NMI` | 无量纲分数 | 防止 rare 彼此分开但 common 仍混进 rare 时过度声称 | expert utility |
| slot-start NMI | 路由和 slot 起点的一致性 | NMI(slot_start, routed expert) | 0--1 | 位置泄漏 guard | 不能证明完全没有位置效应 |

**Decision question:** 在无位置编码的一层 Transformer + 一层 MoE common/rare 合成任务中，common subtraction 能否产生 rare-feature expert separation，还是只是一种 load/concentration control？

**Not in scope:** 真实 DCLM、语义专家、最终部署方法、理论最大学习率、或证明所有 common-removal 方法都失败。

## 2. Physical Priors

**P1: 频率不均匀会让 load repair 看起来像 specialization。**  
含义：高频 common feature 会主导路由负载，所以方法可能降低 max load，但没有分开 rare features。  
可能错误：如果 common subtraction 稳定提高 rare-feature NMI 和 rare margin，同时 slot-start NMI 保持低。

**P2: 全局 common subtraction 不提供 route relevance。**  
含义：减去一个全局 common vector 不告诉 router 哪些 hidden states 才是任务相关 route states。  
可能错误：如果 all-position common-subtracted centers 在 rare-feature 指标上匹配 route-position/oracle centers。

**P3: 有效初始化和训练保持是两个问题。**  
含义：route-relevant centers 可能在 step 0 分开 rare features，但早期训练仍可能擦掉该分离。  
可能错误：如果 raw ordinary training 和 residual controls 一样能保持 rare separation。

## 3. Falsifiable Hypotheses

**H1:** common subtraction 是 load/concentration repair，不是 rare-feature separator。  
支持条件：common-subtracted random/all-position 条件改善 load，但 rare-feature NMI/margin 不超过 raw baselines。  
削弱条件：common subtraction 单独跨 slot lengths 稳定提高 rare-feature NMI 和 rare margins。

**H2:** 不均匀分布下需要 route-position centers 才能分开 rare features。  
支持条件：route-position k-means 或 oracle feature centroids 在 rare-feature NMI/margin 上超过 all-position common-subtracted centers。  
削弱条件：all-position common-subtracted centers 在没有位置泄漏时匹配 route-position centers。

**H3:** 正确初始化后仍需要 preservation controls。  
支持条件：residual router input 或 row projection 比 raw training 更好地保持 final rare margin、降低 sign flips。  
削弱条件：ordinary training 同样保持 rare separation。

## 4. Mathematical Model

**对象：** hidden state $h_i$，common vector $c$，residual hidden state $r_i=h_i-c$，router row $w_e$，feature id $f_i$，route score $z_{i,e}=w_e^\top h_i$，matched expert $m(f)$。

**核心分解：**

$$
h_i = c + r_i,\qquad z_{i,e}=w_e^\top c + w_e^\top r_i.
$$

**机制关系：** 如果 common term 只控制 concentration，那么移除 $c$ 应降低 common bias 或 load imbalance，但不会自动产生 rare feature 一对一映射；除非 residual states 来自 route-relevant population。

**可观察指标：** rare-feature NMI、common/rare binary NMI、joint feature score、rare margin、sign-flip rate、max load、effective experts、slot-start NMI、task loss、target accuracy。

**证伪条件：** common-subtracted all-position routing 在多个 slot lengths 上达到高 rare-feature NMI 和正 rare margins，同时 slot-start NMI 低、task loss 不恶化。

## 5. Computational Realization

**输入对象：** 一个高频 common feature、三个低频 rare features、repeated feature slots、随机 neutral background tokens、feature-specific target tokens。

**计算变量：** route-position hidden states、all-position hidden states、common vectors、k-means centers、oracle feature centroids、router assignments、rare margins、position nuisance metrics、training trajectories。

**算法阶段：**

1. 构造无位置编码的一层 causal Transformer + 一层 weighted top-1 MoE。
2. 训练前提取 calibration hidden states。
3. 用 random rows、all-position centers、route-position centers、oracle feature centers 初始化 router。
4. 在 balanced held-out route-position set 上评估 step-0 rare separation。
5. 用 imbalanced common/rare objective 训练并评估 preservation。
6. 比较 raw routing、common-subtracted routing、residual router input、router-row projection。

**阶段证据：** step-0 rare-feature NMI、final rare-feature NMI、rare margin、sign flips、load、target accuracy、slot-start NMI。

**预期产物：** `protocol.md`、`summary.md`、`detailed.md`、CSV 表、PNG 图、logs、ACP submission record。

## 6. Minimal Falsification Tests

| Test | 问题 | 干预 / 比较 | 主指标 | 通过 / 失败 / 不充分 | 为什么决定 | 失败含义 |
| --- | --- | --- | --- | --- | --- | --- |
| Step-0 common subtraction audit | common subtraction 本身能否分开 rare features？ | random/raw 和 all-position/raw 对比 common-subtracted variants | rare-feature NMI | 若 load 改善但 rare NMI 不改善则支持 H1；若 rare NMI 跨 slot lengths 提升则削弱 H1 | 区分 load repair 和 feature separation | 证伪 common-subtraction-as-separator 的这个实现 |
| Route-relevant center audit | 不均匀下是否需要 route-relevant pool？ | all-position k-means vs route-position k-means vs oracle centroids | rare-feature NMI 和 rare margin | route/oracle centers 胜出支持 H2；all-position common centers 匹配则削弱 H2 | 检查 sample-pool mismatch | 证伪本 synthetic surface 中 route-pool necessity |
| Preservation audit | 有效 rare separation 是否更需要 common control？ | route/oracle init raw training vs residual input/row projection | final rare margin 和 sign-flip rate | controls 更好保持 margin 支持 H3；raw 等同则削弱 H3 | 分离 initialization 与 preservation | 证伪测试过的 preservation controls |

## 7. Current Evidence

**观测：** A06_07 显示 common-centering 降低 load 但几乎不提高 feature NMI；A06_17 addendum 显示 training-time common subtraction 不能救 all-position merge basin；A06_20 显示 routing-aware common estimator 不比 raw all-position 更能恢复 feature。

**A06_24_synthetic 结果：** 4 卡 full run `pt-hb9swzcm` 完成 32 个 seed/slot cells。Step 0 时，all-position common-subtracted centers 的 rare-feature NMI 为 `0.690`、joint feature score 为 `0.405`、rare margin p05 为 `-2.759`；route-position residual centers 和 oracle centers 达到 rare-feature NMI `1.000`、joint score `0.637`、rare margin p05 约 `11.6`。训练后所有条件 target accuracy 都到 `1.0`，但 all-position common-subtracted 仍弱（`joint=0.432`、rare margin p05 `-5.427`），低于 route-position raw（`joint=0.620`、rare margin p05 `5.227`）和 oracle row-projected（`joint=0.636`、rare margin p05 `8.646`）。Slot-start NMI 很低，step-0 最大均值 `0.024`。

**解释：** 旧证据已经削弱 simple common subtraction 作为 feature-separation method；A06_24_synthetic 在训练过的 no-position Transformer-MoE 表面支持同一边界。新的细化是：residual input 保 rare-rare separation 和 margin，但 row projection 更能保持完整 common/rare partition。

**边界：** A06_24_toy 向量审计只能说明 residual-control mechanism 可能存在，不能作为主线证据。

**证据链接：**

- `Projects/from-attention-to-search/main/experiments/A06/A06_07_label_free_common_residual_control_router/summary.md`
- `Projects/from-attention-to-search/main/experiments/A06/A06_17_all_position_route_relevant_feature_discovery/addendum_common_subtraction_rescue/summary.md`
- `Projects/from-attention-to-search/main/experiments/A06/A06_20_route_logit_common_estimator_random_init_feature_recovery/summary.md`
- `Projects/from-attention-to-search/main/experiments/A06/A06_24_toy_common_rare_residual_proxy_synthetic/summary.md`
- `Projects/from-attention-to-search/main/experiments/A06/A06_24_synthetic_common_rare_transformer_moe/summary.md`
- `Projects/from-attention-to-search/main/experiments/A06/A06_24_synthetic_common_rare_transformer_moe/detailed.md`

## 8. Claim Boundary And Next Decision

**可声称：** 在这个无位置编码 Transformer-MoE synthetic surface 中，simple global common subtraction 不是可靠 feature separator。不均匀下 rare-feature expert separation 用 route-relevant centers 或 oracle centers 更干净。Target accuracy 不证明 specialization。若 claim 包含 common-vs-rare 和 rare-rare separation，row projection 是比 residual input 更强的 preservation candidate。

**不可声称：** 真实语言迁移、语义专家、可部署 gating、优化器最优性、或所有 label-free route-relevance 方法不可能。

**下一步决策：** 进入 task-aware route-relevant state selector 或 row-projected margin-preserving update 的方法 anchor；不要把 simple global common subtraction 作为下一轮主方法。
