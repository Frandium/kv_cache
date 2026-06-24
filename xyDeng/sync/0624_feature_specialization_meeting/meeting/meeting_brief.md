# 均匀特征如何进入稳定专家分工

## 执行摘要

本次要请导师裁定的问题是：**均匀出现的特征，能否稳定、均匀、并且有功能价值地分到不同专家上？**

这里的“特征”指数据中被我们操作性定义的一类样本结构；“专家”指稀疏专家模型中被门控器选择的子网络；“稳定均匀分发”不只要求负载均匀，还要求同一特征进入同一专家、不同特征不被合并、训练后还能保持。

当前判断：**特征级分区是可达的，但不是随机门控自然产生的；它需要先找到路由位置的 feature center，并且必须防止训练早期把这个分区覆盖掉。**

最小机制解释是：hidden state 可写成共同成分加特征剩余成分，随机门控先看到共同成分造成的隐式 expert 偏置；减去共同成分后，在干净的路由位置上聚类可以找回 feature center；但一旦把非路由位置或背景干扰混进聚类对象，初始化就会偏。

主边界：这还不是“真实语言语义专家已经形成”的结论，也不是可部署方法。真实 DCLM 上，proxy feature 可以被发现并在第 0 步线性路由，但普通训练会在第 5/10 步覆盖这个分区。

执行动作：建议下一步优先做**真实文本早期保存 / 反反馈**判定卡；同时把两个未闭合机制交给理论 AI 审核，分别是“随机门控为什么会失败”和“centered route-position clustering 为什么会成功但 all-position clustering 会失败”。理论审核 prompt 见 [`theory_ai_prompts.md`](theory_ai_prompts.md)。

## 共享机制与最小模型

路由位置的 hidden state 记为：

$$
h_f = c + r_f + \epsilon_f
$$

其中 $c$ 是不同特征共享的共同成分，$r_f$ 是 feature $f$ 的特征剩余成分，$\epsilon_f$ 是噪声。门控器给 expert $e$ 的分数是：

$$
z_{f,e}=w_e^\top h_f=w_e^\top c+w_e^\top r_f+w_e^\top \epsilon_f
$$

这说明一个关键点：**feature 均匀出现，不等于 expert 会均匀使用。** 因为 $w_e^\top c$ 对所有 feature 都共同存在，它会让某些 expert 在所有样本上都有共同优势。top-1 门控只选最高分 expert，这个早期优势还会被训练反馈放大。

因此当前故事线分成三层：

1. 随机门控失败：共同成分和随机超平面不对齐会造成 feature merge 或单 expert collapse。
2. 聚类初始化可行：如果只看真正参与路由的 hidden states，减去共同成分后聚类能恢复 feature center。
3. 训练保存仍是瓶颈：真实 DCLM 的第 0 步 proxy routing 可达，但普通训练很快覆盖它。

两个机制还需要理论审核：

- 理论审核 A：随机 top-1 门控为什么不能从均匀 feature 自动得到均匀 expert 分区。见 [`theory_ai_prompts.md#prompt-a-why-random-top-1-gating-does-not-give-uniform-feature-to-expert-partition`](theory_ai_prompts.md#prompt-a-why-random-top-1-gating-does-not-give-uniform-feature-to-expert-partition)。
- 理论审核 B：为什么 centered route-position clustering 可恢复 feature center，而 all-position clustering 会失败。见 [`theory_ai_prompts.md#prompt-b-why-centered-route-position-clustering-works-and-why-all-position-clustering-fails`](theory_ai_prompts.md#prompt-b-why-centered-route-position-clustering-works-and-why-all-position-clustering-fails)。

## 合并主结果

**目前最稳妥的结论是：feature-level expert partition 可以被构造出来，但失败点已经从“feature 是否存在”推进到“如何选择正确 hidden-state 样本池”和“训练早期如何保存分区”。**

### 1. 减去共同成分后的路由位置聚类可以证明特征分区是可达的。

机制对应：如果 $h_f=c+r_f+\epsilon_f$，那么聚类应主要作用在 $r_f$ 上，而不是被 $c$ 支配。

证据：A06_08 中，route-position residual k-means 和 spherical k-means 在合成四特征设置中达到 `feature_NMI=1.0`、load $L=0$；A06_09 中，A06_08 的 pseudo-center 初始化训练到 1600 步仍保持 `feature_NMI=1.0`、accuracy `1.0`。A06_16 修正 positional embedding 后，C0-C3 在 route-position discovery 和 same-model preservation 上全部通过。

解释：这说明 hidden state 里确实存在可用于专家分区的 feature geometry；失败不是因为 feature 不在表征里。

边界：这些结果主要在受控 synthetic / synthetic-to-realistic bridge 上成立，不能推出真实语义 feature，也不能推出真实 DCLM 训练能保存。

来源：`A06_08_label_free_feature_discovery_initialization/summary.md`，`A06_09_training_basin_preservation/summary.md`，`A06_16_synthetic_to_realistic_proxy_bridge/summary.md`。

### 2. 随机门控和全位置聚类都会把问题推回到错误的几何对象。

机制对应：随机门控直接看 raw hidden state，容易被 $w_e^\top c$ 的共同偏置影响；全位置聚类把路由位置和非路由位置混在一起，优化的是混合样本池，而不是路由位置 feature center。

证据：A05_04_02 在 toy dot-product setting 中显示，common logit 早于 collapse，能预测优势 expert；common-logit cancellation 把 baseline final slot NMI 从 `0.080` 提升到 `0.896/0.963`，accuracy 保持 `1.0`。A05_04_03 在真实 DCLM 上削弱了“step 0 完全由 common domination 决定”的强版本，但仍显示 centering 能降低 max load，且 step 10 common channel 快速放大。A06_17 显示 route-only 和 slot 最后位置聚类 `feature_NMI=1.0`，但 all-position 聚类均值只有 `0.797`，且经常把完整 feature 合并到同一 expert。

解释：随机门控失败和 all-position clustering 失败不是同一个技术细节，但它们共同说明：门控初始化必须对准正确的几何对象，不能只依赖 feature 频率均匀或全局隐藏状态聚类。

边界：common cancellation 是机制证据，不是最终方法；A06_17 只证明 controlled bridge 中全位置聚类对象不可靠，还没有给出真实语料上的无标签 route-relevant state selector。

来源：`A05_04_02_round2_dotproduct_common_logit/summary.md`，`A05_04_03_real_text_common_logit_audit/summary.md`，`A06_17_all_position_route_relevant_feature_discovery/summary.md`。

### 3. 专家分区必须通过训练保存和功能价值两道门槛。

机制对应：即使第 0 步分区存在，top-1 训练反馈也可能覆盖它；即使分区被保存，也还要证明被选 expert 对对应样本真的有用。

证据：A06_10 在真实 DCLM hidden state 中找到稳定 proxy clusters；A06_11 证明 raw-center equal-norm 可以把 proxy center 转成第 0 步线性路由；A06_12/A06_13 显示普通 DCLM 训练把 raw-center proxy routing 从 step-0 NMI `0.7549` 快速降到 step-5 `0.0410`、step-10 `0.0131`，而 loss 与 random 接近。A07_01 到 A07_03 在 controlled D07 上进一步显示：共同成分控制不仅改善负载，还能降低 rare loss，并且 route assignment 绑定 expert utility。

解释：05/06 解决“分区为什么坏、如何可达、在哪里坏”；07 解决“这个分区为什么值得保存”。

边界：A07 是 controlled synthetic analytic audit，不是真实 checkpoint，不证明真实语义专家或真实 DCLM utility。

来源：`A06_10_real_dclm_proxy_feature_operationalization/summary.md`，`A06_11_real_dclm_proxy_center_router_initialization/summary.md`，`A06_12_real_dclm_proxy_init_training_preservation/summary.md`，`A06_13_real_dclm_proxy_init_failure_decomposition/summary.md`，`A07_01/02/03` summaries。

## Source Result Modules

| 模块 | 它回答的问题 | 更新的认知 | 主要边界 |
| --- | --- | --- | --- |
| 05 failure mechanism | 为什么均匀 feature 下随机 top-1 仍会 collapse | common-logit / common-bias 和早期 feedback 是核心风险 | toy 结论不能直接外推真实 LM；真实 DCLM step-0 强版本被削弱 |
| 06 geometry proxy preservation | feature center 能否被发现、初始化、保存 | 受控路由位置可发现可保存；真实 DCLM 第 0 步可路由但训练保存失败 | proxy 不等于语义；all-position selector 未解决 |
| 07 features overlap | 分区是否有功能价值 | controlled D07 中 common-control 降低 rare interference，route assignment 绑定 expert utility | 只限 synthetic D07，不是真实 checkpoint |

## 剩余冲突与竞争解释

1. 随机门控失败可能来自共同成分优势，也可能来自随机超平面与 feature centers 不对齐。两者都会导致 feature merge，但可通过 common-cancel 前后的 step-0 NMI、max load、active experts 区分。
2. Centered clustering 成功可能依赖“已知路由位置”。如果不知道哪些 hidden states 是路由相关状态，all-position 聚类会失败；因此 route-relevant state selection 是一个独立问题。
3. Slot 最后 token 的成功可能包含 token shortcut，不一定证明 whole-slot compositional feature。这个边界要在理论审核 B 和后续 whole-slot/real-text 设计中保留。
4. 真实 DCLM 的失败点目前是 early training override，但还没拆清楚是 gate update、hidden-state drift、expert-output feedback，还是 optimizer geometry 主导。

## 执行动作

主动作：建立下一张判定卡，测试真实文本中能否保存第 0 步 proxy partition。

目的：判断 “proxy center 已经可线性路由” 是否能穿过 step 5/10 的训练覆盖窗口。

最小做法：在同一 DCLM proxy surface 上，对比普通训练、router freeze / delayed router update、低 router learning rate、common/load anti-collapse regularizer、proxy-preservation auxiliary loss。主指标是固定 step-0 proxy labels 下的 step-5/10 `proxy_route_NMI`，约束指标是 LM loss 不显著变坏。

完成标准：如果任一保存机制让 step-10 proxy-route NMI 明显高于 raw-center 普通训练，并且 LM loss 接近 random，则进入真实 checkpoint utility audit；如果全部失败，则回到 failure decomposition，把 gate update、hidden drift、expert feedback 分开冻结。

辅助动作：把 [`theory_ai_prompts.md`](theory_ai_prompts.md) 中两个 prompt 交给理论 AI。理论回复只作为机制审核材料，不能直接当作实验证据；它应该输出可证伪条件和下一步实验预测。

## Evidence Index And Source Index

| 证据 | 文件 |
| --- | --- |
| common-logit collapse and cancellation | `Projects/from-attention-to-search/main/experiments/A05_04_02_round2_dotproduct_common_logit/summary.md` |
| real-text common-logit audit | `Projects/from-attention-to-search/main/experiments/A05_04_03_real_text_common_logit_audit/summary.md` |
| label-free route-position discovery | `Projects/from-attention-to-search/main/experiments/A06_08_label_free_feature_discovery_initialization/summary.md` |
| synthetic training basin preservation | `Projects/from-attention-to-search/main/experiments/A06_09_training_basin_preservation/summary.md` |
| real DCLM proxy discovery | `Projects/from-attention-to-search/main/experiments/A06_10_real_dclm_proxy_feature_operationalization/summary.md` |
| real DCLM proxy router bridge | `Projects/from-attention-to-search/main/experiments/A06_11_real_dclm_proxy_center_router_initialization/summary.md` |
| real DCLM preservation failure | `Projects/from-attention-to-search/main/experiments/A06_12_real_dclm_proxy_init_training_preservation/summary.md` |
| first failed stage decomposition | `Projects/from-attention-to-search/main/experiments/A06_13_real_dclm_proxy_init_failure_decomposition/summary.md` |
| synthetic-to-realistic bridge | `Projects/from-attention-to-search/main/experiments/A06_16_synthetic_to_realistic_proxy_bridge/summary.md` |
| all-position route-relevant audit | `Projects/from-attention-to-search/main/experiments/A06_17_all_position_route_relevant_feature_discovery/summary.md` |
| D07 metric and utility chain | `Projects/from-attention-to-search/main/experiments/A07_01_common_rare_conflict_metric_audit/summary.md`; `A07_02_common_controlled_rare_interference/summary.md`; `A07_03_route_function_binding/summary.md` |
| theory audit prompts | `daily_research_reports/0624/meetings/theory_ai_prompts.md` |

## Excluded Material And Reason

| 未纳入主证据的材料 | 原因 |
| --- | --- |
| A06_14/A06_15 | 已暂时 parked，当前组会问题回到机制审计和保存瓶颈 |
| 调试 heatmap 与未整理本地 replay | 没有独立 summary / detailed 记录前，只能作为内部诊断，不能作为 meeting 主证据 |
| 真实语义 feature claim | 当前 proxy clusters 不是语义标注，不能外推 |
| 可部署 gating 方法 claim | 目前只是机制定位和受控初始化，不是完整方法 |

## 汇报最后一句

**我建议下一轮主锚点不是继续证明 feature 存在，而是测试真实文本训练早期能否保存已经存在的 proxy partition；同时让理论 AI 审核随机门控失败和 centered clustering 成功/失败的数学条件，确保下一步实验不是只在调参，而是在裁定机制。**
