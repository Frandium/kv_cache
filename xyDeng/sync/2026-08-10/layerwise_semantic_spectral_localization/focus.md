# 从谱后移到逐层细语义方向：A15 组会 Focus

```text
type: meeting_knowledge_update
status: AI_DRAFT_AWAITING_HUMAN_CONFIRMATION
scope: one_problem
```

**问题来源：**研究者在 0807 meeting notes 中提出的问题；私人原始记录未纳入本同步包。
**证据范围：**[匹配 taxonomy 十层结果](evidence/a15_02_07/summary.md)、[shared-tail 结果](evidence/a15_04/summary.md)、[固定频带 matched-dispatch 结果](evidence/a15_05_05/summary.md)和候选 [A15_07 Anchor](anchor/15_07_layerwise_conditional_fine_discriminant_directions_anchor_cn.md)。
**排除范围：**Router 实现、专家功能增益、预设 middle/tail 频带，以及把条件细类差直接称为“本层新创造的知识”。

## 1. 组会主题与核心问题

**主题：**从“层越深，细语义越靠后谱”推进到“每层是否存在跨表达稳定的细语义低秩方向”。

**核心问题：**模型深层可能在共同概念上形成更细的表征；现有实验究竟支持哪一部分，又把 layer-wise gating 的下一步收窄到了什么？

**术语：**“本地参数秩”是每层独立按 MLP 参数增益排序的位置，不代表跨层同一向量；“conditional-fine”是在同一父类内去掉父类共同中心后的细类差异；“Haar 对照”是同维随机子空间；“功能准入”是候选方向通过表示审计后，另行证明它能预测专家相对效用，而不是直接训练 Router。

## 2. 当前认识更新

我们现在知道，语义方差会随深度在每层自己的参数秩中重新分配，但这种重分配不是细语义独有，也不能识别跨层同一语义方向。匹配粗细类别数后，没有出现稳定的细语义额外后移；本地后移也没有进入一组跨层共享的 broad tail。

因此，导师关于“逐层细化、用少量方向 gating”的动机仍然值得检验，但操作对象必须改变：不再先指定 head、middle 或 tail，而是在每层内部去掉父类共同部分，寻找能够跨模板、事实和措辞复现的低秩细类判别方向。

即使找到这种方向，它也只是候选语义坐标。只有它进一步预测真实 expert utility，并超过 native score、同秩随机空间和 wrong-layer 对照，才可能进入 layer-wise Router。

## 3. 三个递进问题

### 3.1 深层细语义是否获得独有的后谱定位？

**直接答案：没有建立。**匹配 taxonomy 的十层实验中，复杂减简单的晚层相对后移为 $-0.004690$，95% 区间 $[-0.012182,0.009295]$ 跨零；新增线性信息也接近零。因此，粗细目标共同发生本地秩重分配，但没有可靠的 fine-specific 深度趋势。

![匹配 taxonomy 的十层参数秩谱与层块趋势](evidence/a15_02_07/figures/tax_10layer_overlay_and_block_trends.png)

这只关闭“越细越靠后”的统一定位，不能推出深层没有细语义。下一问必须把“位置”与“方向身份”分开。

### 3.2 能否用 shared tail 或固定 band 代替方向身份？

**直接答案：不能。**Shared-tail 审计中，全局 tail 的晚减早富集变化为 $-0.103453$，而 shared middle 为 $+0.228442$；这说明本地后移没有进入注册的共享 broad tail，但 middle 的正变化仍只是几何现象。随后固定频带虽改变了 4.98%--21.88% 的路由，head、middle、tail 和 middle+tail 的 held-out NLL 都严格差于完整 native Router。

这两项结果共同关闭了“找到某个 band 就可以做 Router”的捷径，却没有关闭逐层低秩语义方向。下一问必须直接检验方向能否跨表达复现。

### 3.3 下一步怎样取得逐层方向身份？

**候选答案：**A15_07 在每层分别构造“细类中心差异大、同类跨表达波动小”的低秩方向；一半表达构造、另一半确认，再双向交换，并与同秩 Haar 空间和父类内标签置乱比较。

```mermaid
flowchart LR
    A["观察：语义谱随深度重分配"] --> B["匹配粗细：没有细语义额外后移"]
    B --> C["shared tail 与固定 band 均未获功能准入"]
    C --> D["改对象：每层独立寻找 conditional-fine 低秩方向"]
    D --> E["跨表达 + Haar + 置乱确认"]
    E --> F["若通过，只开放独立 expert-utility 准入"]
    F --> G["只有预测真实效用后才讨论 layer-wise Router"]
```

A15_07 若通过，只能说明某些层存在候选细语义坐标；若全空间可分但低秩不优于随机，说明信息更可能是分布式的；若留出表达失败，则优先判断表达捷径或样本能力不足。

## 4. Claim Boundary 与唯一下一决策

**当前边界：**可以说 band 位置不足以识别语义方向，且固定 band 尚无 Router 功能收益；不能说深层没有细语义、middle/tail 没有信息、A15_07 方向会被模型自然使用，或不同层功能上必须使用不同方向。

**唯一下一决策：**确认或修改 [A15_07 Anchor](anchor/15_07_layerwise_conditional_fine_discriminant_directions_anchor_cn.md) 的六个设计点，并裁定“投影公共均值方向”是否加入必要对照。
**完成标准：**冻结表征对象、conditional-fine 标签、跨表达拆分、候选秩与收缩范围、逐层留出优势指标、随机对照，以及 Pass 只开放功能准入而不开放 Router 训练的边界。
**恢复动作：**人工确认后，从同一 Anchor 写一份 `DRAFT_NOT_EXECUTABLE` Protocol；Protocol 未经单独批准前，不实现、不 smoke、不运行。
