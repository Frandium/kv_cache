# A14 NTP Conditional Low-Rank Theory Sync

这是 2026-07-24 同步到 `xyDeng` 分支的理论证明包。它只包含理论、证明附录、
物理先验审计和参考文献；实验协议未包含在本次同步与提交中。

源包：`NTP_conditional_low_rank_theory_package.zip`<br>
源包 SHA-256：`9f16514854e10299d25ebf0358f519a77eef9b4b3a19698a2d4dfd00b5056746`

## 推荐阅读顺序

1. [03_物理先验与假设审计.md](03_物理先验与假设审计.md)：先区分语言先验、
   数学假设、测量守卫和开放桥接条件。
2. [01_理论论文_共享组合结构与条件低有效秩.md](01_理论论文_共享组合结构与条件低有效秩.md)：
   阅读完整故事、定义和 T0–T9 定理链。
3. [02_证明附录.md](02_证明附录.md)：核验集中界、优化器子空间保持、
   Jacobian covariance、effective-rank 上界和两张证明流程图。
4. [references.bib](references.bib)：文献条目。

## 当前结论

在简化一层 NTP Transformer 中，若共享任务映射、概率稀疏调用、独立 context
消噪、受控任务参数更新和共享 ROOT path image 同时成立，则固定读出任务功能秩
在显式误差预算下以高概率受 $sr$ 控制。

这里：

- $s$ 是单输入在头部事件中同时调用的有效映射—路径像块数；
- $r$ 是每个有效块传向 ROOT 的局部任务维数上界；
- “固定读出”表示投影后继续使用训练完成的原输出头；
- “功能秩”属于冻结评估分布，不是单个样本的属性。

## 当前核心边界

一般可训练 softmax attention 中，共享任务映射并不自动蕴含跨 contexts 共享的
ROOT Jacobian/path image。主文 H9 明确保留为开放桥接条件；本文没有把它包装
成由 NTP loss 单独推出的结论。

## 本次审核修正

- 全部 Markdown 展示公式统一为 `$$...$$`，行内公式统一为 `$...$`。
- 将 calibration-only 投影约束写入 fixed-readout functional-rank 的数学定义。
- 明确有限-context 集中定理在 $U=0$ 的同一注册任务条件下成立。
- 明确 T4/T5 的单模式参数秩界不能直接套到任意混合模式共享参数；混合训练须
  使用联合子空间或 H9/误差预算。
- 补充 raw effective-rank 上界的端点条件。
- 在证明附录增加自然语言任务建模图和定理依赖图。

## 明确排除

- 原 zip 中的实验协议；
- 实验实现、运行记录、raw logs、checkpoints 或数据；
- “完整 LLM 参数/hidden state 普遍低秩”以及任何 Router/MoE 结论。
