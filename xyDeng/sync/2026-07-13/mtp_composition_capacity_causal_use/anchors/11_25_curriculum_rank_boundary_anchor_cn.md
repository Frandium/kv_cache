# A11_25 课程学习秩边界 Anchor

## 0. researcher judgment record

A11_24 已建立结构性 rank-1 tradeoff。课程可改变优化顺序，但不能创造缺失维度。必须在等辅助剂量下判断是否同时改善标准目标且不损害未来目标。

## 1. Problem Definition

唯一问题：等剂量 NTP→MTP 课程是否 Pareto 改善静态 MTP 的 rank-1 结果？

## 2. Prior

训练顺序不能突破信息瓶颈；容量足够时才可能出现纯优化路径收益。

## 3. Hypothesis

rank 1 不通过 Pareto，rank 2 两种 schedule 均接近零损失。

## 4. Model

复用 A11_24。静态 $\lambda=2$；课程前 250 steps 为 0、后 750 steps 为 $8/3$，累计辅助剂量相同。

## 5. Computational Realization

完全复用 A11_24 的 Gaussian factors、线性 encoder/readout、优化器、五 seed 和 1000-step budget，只改变辅助权重 schedule。

## 6. Minimal Falsification Test

课程需在 5/5 seed 将标准 loss 降低 `>0.1`，且 auxiliary MSE 增量 `<=0.02`。

## 7. Current Evidence

Pareto pass `0/5`；标准收益 `-0.00213` 到 `+0.000134`；rank 2 全部近零。

## 8. Claim Boundary And Next Decision

课程不能修复结构 rank 不足，但不否定神经非凸/有限预算收益。课程现阶段 parked，等待容量充足但存在优化冲突的 neural bridge。
