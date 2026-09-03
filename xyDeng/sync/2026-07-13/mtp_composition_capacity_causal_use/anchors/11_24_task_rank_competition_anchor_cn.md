# A11_24 任务秩竞争 Anchor

## 0. researcher judgment record

序列校准失败使容量问题无法在 Transformer 上解释；先用 oracle 提供独立因素 $u,z$，直接干预共享表示 rank。若 rank 1 下 MTP 挤出标准方向、rank 2 下代价消失，则支持结构性秩竞争。

## 1. Problem Definition

唯一问题：只把共享 rank 从 1 增至 2，是否消除 MTP 标准损失代价并保留未来学习？不外推到 LLM。

## 2. Prior

线性瓶颈中，两个独立因素需要两个独立方向；目标权重决定 rank 不足时谁被保留。

## 3. Hypothesis

rank 1 的 MTP 标准 loss 高于 NTP 且未来 MSE 更低；rank 2 gap 消失。

## 4. Model

$h=W[u,z]$，$L_N=2L_u+L_z$，$L_{MTP}=L_N+2L_A(z)$。

## 5. Realization

Gaussian factors、线性 encoder/readout、ranks 1/2、五 seed、1000 steps。

## 6. Test

通过条件：rank-1 gap 每 seed `>0.5`，rank-2 绝对 gap `<0.05`，rank-1 未来 MSE 在 5/5 改善。

## 7. Evidence

rank-1 gap `0.9084--1.0218`，未来 MSE 5/5 改善；rank-2 最大绝对 gap `1.23e-13`。支持。

## 8. Boundary And Next Decision

只支持该线性模型的 rank 因果，不支持 Transformer 尺度律。下一步测试课程是否能修复 rank-1 frontier。
