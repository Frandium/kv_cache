# A11_23b 显式绑定校准 Anchor

## 0. researcher judgment record

原三 token 行序列在更深、更宽、更长训练下仍无法拟合。候选阻塞是关系角色与 key-value 绑定不显式。以带关系角色的原子 mapping token 隔离该阻塞，主指标为五 seed 未见置换 test 准确率。

## 1. Problem Definition

唯一问题：显式 mapping token 是否使两层模型达到 `5/5 >=0.70`？不讨论长上下文或 MTP 收益。

## 2. Physical Prior

先解决局部绑定，才能解释组合失败；若仍失败，阻塞更接近算法归纳偏置。

## 3. Hypothesis

full task 通过，任一单表缺失控制低于 `0.25`。

## 4. Model

$P^{(r)}_{ij}$ 表示关系 $r$ 将 $i$ 映射到 $j$；答案仍为 $Y=\pi_2(\pi_1(A))$。

## 5. Realization

两层 width-64 Transformer，五 seed，3000 steps，全局互斥置换 split。

## 6. Test

要求答案 `>=0.70`、格式 `>=0.95`、两项单表控制 `<0.25`。

## 7. Evidence

五 seed test 为 `0.1274--0.1353`，格式 `1.0`，显式绑定修复被削弱。

## 8. Boundary And Next Decision

不能声称所有模型不能组合。停止调该序列模型，转向 oracle factor 的 rank 机制隔离。
