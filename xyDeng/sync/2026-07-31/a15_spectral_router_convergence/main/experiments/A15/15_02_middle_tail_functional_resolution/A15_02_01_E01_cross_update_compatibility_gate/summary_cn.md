---
experiment_id: A15_02_01_E01_cross_update_compatibility_gate
status: completed_fail
verdict: fail
completed: 2026-07-30
protocol: protocol_cn.md
primary_anchor: 15_02_01_cross_update_compatibility_gate
---

# A15_02_01_E01 结果：non-head 有额外划分，但未通过功能准入

## 结论

**直接结论：** middle、long-tail 和二者联合都能在 native Router 分数之外产生
不同的 token 邻域，但在注册的一步共同训练测试中，没有一个频带能在 LB 与
decommon 两条谱系上稳定提供额外功能预测，并同时超过同维随机方向与错误层
基底。因此本实验裁定为 **Fail**，没有频带获得 8×5090 联合训练资格。

更直白地说：

> non-head 确实提供“不同的分法”，但现有证据没有表明这种分法更能把适合共同
> 训练的 token 放到一起。

这里的失败不是测量失败。两条谱系、三层、Fit/Validation 共 3,072 对 A/B
token groups 全部完成；每次一步更新都降低本组 loss，专家参数均精确恢复，主
步长与半步长的兼容性排序相关为 0.87--1.00。功能 target 与专家梯度余弦的相关
为 0.71--0.96，说明 target 有可测动态范围。

## 如何理解两个指标

**残差邻域新颖度**先去掉 native logits、margin、专家、负载、难度、范数等
线性可解释部分，再问某个频带找到的近邻有多少不同。数值高只说明“重新分组”，
不说明分组有益。

**兼容性增量 $\Delta R^2$**比较：只用 native controls 预测一步交叉更新收益，
与再加入频带的两个 pair features 后相比，在未见文档上多解释多少。它是本实验
的准入指标；正值还必须超过同维随机 q95 和错误层，才说明 covariance rank
位置具有特殊功能关系。

## 关键结果

在预注册层 1/6/12 上，真实 M/T/N 的残差邻域新颖度为 73.2%--90.2%；但一个
固定同维随机子空间也达到 71.4%--87.7%。因此额外几何划分真实存在，但其中很
大一部分是高维子空间普遍会产生的新划分。

Validation 上的模型级三层中位 $\Delta R^2$ 如下，数值单位为 $10^{-4}$：

| 频带 | LB 真频带 | LB 随机 q95 | decommon 真频带 | decommon 随机 q95 | 准入 |
| --- | ---: | ---: | ---: | ---: | --- |
| Middle | -0.735 | +0.725 | -0.430 | +0.536 | Fail |
| Long-tail | +2.237 | -0.233 | -0.520 | +0.244 | Fail |
| Middle + long-tail | -0.590 | -0.554 | -0.429 | -0.198 | Fail |

Long-tail 在 LB 中通过点估计方向门，但在 decommon 中为负且不超过随机；其余
候选在两条谱系的模型级中位数均不为正。因而没有允许锁定为 $S^*$ 的候选。

![静态新颖度与功能准入](figures/static_vs_functional_gate.png)

完整逐层数值见
[validation_functional_cells.csv](tables/validation_functional_cells.csv)，模型级
准入见 [validation_candidate_gate.csv](tables/validation_candidate_gate.csv)，
运行护栏见 [measurement_guards.csv](tables/measurement_guards.csv)。

## Verdict 与停止规则

- **Q2-A 静态分辨率：有。** M/T/N 都能产生 native logits 之外的新邻域；但
  随机方向也很高，因此不能称为 covariance rank 特异的功能结构。
- **Q2-B 局部功能准入：Fail。** 没有候选在两条谱系同时满足正增量、随机 q95
  和错误层门。
- **Final/40k/4-layer transfer：未运行。** Protocol 规定 Validation 无候选即
  停止；这不是遗漏。
- **Q2-C 8×5090 联合训练：未提交。** 条件性执行授权没有生效，训练当前没有
  合法 treatment。

## 结论边界

本结果成立于：两个 12-layer H768、8-expert、top-1 DCLM checkpoint 的 80k
状态；layers 1/6/12；固定 native routes；每组 32 个 token；float32 局部 loss
probe；以及预注册的 cosine/平方距离两个 band features 与低容量 ridge。

它不能证明：

- middle/tail 没有任何非线性、语义或长期训练价值；
- 一步固定路由兼容性等于真实 AdamW 联合动力学；
- 所有模型、层、数据或 Router 都会失败；
- Q1 的 head alignment 是有益或有害。

它能够否定的更窄命题是：

> 在当前定义下，固定 covariance-rank 的 M/T/N pair geometry 没有获得足够稳定、
> 超过随机与错误层的局部功能证书，因此不应直接投入匹配联合训练。

## 下一决策

当前唯一下一决策是：**是否关闭“固定 covariance M/T/N 直接作为 dispatch
坐标”的训练路线，转而先定义由专家梯度或交叉更新残差直接监督的功能子空间？**

在新的功能对象通过独立兼容性准入之前，不恢复 A15_02_E02。

完整执行与证据账本见 [detailed.md](detailed.md)。
