# Meeting Brief: A06_E01 2x2 Expert-Geometry Metric Audit

Use this as a short group-meeting reading surface. It is not a replacement for
`summary.md` or `detailed.md`.

## Question

Core question:

在两个 $2\times2$ experts 和二维均匀输入中，哪一种由 expert 初始化矩阵诱导的兼容分数 $q_e(x)$ 可以作为下一步路由器构造的核心几何对象？

Why it matters to the mainline:

如果先不确定 $q_e(x)$，后续路由器的训练、近似和负载控制都没有清楚目标。这个实验先审计“直接用分数最高者分配”的理想静态规则，而不是证明训练一定会学到它。

Current subproblem:

M1 signed prototype: $q_e(x)=x^\top m_e$。  
M2 unsigned prototype: $q_e(x)=(x^\top m_e)^2$。  
M3 top-1 projection: $q_e(x)=\|V_{e,1}^\top x\|^2$。  
M3 full-span control: $q_e(x)=\|V_{e,1:2}^\top x\|^2$。  
M4 matrix response: $q_e(x)=x^\top A_e^\top A_ex=\|A_ex\|^2$。

比较 M1/M2/M3/M4 在最小二维设置下是否能同时支持两个目标：第一，均匀分布的 feature 能被均匀分到两个 experts；第二，分配不是随机凑均匀，而是被选中的 expert 对该 feature 的分数更高，并且分数优势足够清楚。后文把第二点简称为“由分数定义的专家偏好”。

## Conclusion

One-sentence summary:

二维情形下，M4 的“矩阵响应能量”是当前最合适的核心分数定义；M3-top1 是低秩近似候选；M1/M2 可以作为用于比较的对照分发方式。

1. 不是只有 M4 能把均匀 feature 分成均匀负载：M1/M2/M3-top1 在非退化各向异性条件下也能做到负载误差为 0。
2. M4 的优势是它保留完整矩阵响应，在 C3/C5 的分数优势最大，并且在 C2 这个“没有方向差异”的对照组中正确退化。
3. 只看负载是否均匀不能证明 expert 真的有偏好；C4 近似相同 experts 的有效/无效标记过于宽松，必须看分数优势大小。

## Setup

Only include setup details needed to judge the evidence.

Data / task:

二维输入 $x$ 从单位圆 $S^1$ 均匀采样；每个 expert-geometry seed 采样 10,000 个点，共 100 个 seeds。条件包括随机矩阵、等奇异值正交矩阵、不等奇异值正交矩阵、近似相同 experts、明显分离各向异性 experts。

Model / router（模型 / 路由规则）:

两个 $2\times2$ expert matrices $A_1,A_2$。路由规则是理想静态规则：$g(x)=\arg\max_e q_e(x)$，也就是把 $x$ 分给分数最高的 expert；这里没有训练。

Comparison:

M1 signed prototype: $q_e(x)=x^\top m_e$。  
M2 unsigned prototype: $q_e(x)=(x^\top m_e)^2$。  
M3 top-1 projection: $q_e(x)=\|V_{e,1}^\top x\|^2$。  
M3 full-span control: $q_e(x)=\|V_{e,1:2}^\top x\|^2$。  
M4 matrix response: $q_e(x)=x^\top A_e^\top A_ex=\|A_ex\|^2$。

Primary metric:

`metric_validity` 是主判据；审核时必须同时读自然负载误差、平均分数优势、退化比例和符号敏感性。最重要的是平均分数优势，因为它回答“分发是否带有可分辨的 expert 偏好”，而不是只回答“数量是否 50/50”。

Important omission:

本实验不包含可训练路由器、不包含真实 MoE、不包含语义特化、不包含高维公共成分移除，也不证明真实训练会学到 M4 的理想分配规则。

## Key Evidence

1. **M4 是最强的核心分数定义**
   Evidence:
   C3/C5 是非退化各向异性 expert geometry；C2 是等奇异值 isotropic control。
   Result:
   M4 在 C3 的负载误差 = 0.0000、平均分数优势 = 1.5444；在 C5 的负载误差 = 0.0000、平均分数优势 = 2.3873；在 C2 的退化比例 = 1.0000、平均分数优势约为 0。
   Interpretation:
   M4 不只是能均匀分发，它还能在应该有 expert 几何差异时给出强分数优势，并在没有几何差异时正确失败。

2. **其他方法也能分开 feature，但证据更弱**
   Evidence:
   在 C3/C5 中，M1/M2/M3-top1 也能得到负载误差 = 0。
   Result:
   C5 平均分数优势：M1 = 0.9003，M2 = 0.6366，M3-top1 = 0.6366，M4 = 2.3873。M1 的平均符号敏感性 = 0.5333；M2 的平均符号敏感性 = 0.2633。
   Interpretation:
   这些方法不是完全失败；但 M1/M2 依赖奇异向量符号或原型向量选择，M3-top1 丢掉奇异值强度和次方向，所以它们更适合作为对照方法或低维近似，而不是核心矩阵几何。

3. **只看负载均匀不够**
   Evidence:
   C4 近似相同 experts 和 C1 随机矩阵暴露了“只用有效/无效二值标记判断”的边界。
   Result:
   C4 中 M4 被标为有效，但自然分配下的平均分数优势只有 0.0226；C1 中 M4 自然负载误差 = 0.2377，但校准后负载误差 = 0.0000，校准后平均分数优势 = 1.4801。
   Interpretation:
   近似相同 experts 不能因为有效/无效标记通过就说有强专家偏好；随机矩阵也可能需要校准才能满足负载均匀。

## Boundary

This conclusion does not cover:

- 不证明可训练路由器能自动学到 M4。
- 不证明真实 MoE collapse 已解决。
- 不证明语义 feature 特化，也不证明 feature 被分到某个 expert 后一定带来真实任务收益。
- 不证明高维低秩近似和 common removal 已经成立。
- 不证明 `metric_validity` 这种有效/无效二值判据在近似相同 experts 上足够严格。

## Next Step

Next decision:

是否把 M4 的矩阵响应能量作为下一轮路由器构造目标，同时把 M3-top1 当作低维近似候选？

Next minimal experiment or action:

在同一个二维设置中比较三种分配规则：精确 M4 二次分数规则、M3-top1 低维近似规则、最佳线性近似规则。核心看负载误差、平均分数优势、与 M4 理想规则的一致率，以及近似相同 experts 中的误判率。
