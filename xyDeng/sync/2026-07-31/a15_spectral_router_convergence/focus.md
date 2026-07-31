# 从频谱偏置到功能路由：covariance 频带能否成为分发原则？

状态：阶段性认识已收敛；A15_00_02/03 已启动并通过 smoke，尚无正式结果。

## 1. 本日唯一回答的问题

**问题：** 把实际 Router 输入按 covariance 的 head、middle、tail 切分后，我们是否已经证明这些频带不仅改变 Router 的学习与划分，还能指导对专家训练有益的分发？

**为什么重要：** 若答案为是，可以直接据频带设计 Router；若答案为否，就必须先找到跨数据稳定、能预测专家效用的功能变量。

**术语解释：** head、middle、tail 只是输入变化量从大到小的方向区间，不天然表示高低频 token、语义层级或功能重要性；“有益分发”指在相同计算与容量下，把 token 交给更合适且共同训练冲突更小的专家，最终降低留出损失。

## 2. 认识更新

**一句话回答：** 我们已经证明 covariance 各向异性会让线性 Gate 更早学会高方差方向，但尚未证明 covariance rank 本身是有功能价值的分发坐标。

1. 训练后的 Gate 在等能比较下明显偏 head，但 middle/tail 仍可见、仍会影响一部分原生路由，因此不是“线性 Router 只能看 head”。
2. 受控实验已给出因果结论：目标相同且总能量固定时，方差越大的方向达到相同学习进度越快；白化后速度差消失，tail 单独承载目标时仍能学会。
3. 固定 middle/tail 确实产生 native logits 没有保留的新邻域，但同维随机方向也产生大量新邻域；它们没有在 LB 与 decommon 两条谱系中稳定预测共同训练兼容性。
4. 真实 Router--Expert 联合形成过程仍未回答：三条真实训练在得到有效频谱形成判断前发生近单专家负载集中；浅层 head 指导深层的 Pilot 又因 head 与随机子空间同时满分而失去分辨力。
5. 因此新的候选内核不是“继续删 head 或指定 M/T 路由”，而是：**用跨数据稳定的浅层条件确定粗分工，再让后层只处理该条件内尚未解决的功能差异；频谱只作候选载体，必须由专家相对效用与留出损失认证。**

## 3. 决定性证据表

| 要回答的层级 | 最少直接证据 | 阶段性答案 |
| --- | --- | --- |
| Gate 能否访问不同频带？ | 等能 head:middle 增益为 4.03--6.36 倍，head:tail 为 14.61--25.36 倍；middle/tail 作用非零 | 能访问，但明显偏 head |
| 方差是否因果改变学习速度？ | flat 时 H/M/T 约 1:1:1；方差 4:2:1 时学习时间约 1:2:4；16:4:1 时约 1:4:16；白化后恢复 1:1:1 | **已回答：会改变有限时间学习速度** |
| 新频带划分是否天然有功能？ | 真实频带改变 73%--90% 邻居，随机同维也改变 71%--88%；M/T/M+T 无候选跨两谱系通过兼容性门 | **已回答：不同划分不等于有益划分** |
| 真实形成与浅层指导是否成立？ | 真实训练先触发约 0.99 最大专家份额；浅层 Pilot 中 head probe 与 random q95 都为 1.0 | **部分回答：当前操作化不足，不能下功能结论** |

**读表结论：** 频谱已经是可解释的优化偏置，却还不是经过验证的 MoE 功能坐标。

**完整证据：** [A15 综合稿](main/stories/15_linear_gate_spectral_training_bias/01_linear_gate_spectral_access_and_learning_dynamics_cn.md)；[受控学习速度结果](main/experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_S_controlled_spectral_learning_dynamics/summary.md)；[真实轨迹边界](main/experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_R_real_early_spectral_learning_dynamics/summary.md)；[功能准入结果](main/experiments/A15/15_02_middle_tail_functional_resolution/A15_02_01_E01_cross_update_compatibility_gate/summary_cn.md)。

## 4. 解释流程图（AI 初稿，研究者修改）

```mermaid
flowchart LR
    O["观察：训练 Gate 等能后仍明显偏 head"] --> C["受控干预：只改变 covariance 方差"]
    C --> S["结果：高方差方向更早学会，白化后顺序消失"]
    S --> G["因此频谱是学习速度偏置，而不是自动生成的功能目标"]
    G --> F["功能检验：M/T 有新划分，但不过随机与跨谱系兼容性门"]
    F --> K["认识更新：固定 covariance rank 不能直接充当有益分发原则"]
    K --> N["候选新内核：稳定浅层条件 + 条件内功能细分 + 专家效用裁定"]
```

## 5. 边界与下一步

- **成立范围：** 线性 Gate、已注册的受控 SGD 系统、两条 12 层 DCLM 检查点谱系，以及固定频带的一步局部兼容性测试。
- **仍不能说：** head 语义上更重要；middle/tail 没有非线性或长期价值；最深层语义一定进入 tail；浅层 common 指导深层一定改善训练。
- **最强未决解释：** decommon 可能只去掉了均值，而去均值后仍存在跨文档稳定的 centered-common，Gate 继续优先学习它；也可能所谓 pooled common 只是估计或能量效应。
- **唯一下一决策：** 完成并联合裁定 A15_00_02 与 A15_00_03，判断“去均值后仍有稳定 common，且 Gate 更依赖它而非 local residual”是否成立。
- **完成判据：** 两份批准 Protocol 在全部注册 checkpoint、层、随机方向与 held-out 文档上形成正式 Pass/Fail/Insufficient；只有联合 Pass 才允许把“稳定浅层条件”升级为下一 Router 设计先验。
- **恢复时第一动作：** 读取两份结果摘要；若联合 Pass，起草“稳定浅层条件指导后层功能分发”的新 Anchor，否则关闭 residual-instability 解释并回到直接专家效用监督。
