# 0731 Spectral Router Group-Meeting Convergence Package

本目录是 A15 频谱 Router 研究线的组会/同步最小包。它回答：

> covariance 的 head、middle、tail 是否已经从描述 Router 偏置的坐标，升级为
> 能指导有益专家分工的功能原则？

## 阶段性结论

当前答案是：**尚未。**

- 已建立：在匹配功能目标的受控线性 Gate 中，输入方差越大，相应方向达到同等
  学习进度越快；白化会消除该速度顺序，tail 单独承载目标时仍可学会。
- 已建立：训练 Gate 等能后明显偏 head，但 middle/tail 的访问和原生路由作用
  非零。
- 已建立：固定 middle/tail 产生额外几何划分，却没有在两条谱系中稳定通过一步
  共同训练兼容性门，因此没有获得匹配联合训练资格。
- 尚未建立：真实 Router--Expert 共同形成路径、频带的语义层级、浅层信息指导
  深层的训练收益，以及匹配 FLOP 下的验证损失改善。
- 已启动、尚无正式结果：去均值后是否仍有跨文档稳定的 centered-common，以及
  Gate 是否更依赖它而不是 shard-local residual。

## 推荐阅读顺序

1. [0731 Focus：阶段性认识与唯一下一决策](focus.md)
2. [A15 中文综合稿](main/stories/15_linear_gate_spectral_training_bias/01_linear_gate_spectral_access_and_learning_dynamics_cn.md)
3. [线性 Gate 频谱学习动力学理论主文](daily_research_reports/0731/router_spectral_learning_dynamics_theory_package/01_理论论文_线性MoE_Router的频谱学习动力学.md)
4. [A15 实验索引](main/experiments/A15/README.md)
5. [实际 Router 输入频带响应](main/experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E01_actual_router_input_band_response/summary_cn.md)
6. [早期 head alignment 轨迹](main/experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E02_early_head_alignment_onset/summary_cn.md)
7. [受控 covariance 学习速度因果结果](main/experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_S_controlled_spectral_learning_dynamics/summary.md)
8. [真实 DCLM 轨迹的负载有效性边界](main/experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_R_real_early_spectral_learning_dynamics/summary.md)
9. [浅层 head 指导深层 Pilot](main/experiments/A15/15_01_shallow_head_guided_deep_routing/A15_01_01_E01_controlled_four_layer_shallow_head_pilot/summary.md)
10. [Middle/tail 功能准入结果](main/experiments/A15/15_02_middle_tail_functional_resolution/A15_02_01_E01_cross_update_compatibility_gate/summary_cn.md)

## 已启动、尚无正式结果的协议

- [跨文档 centered-common 稳定性 Protocol](main/experiments/A15/15_00_covariance_head_gate_alignment/A15_00_02_E01_centered_common_subspace_stability/protocol_cn.md)
- [Gate 对 pooled common 与 local residual 的偏好 Protocol](main/experiments/A15/15_00_covariance_head_gate_alignment/A15_00_03_E01_gate_transferable_vs_local_residual_alignment/protocol_cn.md)

这两份实验已完成实现前护栏和 smoke，但尚未形成正式科学裁定；包内 Protocol
只证明问题、指标和裁定合同已经冻结，**不构成实验结果**。只有两项均形成正式
结果且联合 Pass，才允许把“稳定浅层条件”升级为下一 Router 设计先验。

## 包内结构

```text
focus.md
daily_research_reports/0731/
  router_spectral_learning_dynamics_theory_package/
main/
  problem_anchors/15_linear_gate_spectral_training_bias/
  stories/15_linear_gate_spectral_training_bias/
  experiments/A15/
```

完整实验目录保留了协议、摘要、证据账本、关键图与小型表格。Focus、Story、
Anchor、Protocol、Summary、理论主文及其关键图表可在包内直接阅读。`detailed.md`
中指向 worker 代码、原始 run 目录或日志的链接只保留为 provenance 指针；这些对象
因同步边界而刻意不打包。

包内不含 研究者判断记录、原始会议录音、私有讨论问题、数据集、activation
cache、checkpoint、运行日志或集群作业产物。Anchor 中指向 研究者判断记录 的本地
来源字段也已从外部同步副本移除；项目主仓中的原始记录未改变。`MANIFEST.sha256`
用于核对同步前后的文件完整性。

## 同步边界

本目录是本地准备完成的交接包；尚未替研究者 commit、push 或同步到外部仓库。
同步时应复制整个目录，避免破坏包内相对链接。
