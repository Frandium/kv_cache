# Theory AI Prompts For Mechanism Audit

## Prompt A: Why Random Top-1 Gating Does Not Give Uniform Feature-To-Expert Partition

你是一名理论研究助手。请审核下面这个机制解释是否成立，并给出可以被实验推翻的数学条件。

研究问题：数据里的 feature 严格均匀出现，expert 数量也等于 feature 数量时，为什么随机初始化的 top-1 gating 仍然不会自动把不同 feature 均匀分到不同 expert？

最小模型：

$$
h_f = c + r_f + \epsilon_f
$$

其中 $h_f$ 是 feature $f$ 的路由位置 hidden state，$c$ 是所有 feature 共享的共同成分，$r_f$ 是 feature 自己的剩余成分，$\epsilon_f$ 是噪声。门控器第 $e$ 个 expert 的分数为：

$$
z_{f,e}=w_e^\top h_f=w_e^\top c+w_e^\top r_f+w_e^\top \epsilon_f
$$

请完成四件事：

1. 用这个模型解释为什么“feature 频率均匀”不推出“expert 分配均匀”。
2. 给出随机 gate 失败的两个可能机制，并区分它们：共同成分优势 $w_e^\top c$ 造成所有 feature 偏向同一 expert；随机超平面和 feature residual center 没有对齐，导致多个 feature 被合并。
3. 推导或描述一个可检查的不等式：什么时候共同成分优势会压过 feature residual 差异，什么时候减去共同成分只能改善但不能保证一对一分配。
4. 给出 3 个可观测预测：初始化时的最大 expert 负载、feature-to-expert NMI、减共同成分前后变化，以及训练早期 top-1 feedback 会如何放大或削弱这种偏差。

请严格区分：已经由模型推出的结论、需要分布假设才能推出的结论、目前实验还不能支持的过强说法。

## Prompt B: Why Centered Route-Position Clustering Works And Why All-Position Clustering Fails

你是一名理论研究助手。请审核下面这个 feature-center discovery 机制是否成立，并给出 claim boundary。

研究问题：为什么对路由位置 hidden states 减去共同成分后聚类，可以恢复 feature center；但把所有位置 hidden states 一起聚类时，会把完整 feature 合并到同一个 expert？

最小模型：

$$
h_{i,p}=c_{\rho(p)}+\mu_{f_i,p}+\eta_{i,p}
$$

其中 $i$ 是样本，$p$ 是序列位置，$\rho(p)$ 是位置角色，$f_i$ 是 feature，$c_{\rho(p)}$ 是角色相关共同成分，$\mu_{f_i,p}$ 是 feature 在该位置的信号，$\eta_{i,p}$ 是噪声。真正做 routing 的位置是 slot 的最后一个 token。

请完成五件事：

1. 说明为什么在只取路由位置、并且 feature 均匀时，减去校准均值后做 k-means 可能恢复 feature center。
2. 给出需要满足的分离条件：feature center 之间的距离、组内噪声、样本数、common/role component 的大小关系。
3. 解释为什么 all-position clustering 会失败：不是 feature 不存在，而是不同位置角色的 hidden geometry 混在同一个聚类目标里，导致聚类中心服务于“整体样本池”，不是服务于“路由位置”。
4. 审核一个危险替代解释：slot 最后一个 token 是否只是 feature-specific token shortcut，而不是 whole-slot compositional feature？请说明这会限制什么结论。
5. 给出下一步实验建议：如何无标签地筛出 route-relevant hidden states，并用什么指标证明筛选器真的接近 route-only clustering。

请输出：机制图景、数学条件、可证伪预测、过强 claim 清单、最小下一步实验。
