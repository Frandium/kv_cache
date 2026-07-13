# Protocol：A11_25 课程秩边界

复用 A11_24 模型、数据、优化器和 seeds。静态辅助权重为 2；课程前 250 steps 为 0、后 750 steps 为 $8/3$，保证累计辅助剂量相同。主判定是 rank-1 Pareto 改善：标准 loss 在 5/5 seed 降低 `>0.1`，auxiliary MSE 增量不超过 `0.02`。rank 2 是容量充足守卫。不能外推到语言模型课程训练。
