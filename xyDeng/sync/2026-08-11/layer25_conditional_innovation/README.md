# 第 25 层局部新增可访问性

本目录是 2026-08-11 经研究者审计后的外部阅读副本。

## 阅读顺序

1. [Self-contained 日报](daily_report.md)
2. [A15_08 中文 Anchor 快照](source_records/anchor/anchor_cn.md)
3. [冻结 E04 Protocol](source_records/e04/protocol.md)
4. [eligible E04 Summary](source_records/e04/summary.md)
5. [学长的方差区间报告](source_records/advisor/variance_interval_report.md)

## 当前认识更新

对冻结 Qwen3-8B、受控两跳任务、最终 `Answer:` 前 token 和第 25 层
post-attention 归一化写入，完整剩余更新 $R_U$ 在全新确认数据上增加了旧状态
之外的终端答案线性可访问性，并超过目标独立、预算相同的注册对照。

## 人工审计边界

本次只审计 $X,Z,U,R_U$ 和由 $G_{true}$、$G_{state}$、$T_{cap}$ 支撑的局部
新增可访问性结论。canonical 记录中的广义特征方程、目标条件方向、二维充分性
和秩阶梯尚未人工审计，也不直接回答当前问题；它们不进入本同步包的认识更新
或下一决策。

本结论不能推广为相邻层普遍新增、整个表征低秩、Shannon 信息创造、模型原生
使用、专家效用或 Router 收益。

## 下一决策

把同一度量扩展到全部 36 层，在匹配的一跳/两跳 × 近程/远程任务上比较深层
和浅层的必要性 × 距离交互。

## 包含与排除

包含 reader-friendly 日报、Anchor/Protocol/Summary 快照、学长来源报告及其
图表。排除私有组会记录、完整 detailed、原始数组、bootstrap 明细、日志、
缓存、模型权重、数据集和未审计方向实验的读者侧图表。
