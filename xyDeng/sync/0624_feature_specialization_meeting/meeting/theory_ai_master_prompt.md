# Master Prompt For Theory AI

请你作为理论机制审核助手，阅读我随后粘贴的 `meeting_brief` 和两个 `theory prompts`。你的任务不是重写文档，而是帮助我判断：我的研究主线是否清楚、机制解释是否站得住、下一步 anchor 应该怎么定。

请只使用我粘贴的材料。不要编造新的实验结果；如果你使用额外假设，请明确标注为“额外理论假设”。

## 我希望你重点审核的问题

父问题：

```text
均匀出现的 features，能否稳定、均匀、并且有功能价值地分到不同 experts 上？
```

当前候选判断：

```text
feature-level expert partition 可以被构造出来，
但不会由 random gating 自然产生；
当前瓶颈从 feature 是否存在，推进到 hidden-state population selection 和 early-training preservation。
```

最小模型：

$$
h_f = c + r_f + \epsilon_f
$$

$$
z_{f,e}=w_e^\top h_f=w_e^\top c+w_e^\top r_f+w_e^\top \epsilon_f
$$

其中 $c$ 是共同成分，$r_f$ 是 feature residual，$\epsilon_f$ 是噪声。

## 请严格按以下结构回复

### 1. 一句话 Verdict

请直接判断：这条研究主线目前是 coherent / partially coherent / not coherent。给一句理由。

### 2. 我现在应该相信什么

请列出 3 到 5 条“可以相信”的结论。每条都要分成：

- 结论；
- 支持它的证据来自哪个实验；
- 它不能推出什么。

### 3. 我现在不应该相信什么

请列出 3 到 5 条过强 claim。重点检查：

- 是否把 proxy feature 当成 semantic feature；
- 是否把 load balance 当成 specialization；
- 是否把 synthetic result 外推到 real DCLM；
- 是否把 common subtraction 当成最终方法；
- 是否把 all-position clustering 当成可用 discovery method。

### 4. 机制 A 审核：random gating 为什么失败

请审核以下解释是否合理：

```text
random top-1 gating 失败，不是因为 feature 不均匀，
而是因为 router score 里有 w_e^T c 的共同成分优势，
以及 random hyperplanes 未必和 feature residual centers 对齐。
```

请输出：

- 这个解释需要哪些数学条件；
- 一个可检查的不等式或 margin condition；
- 哪些实验现象支持共同成分解释；
- 哪些实验现象支持随机超平面不对齐解释；
- 哪个实验可以区分这两个解释；
- 如果这个解释错了，我们会看到什么。

### 5. 机制 B 审核：centered route-position clustering 为什么成功，all-position clustering 为什么失败

请审核以下解释是否合理：

```text
route-position residual hidden states 包含 feature centers；
all-position clustering 失败是因为把不同 position roles 的 hidden geometry 混进一个聚类目标，
不是因为 route-position feature geometry 不存在。
```

请输出：

- k-means 成功需要哪些 separation / noise / sample balance 条件；
- 为什么 role-balanced all-position 仍可能失败；
- slot 最后 token 是否可能只是 token shortcut；
- 这个 shortcut 会限制哪些 claim；
- 下一步如何无标签筛出 route-relevant hidden states。

### 6. 下一步 anchor 应该是什么

请在以下三个方向中判断优先级，并给出理由：

1. real-text early preservation / anti-feedback；
2. label-free route-relevant hidden-state selector；
3. controlled D07 utility 向 neural / real checkpoint 迁移。

请给出你推荐的下一张 anchor：

- decision question；
- physical prior；
- falsifiable hypothesis；
- primary metric；
- success / failure / insufficient evidence；
- claim boundary。

### 7. 我应该怎么整理自己的 mind

请给我一个 5 行以内的 mental map，形式如下：

```text
父问题：
已经知道：
还不知道：
最大风险：
下一步裁定：
```

### 8. 给导师汇报时最该说和最不该说的话

请给出：

- 3 句应该说的话；
- 3 句不应该说的话；
- 1 句最适合作为汇报最后一句的话。

请保持回答简洁、可执行、不要泛泛鼓励。
