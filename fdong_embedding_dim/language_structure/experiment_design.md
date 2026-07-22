# 第一轮真实语料层次 Pattern 实验

## 要检验的命题

真实语言中的稳定局部 token pair 会以非随机方式组合成稳定的 pair-of-pairs；同一个局部 pair 会被多个不同的二层 pattern 复用。

第一轮只检验：

```text
token
  → C-4-2
  → C-8-(C-4-2, C-4-2)
```

不在第一轮声称已经证明任意深度的语义树或“小 fan-in”。

## 输入

- 语料：`/Users/bytedance/Desktop/dclm/part-*.txt`
- 文档：每行一个 JSON 编码的字符串
- tokenizer：`fdong/Qwen3-0.6B`
- 特殊 token：不添加
- 单文档最大长度：1,024 tokens
- 抽样：固定随机种子打乱 shard；每个 shard 最多取固定数量文档
- train/validation：根据 `relative_path:line_number` 的稳定哈希按 80%/20% 划分

## 推荐规模

```text
正确性测试：人工 token 序列
速度测试：100 documents
第一轮：10,000 documents，约 8M–10M tokens
第二轮：100,000 documents，仅在第一轮候选量和速度可控后运行
```

## C-4-2

### 定义

对文档 token 序列 $x_0,\ldots,x_{T-1}$，枚举：

$$
(x_i,x_{i+d},d),\qquad d\in\{1,2,3\}.
$$

该 occurrence 的最小包围 span 为 $d+1\leq4$。Pattern identity 包含左右 token 和距离 $d$。

### 统计量

- train support
- validation support
- train NPMI
- train/validation 每百万 token rate 的 log difference
- document coverage

### 保留条件

10,000-document 默认值：

```text
train support >= 100
validation support >= 20
NPMI >= 0.10
train/validation rate 相差不超过 2 倍
每个 support 量级最多保留 10,000 个
```

阈值是候选控制参数。正式结论必须做阈值扫描。

## C-8-(C-4-2, C-4-2)

### 定义

令 $p,q$ 为保留的 C-4-2 occurrence。一次二层 occurrence 满足：

```text
p 在 q 前面
p 与 q 的底层 token 位置不重叠
两者最小包围 span <= 8
不跨文档边界
```

Pattern identity 包含 `left_pattern_id`、`right_pattern_id` 和两者的 gap bucket。

主结果只统计 $p\neq q$。$p=q$ 作为 self-composition 单独报告。

### 保留条件

```text
train support >= 20
validation support >= 5
NPMI >= 0.10
train/validation rate 相差不超过 2 倍
每个 support 量级最多保留 10,000 个
```

## 二层 Null 对照

对每篇文档已经发现的 C-4-2 occurrences：

1. 保留 occurrence 的 start、end 和 span；
2. 在相同 span 内随机置换 pattern identity；
3. 因而保留每个一级 pattern 在该文档中的 occurrence 数和 span 分布；
4. 重新统计 C-8 pair-of-pairs。

这个 null 破坏具体一级 pattern 之间的配对关系，同时保留一级 pattern 的边际频率和局部 occurrence 密度。

## 支持、失败与证据不足

### 支持当前二层命题

- 真实语料通过筛选的异质 C-8 patterns 显著多于 null；
- 真实语料的二层 NPMI 和 document coverage 高于 null；
- 多个 C-4-2 child 被两个及以上不同的异质 C-8 parents 复用；
- 结果在 validation 文档中稳定复现。

### 不支持当前二层命题

- 真实语料与 null 的异质 C-8 数量、NPMI 和复用度接近；
- 二层 pattern 主要来自 self-composition；
- 通过筛选的二层 pattern 只出现在极少数文档或固定网页模板中。

### 证据不足

- 一级筛选后 active C-4-2 太少，无法形成二层候选；
- beam 或 support 阈值截断了大部分候选；
- tokenization 或样本量改变后结果不稳定；
- 运行只完成 smoke test，尚未完成 10,000-document 实验。

## 阶段输出

```text
run_config.json
cache/tokens.bin
cache/offsets.npy
cache/splits.npy
cache/documents.jsonl
c4_patterns.csv
c8_real_patterns.csv
c8_null_patterns.csv
examples.jsonl
summary.json
run.log
```

进度每隔固定文档数打印：累计 documents、tokens、docs/s、tokens/s、事件数和 ETA。
