# 真实语料层次 Pattern 实验结果

## 10,000-document pilot

运行目录：`results/pilot_10k`

```text
documents:                  10,000
tokens:                  6,950,856
documents at 1,024 limit:   4,113
total runtime:              5m 19s
```

`documents at 1,024 limit` 表示 tokenized 长度达到截断上限，不能据此区分恰好为 1,024 和实际被截断的文档。

## C-4-2

```text
train occurrence events:        16,564,305
validation occurrence events:    4,228,263
unique train candidates:          6,586,545
train support >= 100:                10,308
validation support >= 20:             9,989
NPMI >= 0.10:                         4,828
held-out stable:                       4,818
```

一级统计从 658 万种实际出现的 token pair 中保留 4,818 个高频、关联且跨文档稳定的 C-4-2。

## 真实语料与二层 Null

真实与 null 都生成 4,898,336 个二层 occurrence events。Null 保留每篇文档中一级 occurrence 的位置、span 和 pattern 频率，只在相同 span 内置换 pattern identity。

| 指标 | 真实语料 | Null |
|---|---:|---:|
| unique train C-8 candidates | 2,005,763 | 2,868,704 |
| 通过 support 的 candidates | 13,633 | 3,719 |
| 通过 NPMI 的 candidates | 8,201 | 325 |
| held-out 稳定异质 C-8 | 7,535 | 242 |
| held-out 稳定 self-composition | 78 | 66 |
| 异质 C-8 median NPMI | 0.263 | 0.145 |
| 异质 C-8 median train support | 42 | 33 |
| 异质 C-8 median document coverage | 0.42% | 0.34% |

Null 产生更多 unique pair identity，但同一 identity 很少重复。真实语料把相同数量的事件集中到少数可复现组合中，稳定异质组合数量是 null 的 31.1 倍。

## 一级 Pattern 的 Parent Reuse

在 4,818 个保留的 C-4-2 中：

```text
被至少 1 个异质 C-8 parent 使用: 1,422
被至少 2 个异质 C-8 parents 使用:  934
被至少 5 个异质 C-8 parents 使用:  593
90th percentile parent reuse:             6
maximum parent reuse:                   409
```

这说明真实语料中存在可被多个不同二层组合复用的一级 pattern。仍有一半以上一级 pattern 没有进入通过筛选的二层 parent。

## Pattern 内容检查

7,535 个真实异质 C-8 的粗分类：

```text
包含数字:              5,576  (74.0%)
包含字母且不含数字:    1,880  (24.9%)
只有标点或空白:           79  (1.0%)
```

support 最高的前 100 个中有 97 个包含数字，主要来自编号列表、日期和格式化网页。非数字组合中可以看到：

```text
in the + of the
at the + of the
the end + of the
as + well as
I don + 't know
in the + United States
For + example,
one of + the most
```

## 当前结论

当前结果支持：真实 DCLM token 序列中存在显著强于一级边际保持 null 的局部二层组合；部分一级 pattern 被多个二层 parent 复用。

当前结果没有证明：

- 这些 pattern 对应高层语义节点；
- 结构可以继续稳定递归到 C-16、C-32；
- 人类语言天然具有二叉或小 fan-in 结构；
- 结果独立于 Qwen BPE tokenization、网页模板和数字格式。

最主要的失败模式是数字列表和网页格式支配高 support 二层 pattern。下一轮需要分别报告 `digit / alphabetic / punctuation` 三个 stratum，并加入文本去重或模板过滤，然后再测试 C-16 和 flat four-token pattern 对照。
