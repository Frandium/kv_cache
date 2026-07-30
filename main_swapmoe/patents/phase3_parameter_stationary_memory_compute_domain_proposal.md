# 面向端侧 MoE 解码的容量-带宽解耦分层存储加速器架构、执行方法及芯片

> 文档性质：三阶段专利初步 proposal / 技术交底前置材料
>
> 版本日期：2026-07-29
>
> 状态：供项目组、甲方芯片团队和专利代理人讨论，尚非正式权利要求书
>
> 建议英文标题：**A Capacity-Bandwidth-Decoupled Hierarchical Memory Accelerator Architecture, Execution Method, and Chip for Edge MoE Decoding**

## 摘要

本 proposal 面向手机、车机和个人电脑上的大规模 MoE 模型 batch size = 1 自回归推理。与训练和云端大批量推理相比，该 workload 在一个 decode 时段内只访问完整模型参数的一小部分，并形成稳定的 working set 与 idle set：

1. **Working set**：当前持续产生主要读取流量的数据，包括 attention/dense 权重、输出头、历史 KV Cache 和近期激活 experts；
2. **Idle set**：当前 token 不访问的大量 inactive experts。

working set 与 idle set 是对 workload 的逻辑分类。硬件侧设置与之对应的两类存储资源：

1. **高带宽层（fast tier）**：容量较小，保留较高的存储接口带宽，用于承载 working set；
2. **容量层（capacity tier）**：容量较大、单位容量成本较低、带宽较低，用于承载 idle set。

当前 token、hidden state、query、路由结果和中间向量的容量很小，统一存放在高速层；执行算子时，它们按常规数据通路进入计算核心已有的寄存器、L1、共享存储或其他局部存储。该架构解除“模型全部容量必须采用同一带宽等级存储”的绑定。对于一个逻辑容量约为 20% 的 working set，只为该部分配置高带宽接口，其余约 80% 的 inactive parameters 使用容量优先的存储。高带宽层与容量层可以分别采用浅层 HBM 与 DDR/LPDDR、高速 DRAM 与容量型 DRAM、HBM 与 Flash，或 SRAM、DRAM、Flash 三级组合。专利核心是两类存储的容量-带宽非对称配置及其面向端侧 MoE 的数据放置和访问路径，具体存储介质与封装形态属于实施方式。

以 Qwen3-30B-A3B 为例，在 BF16、每层为 top-8 激活预留 16 个高速 expert slots 的配置下，4K 和 32K context 对应的高速层容量比例约为 16.4% 和 20.1%。将本项目测得的 $92.70\%$ expert 驻留命中率作为代理工作点时，高速层可覆盖约 $95.49\%$--$96.95\%$ 的当前读取字节。在本文采用的峰值带宽和端到端时延模型中，“保留全 HBM 聚合接口带宽的浅层 HBM + 546 GB/s 高速 LPDDR”预计以约 $8.5\%$--$12.6\%$ 的端到端时延代价，换取约 $40\%$--$60\%$ 的一阶介质成本降低。

---

## 1. 问题动机

### 1.1 目标场景与 workload 分类

本方案面向大规模 MoE 模型在手机、车机和个人电脑等边缘设备上的 batch-1 自回归解码。一次 token 前向中，数据具有如下分化：

| Workload 集合 | 典型内容 | 当前访问特征 | 容量特征 |
|---|---|---|---|
| Working set | attention/dense 权重、输出头、历史 KV Cache、近期激活 experts | 当前 token 必须读取或持续复用 | 远小于完整模型，但明显大于片上 SRAM/L2 |
| Idle set | 当前未激活 experts | 当前 token 不产生读取流量 | 占据完整模型的大部分容量 |
| 小状态 | 当前 token、hidden state、query、新增 KV、路由结果、中间向量 | 当前必须处理 | 字节数很小，存放在高速层 |

端侧 batch-1 decode 的关键性质是：**当前产生读取流量的参数远少于完整模型容量**。MoE 每层只激活少量 experts；attention/dense 权重和 KV 虽持续访问，但其集合可以明确界定；当前 token 和中间向量的尺寸又远小于参数。

因此，系统设计应分别考虑两个比例：

$$
f
=
\frac{C_{\text{fast}}}{C_{\text{model}}+C_{\text{KV}}},
$$

$$
\rho
=
\frac{D_{\text{fast}}}{D_{\text{req}}}.
$$

其中 $f$ 是高速层容量比例，$\rho$ 是当前前向请求中由高速层服务的字节比例。端侧 MoE 的有利区域是：

$$
f \ll 1,
\qquad
\rho \approx 1.
$$

这意味着少量高速容量可以覆盖绝大部分运行时流量。

### 1.2 全 HBM 基线与端侧推理需求

本文以“完整模型全部驻留在同质 HBM 中”的加速器作为性能基线。该设计为每一字节模型容量配置相同等级的高带宽接口，适合训练、大 batch 推理和多请求并发：

- 训练需要读取权重并写入梯度和优化器状态；
- 大 batch 或多租户推理可以在同一时段激活更广泛的参数集合；
- 大量 activation 和并行请求能够持续利用全部 HBM channels。

端侧 batch-1 MoE 解码的目标不同：

- 权重只读，大量 inactive experts 只贡献容量，不贡献当前流量；
- 单 token 只激活少量 experts，且 token activation 很小；
- 设计目标聚焦单请求时延、成本和功耗。

全 HBM 基线提供性能上限，但为 idle experts 配置与 working set 相同的单位容量带宽，会使端侧专用芯片承担较高的存储介质和封装成本。由此得到本方案的硬件问题：

> 如何在保留 working set 高带宽供数能力的同时，用单位容量成本更低的存储承载大部分 inactive parameters，使端侧 MoE 推理接近全 HBM 时延，并降低存储系统成本。

### 1.3 连续激活提高高速层命中率

设 $h$ 为当前 token 激活的 expert 权重已经位于高速层的字节比例：

$$
D_{\text{expert,fast}}
=
hD_{\text{expert,active}},
\qquad
D_{\text{expert,capacity}}
=
(1-h)D_{\text{expert,active}}.
$$

连续性增强路由使相邻 token 更可能复用同一组 experts。active expert 进入高速层后可以连续服务多个 token，inactive expert 可以持续留在容量层。该性质降低高速层所需的 expert slot 数量和两层之间的平均迁移字节。

运行时也可以通过编译期 profile、频率统计、固定映射或常规替换策略提高 $h$。这些策略决定具体命中率和迁移频率，硬件架构提供可区分的两类存储资源、地址映射和数据通路。

---

## 2. 现有方案及其痛点

### 2.1 NVIDIA GPU

NVIDIA 数据中心 GPU 使用多组 HBM stack、分片 L2、片上网络和 SM/Tensor Core。完整模型容量通常在相同 HBM 地址空间中跨 channels 条带化，任何模型对象都可以使用聚合 HBM 带宽。这是训练和高吞吐推理需要的通用设计。

对于端侧 batch-1 MoE：

- inactive expert 与 active expert 使用相同等级的 HBM 容量；
- 片上 L2 明显小于跨全部 Transformer 层的 expert working set 和长上下文 KV；
- 通用缓存可以减少短期重复访问，但不能把几十 GB 的模型容量转化为永久高速驻留；
- 存储系统成本随完整模型容量增加，而当前访问流量只集中在其中一小部分。

本方案以全 HBM GPU 为性能基线，重新分配“每 GB 容量所获得的接口带宽和成本”，GPU 计算核心继续执行相同的矩阵运算。

参考：

- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/)

### 2.2 Google TPU

Google TPU 以 MXU、向量单元和 HBM 支持规则的大矩阵与高吞吐计算。训练或 pod 级服务可以通过 batch 和并发提高参数复用。batch-1 MoE decode 更接近动态选择的 GEMV，单 token 的 active expert 集合很小。

TPU 的高带宽内存同样主要作为同质容量池使用。对于端侧专用实现，完整模型容量全部采用同一高带宽等级，会延续容量与带宽绑定的问题。

参考：

- [TPU System Architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
- [Cloud TPU v5p](https://docs.cloud.google.com/tpu/docs/v5p)

### 2.3 华为 Ascend NPU

Ascend AI Core 包含矩阵计算、向量计算及 Global Memory、L1/Unified Buffer、L0 等层级。软件能够显式切块、DMA 和 double buffer，这些机制适合隐藏部分数据传输时延。

端侧 MoE 的存储容量问题位于更低一层：即使片上 DMA 和 buffer 得到优化，当前未激活 experts 仍与高频数据占用同等级的外部存储资源。容量-带宽解耦可以与现有 Cube/Vector、DMA 和片上 buffer 机制组合。

参考：[Ascend NPU 架构与数据通路](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/programug/Ascendcopdevg/docs/guide/%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97/%E9%AB%98%E7%BA%A7%E7%BC%96%E7%A8%8B/%E7%A1%AC%E4%BB%B6%E5%AE%9E%E7%8E%B0/%E6%9E%B6%E6%9E%84%E8%A7%84%E6%A0%BC/NPU%E6%9E%B6%E6%9E%84%E7%89%88%E6%9C%AC3510.md)。

### 2.4 异构存储与权重流式执行基础

现有系统已经证明多级存储可以共同承载大型模型：

- Cerebras WSE-3 在晶圆上分布 44 GB SRAM 和计算核心，MemoryX 使用外部 DRAM 与 Flash 保存大规模参数并按层流入计算晶圆；
- GPU/CPU 系统已经支持 HBM、DDR/LPDDR、CXL memory 和 SSD 等多层容量；
- JEDEC SPHBM4 使用与 HBM4 相同的 DRAM stacks 和不同的接口 base die，使其能够连接标准有机基板，说明 DRAM stack、主机接口宽度和封装成本可以分别设计；
- 现有 hot expert buffer、模型 offload 和 tiered memory 方案说明了参数放置与迁移的可实现性。

参考：

- [Cerebras WSE-3](https://www.cerebras.ai/press-release/cerebras-announces-third-generation-wafer-scale-engine)
- [Cerebras MemoryX](https://www.cerebras.ai/blog/cerebras-cs-3-vs-nvidia-b200-2024-ai-accelerators-compared)
- [JEDEC SPHBM4 公告转载](https://www.businesswire.com/news/home/20251211280454/en/JEDEC-Prepares-SPHBM4-Standard-to-Deliver-HBM4-Level-Throughput-with-Reduced-Pin-Count)

本方案进一步把存储层级的容量比例、流量比例和 MoE 连续激活联系起来，为端侧 batch-1 decode 配置专用的高速层与容量层。

---

## 3. 方案详情

### 3.1 总体架构

![图1：全 HBM 基线与容量-带宽解耦分层存储加速器](./figures/working_idle_hbm_gpu_architecture.svg)

本方案至少包含：

1. **计算核心阵列**：执行 attention、router、expert GEMV、逐元素算子和输出头；
2. **高带宽层（fast tier）**：较小容量、较高聚合带宽，保存 working set；
3. **容量层（capacity tier）**：较大容量、较低单位容量成本和较低聚合带宽，保存 idle set；
4. **层间复制/迁移引擎**：按 expert block、权重页或 KV page 在两类存储之间传输数据。

高带宽层与容量层均位于同一端侧推理设备的可访问存储系统中。两类存储可以处于同一封装，也可以由封装内存与板级/设备内存组合实现。

### 3.2 高带宽层

高带宽层保存每个 token 持续产生主要读取流量的数据，例如：

- 全部 attention/dense 权重和 router；
- 输出 LM head；
- 历史 KV Cache；
- 每层近期激活的 expert slots；
- 当前 token、hidden state、query、路由结果和中间向量；
- 必要的量化 scale、索引和元数据。

高带宽层的设计目标是**用较小容量保留接近全 HBM 基线的接口带宽**。一种实现方式是保持与全 HBM 基线相同数量的宽接口和 memory channels，同时减少每个高速 stack 背后的 DRAM 层数或容量密度。这样，高速层容量下降时，聚合带宽不必按容量同比下降。

高速层中的 expert 可以跨全部高速 channels 条带化，使单个 batch-1 expert GEMV 仍能使用聚合存储带宽和全部计算核心。

### 3.3 容量层

容量层保存：

- 当前未激活 experts；
- 高速层对象的后备副本或可恢复镜像；
- 尚未进入当前上下文 working set 的模型块。

容量层优先优化每 GB 成本、容量密度和静态功耗。其接口带宽低于高速层。容量层中的对象发生访问时，可以：

1. 经容量层控制器直接送入计算阵列；
2. 先进入高速层或 staging buffer，再送入计算阵列；
3. 在后续 token 需要持续复用时迁入高速层。

容量层采用 Flash 时，数据需要经过 DRAM/SRAM staging buffer，迁移粒度按 Flash page、压缩 expert block或连续权重块组织。

### 3.4 软件可见接口

硬件可以向软件暴露：

- `FAST_TIER_MEMORY`：高带宽、小容量地址空间；
- `CAPACITY_TIER_MEMORY`：低成本、大容量地址空间；
- 统一虚拟地址与页属性：由页表或对象描述符标记当前物理层级；
- 复制命令：以 expert、权重页或 KV page 为单位执行层间复制；
- 查询命令：返回对象驻留层级、命中状态和迁移状态。

模型加载器可在初始化时完成静态放置：attention/dense 权重、输出头和 KV 预留进入高速层，inactive experts 进入容量层，并在高速层为每层预留若干 expert slots。

### 3.5 物理实施形态

| 实施形态 | 高带宽层 | 容量层 | 主要取舍 |
|---|---|---|---|
| A | 保留宽接口数量的浅层/低容量 HBM | DDR 或 LPDDR | 最接近全 HBM 带宽，容量层成本较低 |
| B | 宽接口高速 DRAM | 窄接口容量型 DRAM | 可避免标准 HBM 的容量粒度，需定制两类 DRAM 接口 |
| C | HBM 或高速 DRAM | NAND Flash/UFS/NVMe | 容量成本最低，miss 时延高，要求极高字节命中率 |
| D | SRAM scratchpad + 高速 DRAM | 容量 DRAM + Flash | 三级结构，分别承载当前矩阵块、working set 和长期 idle set |

“浅层 HBM”表示减少高速 stack 的容量，同时保留较宽外部接口和 channel 数量。该实现需要存储厂商提供合适的 stack height、die density 或定制 base die。

### 3.6 硬件执行流程

对每个 decode token：

1. 计算核心从高速层读取 attention/dense 权重和历史 KV，生成 attention 输出并写入新增 KV；
2. router 产生 expert ids，映射表查询各 expert 所在层级；
3. 高速命中的 experts 直接由高速层向计算阵列供数；
4. 容量层命中的 experts 经容量接口直接供数或进入 staging buffer；
5. 计算核心完成 expert GEMV、聚合和残差计算；
6. 对预计继续使用的 expert，复制引擎可以将其保留或迁入高速层；
7. 层输出写回高速层中的下一层输入地址，并进入下一层。

---

## 4. 实施例：一个 Token 的四阶段前向过程

| 阶段 | 主要动作 | 大状态流向 | 小状态流向 |
|---|---|---|---|
| 1. Attention 与 KV 更新 | 计算当前 token 的 Q/K/V 和 attention 输出 | attention 权重与历史 KV 从高速层进入计算阵列；新增 K/V 写入高速层 | hidden state 从高速层进入计算核心的现有局部存储，query 等结果按正常算子数据流产生 |
| 2. MoE 路由与层级命中 | 产生 expert ids 并查询存储位置 | 命中高速层时直接进入计算；命中容量层时发起低速读取或迁移 | 路由结果和对象描述符在控制通路中传递 |
| 3. MoE Expert 计算 | 计算被选 experts 并聚合输出 | expert 权重从对应存储层进入计算阵列，高速层承担绝大部分字节 | hidden state 沿计算核心的现有数据通路参与各 expert 计算 |
| 4. 状态保持与下一层推进 | 形成下一层输入并更新驻留状态 | KV 与高频 experts 保持在高速层，inactive experts 留在容量层 | 层输出写回高速层，作为下一层 hidden state |

### 4.1 阶段一：Attention 计算与 KV 更新

当前 hidden state 保存在高速层。执行本层时，hidden state 和该层 attention 权重从高速层进入计算核心，生成 query、key 和 value。新 key/value 写入高速层中的 KV Cache；历史 KV 从高速层送入 attention 计算单元，形成当前 attention 输出。

### 4.2 阶段二：MoE 路由与层级命中

router 产生当前层的 expert ids。分层内存控制器查询 expert 描述符：

- **高速命中**：expert 权重已经位于高速层，直接向计算阵列发出读取请求；
- **容量层命中**：容量层控制器读取对应 expert block。该 block 可以直接服务当前计算，也可以进入 staging buffer，并在后续需要持续复用时迁入高速层。

连续性增强路由使多数 token 复用已有高速层 expert slots，从而使容量层承担较大的静态容量，但只承担较少的运行时字节。

### 4.3 阶段三：MoE Expert 计算

hidden state 沿计算核心已有的寄存器、L1、共享存储或其他局部数据通路参与计算。expert 权重按高速层或容量层的实际带宽顺序到达计算阵列，随后执行 gate/up/down projections 和 expert 输出聚合。计算核心、kernel launch 和算术操作与全 HBM 基线相同，差异来自两类存储服务当前请求字节的时间。

### 4.4 阶段四：状态保持与下一层推进

MoE 输出经过残差连接和归一化后形成下一层 hidden state，并写回高速层中的下一层输入地址。新增 KV、attention/dense 权重和近期 expert working set 保持在高速层。其余 inactive experts 保留在容量层。

该四阶段过程逐层重复，直至输出头产生下一个 token。

---

## 5. 理论性能、容量比例与成本估算

### 5.1 时延模型

设全 HBM 基线的聚合有效带宽为 $B_0$，当前层实际读取字节数为 $D_{\text{req}}$：

$$
T_0
\approx
\frac{D_{\text{req}}}{B_0}
+
T_{\text{compute}}
+
T_{\text{other}}.
$$

本方案将请求字节拆成高速层字节与容量层字节：

$$
D_{\text{req}}
=
D_{\text{fast}}
+
D_{\text{capacity}},
\qquad
\rho
=
\frac{D_{\text{fast}}}{D_{\text{req}}}.
$$

当两类 DRAM 均可向计算阵列直接供数时：

$$
T_{\text{tier}}
\approx
\frac{D_{\text{fast}}}{B_{\text{fast}}}
+
\frac{D_{\text{capacity}}}{B_{\text{capacity}}}
+
T_{\text{compute}}
+
T_{\text{other}}.
$$

若容量层数据必须先进入高速层或 staging buffer，则增加：

$$
T_{\text{stage}}
\approx
\frac{D_{\text{capacity}}}{B_{\text{fast}}}
+
L_{\text{stage}}.
$$

两种架构中的 $T_{\text{compute}}$ 与 $T_{\text{other}}$ 采用相同值。性能差异由 $B_0$、$B_{\text{fast}}$、$B_{\text{capacity}}$ 和容量层实际服务的字节比例共同决定。

当 $B_{\text{fast}}=B_0$ 且容量层可直接供数时，相对全 HBM 的传输时延比例为：

$$
R_T
=
\frac{T_{\text{tier,mem}}}{T_{0,\text{mem}}}
=
\rho
+
(1-\rho)
\frac{B_0}{B_{\text{capacity}}}.
$$

容量上的 20:80 分布不等于流量上的 20:80 分布。本方案成立依赖的是容量层很大、运行时字节占比却很小。

### 5.2 Qwen3-30B-A3B 高速层容量比例

Qwen3-30B-A3B 的官方配置为 48 层、hidden size 2048、128 个 routed experts、每 token 激活 8 个 experts、expert intermediate size 768、4 个 KV heads，BF16 参数量约为 30.53B。

参考：[Qwen3-30B-A3B 官方 config.json](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)。

BF16 模型权重约为：

$$
C_{\text{weight}}
\approx
56.87\ \text{GiB}.
$$

一个 expert 的 BF16 权重约为 9 MiB。高速层为每层保留 16 个 expert slots，用于容纳 top-8 active experts 及切换余量；同时保存全部 attention/router 权重和输出 LM head。对应高速权重约为：

$$
C_{\text{fast,weight}}
\approx
9.04\ \text{GiB}.
$$

BF16 KV Cache 在全部 48 层的容量为：

$$
C_{\text{KV}}(s)
=
48\times\frac{s}{512}\ \text{MiB}.
$$

| Context | 高速权重 | KV Cache | 高速层总容量 | 模型+KV 总容量 | 高速层比例 |
|---:|---:|---:|---:|---:|---:|
| 4K | 9.04 GiB | 0.375 GiB | 9.42 GiB | 57.25 GiB | 16.4% |
| 32K | 9.04 GiB | 3.00 GiB | 12.04 GiB | 59.87 GiB | 20.1% |

因此，20% 高速容量与 80% 容量层可以作为 BF16、最高约 32K context 的首个实施例。若采用 INT4 权重而 KV 保持 BF16，32K context 下 KV 占比上升，高速层比例约为 30.6%；专利可将高速层容量范围定义为总模型运行容量的 10%--40%，优选 20%--30%。

### 5.3 Expert 命中率与 Context Length 对高速存储覆盖率的影响

每层 attention 权重约为 36 MiB，router 约为 0.5 MiB，8 个 active experts 共 72 MiB。每层历史 KV 读取量为：

$$
D_{\text{KV}}(s)
=
\frac{s}{512}\ \text{MiB}.
$$

设 active expert 的高速层字节命中率为 $h$：

$$
D_{\text{fast}}
=
36.5
+
D_{\text{KV}}(s)
+
72h,
$$

$$
D_{\text{capacity}}
=
72(1-h),
$$

$$
D_{\text{req}}
=
108.5
+
D_{\text{KV}}(s).
$$

由此得到高速存储覆盖率：

$$
\rho(h,s)
=
\frac{
36.5+s/512+72h
}{
108.5+s/512
}.
$$

![图2：Expert 命中率与 Context Length 对高速存储覆盖率的影响](./figures/qwen3_fast_byte_coverage_vs_expert_hit.svg)

图中横轴 $h$ 表示 active-expert 权重由高速层命中的字节比例，纵轴 $\rho$ 表示当前层全部运行时读取字节中由高速层服务的比例。context 越长，常驻高速层的 KV 读取量越大，因此在相同 expert 命中率下，整体高速存储覆盖率越高。

在本项目的连续 routing 实验中，实测$h=92.70\%$，4K--32K context 的高速存储覆盖率为 $95.49\%$--$96.95\%$。

### 5.4 多种物理实现的 latency 与介质成本

#### 5.4.1 公共假设

BF16 模型加 KV 需要约 60 GiB。全 HBM 基线采用三组 24GB HBM3E，每组峰值带宽按 1.2 TB/s 计算：

$$
B_0
=
3.6\ \text{TB/s}.
$$

本方案的 HBM 实施例保留三组高速接口，但降低每组高速 stack 的容量，使高速层逻辑容量约为总容量的 20%。该假设用于体现“保留带宽、减少高速容量”的设计目标。

端到端时延由存储供数、计算和 kernel 调度等部分构成。设全 HBM 基线中存储供数时间占端到端时延的比例为：

$$
\beta
=
\frac{T_{\text{HBM,mem}}}
{T_{\text{HBM,e2e}}}.
$$

本轮专利模拟采用 $\beta=0.5$，表示 batch-1 MoE 解码中约 $50\%$ 的端到端时延来自存储供数，其余约 $50\%$ 来自计算、kernel launch、路由和调度。若分层架构的存储供数时延相对全 HBM 的比例为 $R_T$，则端到端时延比例为：

$$
\frac{T_{\text{tier,e2e}}}{T_{\text{HBM,e2e}}}
\approx
1
+
\beta(R_T-1).
$$

$\beta=0.5$ 是面向当前专利比较的一阶模拟值，后续可由全 HBM 目标平台的逐层 profiling 结果替换。

成本采用一阶介质模型：

$$
c_{\text{HBM}}=4,
\qquad
c_{\text{DRAM}}=1,
\qquad
c_{\text{Flash}}=0.1.
$$

设高速层容量比例为 $f$，则相对全 HBM 的介质成本比例为：

$$
R_C
=
\frac{
f c_{\text{fast}}
+
(1-f)c_{\text{capacity}}
}{
c_{\text{HBM}}
}.
$$

例如 $f=20\%$、$c_{\text{HBM}}/c_{\text{DRAM}}=4$ 时，HBM+DRAM 的介质成本比例为 40%，对应下降 60%；当 HBM/DRAM 价格倍数缩小到 2 时，对应下降 40%。

该归一化反映 HBM 历史上约为服务器 DDR5 每 bit 价格 4--5 倍、DRAM 每 bit 成本高于 NAND 一个数量级的结构关系。表中的 HBM+DRAM 介质成本下降 $40\%$--$60\%$，对应 HBM/DRAM 单位容量价格比为 2--4 的情景区间；2026 年普通 DRAM 价格上涨可能使该价格比继续缩小，若两者单位容量价格接近 1，则介质替换本身不再产生显著成本节省。介质成本不包含 compute die、额外控制器、base die、interposer、封装测试和良率。

#### 5.4.2 结果

下表采用本项目更大规模 Attention-Mean Routing 实验测得的 $h=92.70\%$ 作为代理工作点，并使用 $\beta=0.5$ 将存储供数时延变化换算为端到端时延变化。DRAM 容量层按直接向计算阵列供数计算；Flash 实施例额外计入一次写入高速 staging 的序列化时间。

| 物理实现 | $B_{\text{fast}}$ | $B_{\text{capacity}}$ | 4K 端到端 latency 变化 | 32K 端到端 latency 变化 | 一阶介质成本降低 | 判断 |
|---|---:|---:|---:|---:|---:|---|
| 浅层 HBM + 高速 LPDDR | 3.6 TB/s | 546 GB/s | **+12.6%** | **+8.5%** | 40%--60% | 首选实施例，形成约 8.5%--12.6% 的目标区间 |
| 浅层 HBM + 容量 LPDDR | 3.6 TB/s | 273 GB/s | +27.5% | +18.6% | 40%--60% | 需要更高命中率或更高容量层带宽 |
| 高速 DRAM + 容量 DRAM | 819 GB/s | 273 GB/s | +189.6% | +183.2% | 约 45%--70% | 无法逼近全 HBM 性能 |
| 浅层 HBM + UFS Flash | 3.6 TB/s | 4.2 GB/s | +1933.5% | +1305.8% | 约 76%--78% | 仅适合作为第三级后备层 |

在当前 $h=92.70\%$ 的代理工作点下，首选实施例的 4K 与 32K 端到端时延增幅分别为 $12.6\%$ 和 $8.5\%$。因此，本方案当前的模拟结论为：

公开依据：

- [Micron HBM3E：24/36GB，超过 1.2 TB/s](https://www.micron.com/products/memory/hbm/hbm3e)
- [Apple Mac Studio：546/819 GB/s](https://support.apple.com/en-us/122211)
- [NVIDIA DGX Spark：128GB LPDDR5X，273 GB/s](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- [Samsung UFS 4.0：4.2 GB/s](https://semiconductor.samsung.com/estorage/ufs/ufs-4-0/)
- [TrendForce HBM3E/DDR5 价格倍数](https://www.trendforce.com/presscenter/news/20251218-12843.html)
- [SK hynix：DRAM 与 NAND 单位 bit 成本关系](https://news.skhynix.com/the-density-cost-and-marketing-of-semiconductor-memory/)

#### 5.4.3 从介质成本到总制造成本

设存储介质与其基础封装占全 HBM 基线制造成本的比例为 $\alpha$，分层方案的介质成本比例为 $R_C$，新增控制器和封装成本占基线的比例为 $\delta$：

$$
\frac{C_{\text{tier,total}}}{C_{\text{HBM,total}}}
=
(1-\alpha)
+
\alpha R_C
+
\delta.
$$

若存储相关成本占总制造成本的 50%，介质成本下降 60%，且新增控制器与封装成本可以控制在原总成本的 0%--5%，则总制造成本预计下降：

$$
25\%\ \text{至}\ 30\%.
$$

因此，“端到端 latency 增加约 $8.5\%$--$12.6\%$、总制造成本降低约 30%”对应一个待验证的目标设计点：

- 高速容量比例约 20%；
- expert 高速层命中率采用本项目实测代理值 $92.70\%$；
- 高速层保留全 HBM 基线的聚合接口带宽；
- 容量层有效带宽约 546 GB/s；
- 全 HBM 基线中存储供数占端到端时延的比例取模拟值 $\beta=0.5$；
- HBM/容量 DRAM 的结构性单位容量成本差足以覆盖新增控制器与封装。

该结果是一阶峰值带宽与介质成本估算。有效带宽、随机访问效率、迁移延迟、功耗、stack granularity 和封装良率需要在芯片 cycle model、DRAMSim/Ramulator 模型及供应链成本数据中进一步标定。

### 5.5 可实现性与工程边界

组成技术均已有实现基础：

- HBM、DDR/LPDDR 和 Flash 可以由同一 SoC 或加速器通过不同控制器访问；
- 多种地址空间、统一虚拟地址、页迁移和 DMA copy engine 已广泛存在；
- HBM stack 的容量、stack height、base die 和主机接口可以协同设计；
- Cerebras MemoryX 已采用 SRAM、DRAM 与 Flash 组合承载模型权重。

主要工程挑战为：

- 在减少高速 DRAM 容量时保留足够的宽接口和聚合带宽；
- 高速层容量粒度与标准 HBM stack 容量不完全匹配；
- 两类 memory controller、PHY 和封装资源会增加固定成本；
- expert miss 的随机访问、迁移和一致性需要专用描述符与复制引擎；
- Flash 作为容量层时，需要处理页粒度、读取尾延迟和 staging；
- KV Cache 随 context 增长，高速层比例需要支持不同部署配置。

这些边界决定了首选产品形态是“低容量宽接口 DRAM/HBM + 高速 LPDDR/DDR 容量层”，Flash 更适合作为可选第三级。

---

## 6. 相关专利与非专利现有技术检索

> 检索日期：2026-07-29。以下为初步技术检索；权利要求检索、同族核验和法律意见由专利代理人进一步完成。

### 6.1 高相关专利

| 文献 | 相关内容 | 与本方案的关系 |
|---|---|---|
| [Intel US20250356164A1](https://patents.google.com/patent/US20250356164A1/en) | 面向 AI PC 的 full/partial hot expert buffer | 与高速 expert working set 高度相关，需突出容量-带宽配置、全 HBM 基线和端侧 MoE 容量/流量比例 |
| [AMD US11893502B2](https://patents.google.com/patent/US11893502B2/en) | 动态选择硬件执行不同 experts | 与 expert 放置和 fallback 路径有交叉 |
| [IBM US20240086682A1](https://patents.google.com/patent/US20240086682A1/en) | 在 3D compute-in-memory tiers 中保存 MoE experts | 覆盖 MoE expert 与多层存储/计算组合 |
| [IBM US20250190755A1](https://patents.google.com/patent/US20250190755A1/en) | PIM router 和近存 MoE 路由 | 覆盖 router 与存储侧执行组合 |
| [Samsung US20210398597A1](https://patents.google.com/patent/US20210398597A1/en) | PIM bank group 与 non-PIM bank group 共存 | 说明同一器件内异构 memory bank 已有较强现有技术 |
| [Micron US11983619B2](https://patents.google.com/patent/US11983619B2/en) | 在 memory device 中执行 Transformer 网络 | 覆盖通用 Transformer-in-memory |

### 6.2 相关系统与论文

| 文献 | 相关内容 | 对本方案的边界 |
|---|---|---|
| [Cerebras MemoryX](https://www.cerebras.ai/blog/cerebras-cs-3-vs-nvidia-b200-2024-ai-accelerators-compared) | 片上 SRAM 与外部 DRAM/Flash 权重存储 | 多介质模型存储本身已有系统实现 |
| [MoNDE](https://arxiv.org/pdf/2405.18832) | 冷 expert 留在 CXL memory-side NDP，移动 activation | 冷 expert 与容量层组合已有研究 |
| [PIM-GPT](https://www.nature.com/articles/s44335-024-00004-2) | 自回归 Transformer 的 hybrid PIM accelerator | 通用 decode 分层/PIM 已有研究 |
| [Tiered-Latency DRAM](https://arxiv.org/abs/1805.03048) | 在 DRAM 中形成快慢 segment | 单纯的快慢内存分层已有充分先例 |

### 6.3 建议主张的组合创新

建议独立权利要求围绕以下闭环：

1. 面向 batch-1 MoE decode 的加速器芯片或封装，包含高带宽层与容量层；
2. 高带宽层的容量小于容量层，而聚合带宽高于容量层；
3. 高带宽层保存 dense/attention 权重、KV Cache 和每层有限数量的 active expert slots；
4. 容量层保存其余 inactive experts；
5. 分层内存控制器根据对象描述符选择高速读取、容量层直接读取或层间迁移；
6. 高速层容量根据模型 working set 配置，使较小容量比例覆盖显著更高的请求字节比例；
7. 高速层在减少容量时保留多个宽接口或 memory channels，使高速层带宽不随容量等比例下降；
8. 物理实现允许 HBM+DDR/LPDDR、高速 DRAM+容量 DRAM、HBM+Flash 或三级组合。

方法权利要求可以覆盖：

1. 初始化时将 dense/attention 权重、输出头和 KV 空间映射至高速层；
2. 为每层分配 active expert slots，并将其余 experts 映射至容量层；
3. 每 token 路由后查询被选 expert 的层级；
4. 根据命中结果选择高速供数、容量层供数或迁移；
5. 在连续 token 间保持可复用 expert 的高速驻留。

现有技术对 tiered memory、hot expert cache、PIM 和模型 offload 已有较强覆盖。可争取的力度来自“**由端侧 batch-1 MoE 的高速容量比例与请求字节比例共同约束的容量-带宽解耦存储硬件，以及在缩减高速容量时保留聚合接口带宽的实现方式**”。

---

## 7. 结论

本 proposal 的最简逻辑为：

1. 端侧 batch-1 MoE decode 的完整状态可抽象为 working set 与 idle set；
2. working set 只占约 20% 的运行容量；在本项目 $92.70\%$ expert 驻留命中率的代理工作点下，高速层可覆盖约 $95.49\%$--$96.95\%$ 的当前读取字节；
3. 加速器据此设置小容量高带宽层和大容量低成本容量层；
4. 高速层保留接近全 HBM 基线的接口带宽，容量层补足完整模型容量；
5. 两层可以由 HBM、DDR/LPDDR、高速/容量型 DRAM 和 Flash 的不同组合实现；
6. Qwen3-30B-A3B 的一阶估算表明，浅层 HBM + 高速 LPDDR 在 $h=92.70\%$、$\beta=0.5$ 时的端到端时延增量约为 8.5%--12.6%，同时使介质成本下降约 40%--60%；
7. 当存储相关成本约占全 HBM 基线制造成本的一半时，总制造成本降低约 25%--30% 是可检验的目标区间。

该架构面向推理卡与训练卡的不同存储需求进行专用化设计。下一步需要用目标芯片的有效带宽、HBM stack granularity、LPDDR 控制器面积、封装成本和真实 expert 命中轨迹替换本文的一阶假设。
