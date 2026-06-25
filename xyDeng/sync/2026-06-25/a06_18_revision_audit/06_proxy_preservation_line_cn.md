# 06 研究线：proxy discovery、router initialization 与 preservation

## 裁定

06 线把“形成 feature-level expert partition”拆成阶段合同：

```text
feature/proxy 是否存在
-> proxy 是否能转成 router rows
-> routing 是否能被 training preservation 保持
```

当前结论是：

```text
controlled route-position geometry 可发现、可路由、可保持；
real-DCLM proxy clusters 可发现且 step 0 可线性路由；
但 ordinary real-DCLM training 会在 step 5/10 覆盖 proxy routing。
```

## 阶段证据

### 1. Controlled route-position geometry

A06_16/A06_17 说明：在 corrected no-position bridge 中，route-position
hidden states 有干净 feature geometry。

关键判断：

```text
route-only / slot offset 3: feature_NMI = 1.0, 8/8 seeds pass
all-position: mean feature_NMI = 0.797, only 1/8 strict pass
```

解释：

```text
all-position clustering 的失败不是 feature geometry 不存在，而是把非路由状态
混进了拟合样本池。真正需要的是 route-relevant state selector。
```

A06_18 revision 进一步测试了 PCA latent clustering、bottleneck AE latent
clustering、SAE-code clustering：

```text
route-only / slot offset 3: feature_NMI = 1.0, 8/8 seeds pass
raw all-position: mean feature_NMI = 0.831, 2/8 strict pass
best revision selector: PCA q=4 mean feature_NMI = 0.871, 2/8 strict pass
SAE L1 8x reconstruction MSE = 0.0034, but feature_NMI = 0.729
```

判断：

```text
generic representation learning 不能替代 route-relevant selector；
reconstruction quality 不是 route-readout quality；
下一版 selector 必须显式加入 route-local 或 route-readout constraint。
```

### 2. Real-DCLM proxy discovery

A06_10 支持真实 DCLM hidden states 中存在稳定 proxy clusters。

关键边界：

```text
proxy stability 高于 random、frequency 和 position nuisance；
但 proxy cluster 不是语义 feature 证明，也不是 expert specialization 证明。
```

### 3. Step-0 router bridge

A06_11 支持 proxy centers 能转成 linear top-1 router rows。

关键边界：

```text
raw-center raw-input 是当前最强 actual bridge；
residual-center centered-input 是诊断，不是直接可部署 raw router。
```

### 4. Training preservation

A06_12 是当前 bottleneck：

```text
raw-center step-0 proxy_route_NMI = 0.7549
step 5 NMI = 0.0410
step 10 NMI = 0.0131
LM loss 与 random 接近
```

解释：

```text
这不是 proxy discovery failure，也不是 step-0 linear bridge failure；
它是 ordinary next-token training 覆盖了 proxy routing。
```

## 合成到真实的边界

controlled synthetic / bridge 能证明：

- route-position feature geometry 是可达对象；
- all-position sample pool 不合法；
- 在可控目标下 pseudo initialization 可以被保持。

real-DCLM proxy 能证明：

- 真实语料 hidden states 中有可重复 proxy；
- step-0 gate 可以对 proxy routing；
- ordinary training 会覆盖这个 routing。

两者之间仍缺少：

```text
label-free route-relevant state selector for real text；
anti-feedback / preservation objective；
proxy partition 的 expert utility 证据。
```

## 下一步

优先写 preservation / anti-feedback protocol。不要直接设计 utility method。

最小问题：

```text
在固定 step-0 proxy labels 后，哪一种训练控制能让 proxy_route_NMI 穿过
step 5/10，同时 LM loss 不显著变坏？
```

## Source Links

```text
Projects/from-attention-to-search/main/problem_anchors/06_geometry_proxy_preservation/
Projects/from-attention-to-search/main/experiments/A06/A06_10_real_dclm_proxy_feature_operationalization/summary.md
Projects/from-attention-to-search/main/experiments/A06/A06_11_real_dclm_proxy_center_router_initialization/summary.md
Projects/from-attention-to-search/main/experiments/A06/A06_12_real_dclm_proxy_init_training_preservation/summary.md
Projects/from-attention-to-search/main/experiments/A06/A06_17_all_position_route_relevant_feature_discovery/summary.md
Projects/from-attention-to-search/main/experiments/A06/A06_18_label_free_route_relevant_state_selector/summary.md
```
