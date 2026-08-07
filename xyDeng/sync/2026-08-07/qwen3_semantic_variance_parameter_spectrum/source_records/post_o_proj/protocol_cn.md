---
experiment_id: A15_01_05_E04_qwen3_post_o_proj_full_attention_semantic_variance_atlas
anchor_id: 15_01_05_coarse_fine_semantic_variance_atlas
status: COMPLETED_DEPTH_PASS_SPECTRAL_FAIL
canonical_language: en
companion_language: zh
approved: 2026-08-06
completed: 2026-08-06
---

# Protocol：Qwen3-8B post-`o_proj` 完整 attention 粗细语义方差画像

英文 canonical：[protocol.md](protocol.md)。研究者已于 2026-08-06 明确批准实现、smoke、单节点 8×5090 full、分析、0806 自包含报告和 0807 认识更新；不含 commit、push、graph 或外部同步。

## 1. 唯一问题与对象

冻结 Qwen3-8B 中，在 `o_proj` 已完成跨头线性混合、但尚未与 residual 相加时，完整 attention branch

$$g_\ell=W_{O,\ell}a_\ell\in\mathbb R^{4096}$$

是否仍表现出“晚层细语义相对粗语义更可分”，以及该信号位于 post-`o_proj` 自然语料 covariance 的哪些谱带？它不是 32 个可分头、MLP 输入或 block 输出。

## 2. 数据、模型与固定因素

- `/data/share/Qwen3-8B`，36 blocks，冻结，bfloat16 前向；
- 精确复用 512 条 8×8 粗细语义文本、模板/事实包、最终冒号读出位置与 design/confirmation 划分；
- 精确复用 128 篇 DCLM、65,536 token calibration 与两个固定半集；
- 不允许看结果后修改文本、概念、读出、层、频带、随机 seed 或阈值。

## 3. 指标与物理含义

区分度

$$D_{\ell,s}=\frac{\operatorname{tr}(B_{\ell,s})}{\operatorname{tr}(W_{\ell,s})+\epsilon}$$

表示类别中心差异相对同类措辞变化有多清楚；不是准确率或下游效用。细粗对数比

$$R_\ell=\log\frac{D_{\ell,fine}}{D_{\ell,coarse}}$$

为正表示同层细语义相对更可分。主深度差分为 blocks 25--35 的中位 $R$ 减 blocks 1--12 的中位 $R$；block 36 单列。

每层自然语料 covariance 切成 16 个等秩 256 维带：F1=head，F2--F8=middle，F9--F16=tail。逐带报告：实际类别间方差 $b$、背景方差归一化 $q$、类别间/类内区分度 $j$、实际方差富集 $e$。

## 4. 假设、rival 与裁定

- 深度 Pass：主差分的层级 bootstrap 下界 $>0$，4/4 模板和 8/8 父类留一同向；
- 实际方差 head 主导：粗、细的 F1 在 blocks 1--35 至少半数层具有最大每方向 $b$；
- 固定 non-head Pass：middle/tail 必须超过同维随机方向，并通过 design/confirmation、两个 calibration half、干扰变量残差化和三个特征值 floor；
- 最强 rival：`o_proj` 旋转或稀释 pre-`o_proj` 集中；表面 tail 现象只来自 basis 或小特征值噪声。

typed verdict 为 `depth_pass_spectral_pass`、`depth_pass_spectral_fail`、`depth_fail_spectral_pass`、`joint_fail` 或 `insufficient_<guard>`。

## 5. 强制有效性门

36 层 hook/shape；模块输出与直接线性映射相对误差 $<10^{-6}$；确定性 replay；512 条语义与 65,536 calibration token 完整覆盖；covariance 重建、能量守恒、FP32/FP64 Gram、正交性、half-split 和 eigenvalue floor 全部通过。任一门失败即停止，不降低要求或筛选层/概念。

## 6. 必须交付

完整 CSV、模型/数据/运行 manifest；post-`o_proj` covariance 谱、深度轨迹、层×16带热图、决定性区分度图；与 pre-`o_proj`、MLP input（仅同定义量）和完整 residual 的配对描述比较；`summary.md`、`detailed.md`、0806 自包含报告和 0807 五段式 focus。所有主图必须实际打开审核。

## 7. 执行合同

SCO ACP 单节点 `5090-8-spot`：`share-space` / `computing-cluster-5090-01g` / `n12lp.nn.i10a.8` / 1 worker / spot / normal / 8 GPU。顺序为合同测试→smoke→full→分析→视觉审计→证据记录。A--G 均 `CONFIRMED`，一致性检查 `PASS`。

## 8. 结论边界

本实验只比较冻结表示边界。不能把站点差异因果归于 `o_proj`，不能证明深层组合计算、某谱带的 Router 适用性、专家兼容性或训练收益。

## 9. 执行闭环

正式 8×5090 ACP 作业 `om-5y1d8uf1` 零重试成功，全部 hard guards 通过。typed verdict 为 `depth_pass_spectral_fail`：深度命题和实际方差 head 主导通过；middle 相对富集通过局部对照，tail 稳定性失败，因此注册的联合频谱命题失败。结果见 [summary_cn.md](summary_cn.md) 与 [detailed.md](detailed.md)。
