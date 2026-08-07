# A15_01_05_E04 结果摘要：post-`o_proj` 完整 attention 语义方差画像

## 裁定

typed verdict 为 **`depth_pass_spectral_fail`**。

- **深度命题 Pass：**经过 `o_proj` 跨头混合后，晚层细语义相对粗语义的区分度仍显著提高。
- **实际能量 head 主导 Pass：**blocks 1--35 中，粗、细语义的最大每方向类别间方差始终位于 F1。
- **联合 non-head 频谱命题 Fail：**middle 出现小而稳定的相对富集，但 tail 未通过独立 calibration-half 基底稳定性门，因此不能建立完整、稳定的 non-head 坐标。

一句话认识更新是：

> `o_proj` 没有抹去晚层细语义增强，并把一小部分相对语义方差稳定地移向 middle；但这还不是稳定的全 non-head 坐标，更不是已经验证的 Router 特征。

## 主深度结果

区分度 $D=B/W$ 表示类别中心差异相对同类措辞和事实变化有多清楚。$R=\log(D_{fine}/D_{coarse})$ 为正表示同层细语义相对更可分。

早层 blocks 1--12 的 $R$ 中位数为 0.131，晚层 blocks 25--35 为 0.762；主差分 $T_{post}=0.631$，层级 bootstrap 95% 区间为 [0.553, 1.148]。四个模板和八次父类留一全部同向。blocks 35/36 的细粗区分度比分别为 2.13/2.36，最后一层没有反转。

![跨站点深度比较](figures/figure1_cross_site_depth_comparison.png)

pre- 与 post-`o_proj` 轨迹在描述上接近；由于没有进行 activation replacement 或匹配干预，不能把站点间数值差异因果归于 `o_proj`。

## 频谱结果

4096 维严格划分为 16 个等秩 256 维带：F1=head，F2--F8=middle，F9--F16=tail。粗、细语义的实际类别间方差在 blocks 1--35 始终以 F1 最大。

晚层细语义相对粗语义的实际方差份额富集为：head -0.106、middle +0.063、tail +0.159。换成人话：相对各自的总语义方差，细语义比粗语义在 head 少约 10.0%，在 middle 多约 6.5%，在 tail 多约 17.2%。

middle 点估计略高于同维随机方向 q95=0.059，并通过模板、两个独立 calibration half、干扰变量残差化和特征值 floor 检查。tail 点估计更大，但在至少一个独立半集基底的晚层改变方向，因此 tail Fail；注册的联合频谱条件要求 middle 与 tail 均稳定，所以总体仍为 spectral Fail。

![层×频带画像](figures/figure2_post_o_proj_layer_band_heatmaps.png)

![频带区分度](figures/figure3_decisive_post_o_proj_band_discriminability.png)

## 有效性与边界

36 层、65,536 个 calibration token 和 512 条语义文本全部覆盖。正式 smoke/full 中模块输出与直接 $W_Oa$ 重放误差为 0；FP32 Gram 相对直接 FP64 的最大误差为 $7.22\times10^{-8}$。covariance 重建、能量守恒、half-split、模板、父类留一、干扰变量和 floor 门均完成。

第一次 ACP 作业 `om-1hc00w00` 因工程 guard 改变 bfloat16 GEMM shape 而无效；只修复 guard 后，正式 8×5090 作业 `om-5y1d8uf1` 零重试成功。数据、模型、指标、阈值和科学条件均未改变。

本实验只建立冻结表示的信号位置。它不能证明 `o_proj` 因果创造 middle 现象、深层执行组合计算、middle 能预测专家效用，或频谱路由改善训练。

唯一下一决策是：用独立平衡 taxonomy 同时复现 post-`o_proj` 的正深度差分和晚层 middle 小幅富集；在此之前，middle 只是单一 taxonomy 的表征发现。
