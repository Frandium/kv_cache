# Main Summary: A06_03 to A06_07 Router Geometry Line

## Core Question

在 feature frequency 严格均匀时，为什么随机初始化的 dot-product top-1 MoE 仍然不能自然形成 feature-level specialization？

这里的 feature 在 A06_04-A06_07 中指 synthetic `(slot_token,target_token)` pair。4 个 pair 严格均匀，4 个 experts，模型只初始化，不训练。

## 一句话判断

Feature 出现频率均匀，并不等于 router 会形成 feature-level specialization。当前证据把问题拆成一条有顺序的几何链：理想 gate-only 情况下 row norm 会制造偏置；进入 hidden state 后，hidden common component 成为更强的负载偏置；去掉 common 后 residual geometry 仍然存在；如果给 oracle feature label，目标 partition 可达；但不用 label、只做全局去均值、去主方向或白化，只能改善 load，不能恢复 feature routing。

## Decision Chain

```text
A06_03:
Pure gate-only Gaussian input shows router row-norm variation is a separable source of fixed top-1 cell imbalance.
↓
A06_04:
After initialized Transformer hidden-state formation, row-norm control is no longer the main explanation; hidden common component is the strongest tested load-bias source.
↓
A06_05:
Common-centering is not enough; residual hidden states still have structured geometry / covariance anisotropy that creates load imbalance.
↓
A06_06:
With oracle feature labels, feature centroids can produce perfect held-out feature routing. The target partition exists.
↓
A06_07:
Without labels, simple global centering / top-PC projection / whitening mostly improve load, not feature NMI. Load balance is not specialization.
```

## Most Important Numbers

A06_04 primary readout:

| Condition | Load $L$ |
| --- | ---: |
| raw hidden + raw router | 0.5578 |
| row-norm controlled router | 0.5577 |
| common-centered hidden | 0.2578 |

A06_05 primary readout:

| Condition | Load $L$ |
| --- | ---: |
| common-centered residual replay | 0.2577 |
| centered + whitened replay | 0.1071 |
| matched isotropic replay | 0.0874 |

A06_06 primary readout:

| Condition | Feature NMI | Load $L$ |
| --- | ---: | ---: |
| random Gaussian router | 0.1978 | 0.5610 |
| raw feature centroid | 1.0000 | 0.0000 |
| common-centered feature centroid | 1.0000 | 0.0000 |

A06_07 full sweep:

| Condition | Load $L$ | Feature NMI |
| --- | ---: | ---: |
| baseline raw | 0.6867 | 0.2302 |
| calibration mean | 0.2837 | 0.2353 |
| held-out batch mean | 0.2828 | 0.2350 |
| whitened residual | 0.0860 | 0.0150 |
| oracle feature centroid | 0.0000 | 1.0000 |

## Important Definitions

Load $L$ measures whether experts receive equal numbers of samples:

$$
L=m\max_e |p_e-1/m|
$$

Feature NMI measures whether routed expert identity matches the intended feature identity. Load balance can be good while Feature NMI is bad.

Calibration means the held-out estimation half of the synthetic samples. It is used to estimate global statistics such as mean and covariance, not to train the model and not to use feature labels in label-free controls.

Calibration mean uses

$$
c_{\text{calib}}=\frac{1}{N_{\text{calib}}}\sum_{j\in C}h_j
$$

and routes evaluation samples with $w_e^\top(h_i-c_{\text{calib}})$.

Whitening estimates calibration residual covariance

$$
\Sigma=\frac{R_C^\top R_C}{N_{\text{calib}}-1}
$$

then applies

$$
\tilde r_i=(h_i-c_{\text{calib}})U\operatorname{diag}((\lambda+\epsilon)^{-1/2})U^\top
$$

with $\epsilon=10^{-5}$ and scores with the same random router rows.

## Can Claim

- In this synthetic hidden-state setting, uniform feature frequency does not ensure uniform routing or feature specialization.
- Hidden common component is a strong step-0 load-bias source after hidden-state formation.
- Residual geometry remains active after common-centering.
- Oracle feature-centroid routing proves the target partition is geometrically reachable.
- Simple label-free global common/residual controls should not be promoted as feature-specialization methods.

## Cannot Claim

- No claim about real DCLM behavior yet.
- No claim that training preserves the oracle partition.
- No claim that all label-free feature discovery methods fail.
- No expert utility or semantic specialization claim.
- Load balance alone is not a valid specialization metric.

## Next Decision

The next natural iteration should be one of:

1. Feature discovery before routing: cluster, dictionary learning, contrastive grouping, or gradient proxy on residual hidden states.
2. Anti-lockin after a good partition: start from oracle / pseudo-oracle routing and test whether early top-1 training preserves or destroys it.
3. Real DCLM only after defining a feature/proxy specialization metric; do not use load-only success.
