# Protocol：A11_24 任务秩竞争

用户已授权 planned anchor。使用独立 Gaussian 因素 $u,z$、共享线性 rank 1/2 encoder、相同参数族与数据流，比较 NTP 与静态 MTP。$L_N=2L_u+L_z$，MTP 额外加入 $2L_A(z)$。五 seed、1000 steps。通过要求 rank-1 标准 loss gap `>0.5`、rank-2 绝对 gap `<0.05`，且 rank-1 未来 MSE 在 5/5 改善。只能支持线性受控 rank 机制。
