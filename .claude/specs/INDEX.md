# Specs

- [cg-atomistic-mapping-redesign](cg-atomistic-mapping-redesign.md) — 简化 CoarseGrain 至与 Atomistic 对称 [draft]
- [molrs-backend](molrs-backend.md) — molrs 作为 molpy 必选后端：Box 继承 + NeighborList/RDF + 替换 RDKit + 暴露 MSD/Cluster 等 [code-complete]
- [dielectric-susceptibility-01-molrs-signal](dielectric-susceptibility-01-molrs-signal.md) — molrs signal 原语：acf_fft (裸未归一化) + apply_window + frequency_grid，单一职责 [approved]
- [dielectric-susceptibility-02-molrs-dielectric](dielectric-susceptibility-02-molrs-dielectric.md) — molrs dielectric：dipole_moment + current_density + static_dielectric_constant + EH/GK spectra + decompose_current [approved]
- [dielectric-susceptibility-05-validate](dielectric-susceptibility-05-validate.md) — 域验证：Kramers-Kronig + conductivity sum rule + route agreement (molrs) + 合成 Debye 集成测试 [approved]
- [molrs-analyses-expose-02](molrs-analyses-expose-02.md) — 暴露剩余 molrs.compute 分析（Steinhardt/Hexatic/Nematic/SolidLiquid、LocalDensity/GaussianDensity、StaticStructureFactorDebye、BondOrder、PMFTXY、ClusterProperties）为 molpy 薄壳算子 [code-complete]
