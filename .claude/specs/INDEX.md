# Specs

- [cg-atomistic-mapping-redesign](cg-atomistic-mapping-redesign.md) — 简化 CoarseGrain 至与 Atomistic 对称 [draft]
- [molrs-backend](molrs-backend.md) — molrs 作为 molpy 必选后端：Box 继承 + NeighborList/RDF + 替换 RDKit + 暴露 MSD/Cluster 等 [in-progress]
- [dielectric-susceptibility-01-molrs-signal](dielectric-susceptibility-01-molrs-signal.md) — molrs signal 原语：acf_fft (裸未归一化) + apply_window + frequency_grid，单一职责 [approved]
- [dielectric-susceptibility-02-molrs-dielectric](dielectric-susceptibility-02-molrs-dielectric.md) — molrs dielectric：dipole_moment + current_density + static_dielectric_constant + EH/GK spectra + decompose_current [approved]
- [dielectric-susceptibility-04-python-compute](dielectric-susceptibility-04-python-compute.md) — Python Compute 胶水层：ACFAnalyzer + SpectralAnalyzer + DielectricSusceptibility，零 Python 物理计算 [approved]
- [dielectric-susceptibility-05-validate](dielectric-susceptibility-05-validate.md) — 域验证：Kramers-Kronig + conductivity sum rule + route agreement (molrs) + 合成 Debye 集成测试 [approved]
