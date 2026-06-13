---
title: CL&Pol short-range damping potentials (Thole + Tang-Toennies)
status: code-complete
created: 2026-06-10
---

# CL&Pol short-range damping potentials (Thole + Tang-Toennies)

## Summary

为 molpy 的 potential 层新增 CL&Pol 力场的两个短程阻尼势能评估器：Thole 阻尼（屏蔽偶极-偶极 / core-shell 库仑相互作用）与 Tang-Toennies 阻尼（屏蔽电荷-诱导偶极即 charge-Drude 库仑相互作用，防止极化坍缩）。两者均作为纯 potential 层的真实评估器实现（`calc_energy`/`calc_forces`），对核心数据模型零耦合。这是 CL&Pol 三段式规格链的第 02 段，依赖第 01 段（clpol-01-virtualsite-drude）；本段消费第 01 段产出的极化率数据（per-atom `alpha`、`a_thole`），以入参形式传入。

## Domain basis

两个阻尼形式逐字采用 scientist agent 校验过的、与官方 paduagroup/clandpol 及 LAMMPS `pair_style thole` / `pair_coul_tt` 一致的闭式表达。

1. Thole 阻尼（指数 Thole 形式）：
   - `T_ij(r) = 1 - (1 + s_ij r / 2) exp(-s_ij r)`，`s_ij = a_ij / (alpha_i alpha_j)^(1/6)`，`a_ij = (a_i + a_j)/2`。
   - 单位：`r` 为 Å，`alpha` 为 Å^3，`a` 无量纲。默认 `a = 2.6`，cutoff 12.0 Å。
   - 参考：Thole, Chem. Phys. 59 (1981) 341，DOI 10.1016/0301-0104(81)85176-2。

2. Tang-Toennies 阻尼：
   - `f_n(r) = 1 - c exp(-b r) sum_{k=0}^{n} (b r)^k / k!`。
   - CL&Pol 规范：`n = 4`，`b = 4.5` (1/Å)，`c = 1.0`，cutoff 12.0 Å。
   - 参考：Tang & Toennies, JCP 80 (1984) 3726，DOI 10.1063/1.447150。

阻尼因子乘以一对原子的库仑能量得到被阻尼能量；长程 `T_ij -> 1`、`f_n -> 1`，短程 `r -> 0` 阻尼显著（`-> 0`）。

## Design

新增两个 `PairPotential` 子类，遵循现有 pair 评估器模式（`lj.py` 的 `LJ126`：`calc_energy(dr, dr_norm, pair_types)`、`calc_forces(..., pair_idx, n_atoms)`，numpy 按类型索引参数）。阻尼依赖两端原子的极化率，故能量/力方法取分离的端点类型索引 `pair_types_i`/`pair_types_j`。

- `Thole(charge, alpha, a_thole)`：`damping(dr_norm, ti, tj)` 返回 `T_ij`；`calc_energy` 求和 `T_ij q_i q_j / r`；`calc_forces` 为能量的负梯度（解析力 `T'(r)=(s/2)(1+x)e^{-x}`，FD 校验）。
- `TangToennies(charge, b=4.5, n=4, c=1.0)`：`damping(dr_norm)` 返回 `f_n`；`calc_energy` 求和 `f_n q_i q_j / r`；`calc_forces` 负梯度（解析力 `f'(r)=c b e^{-br}(br)^n/n!`，FD 校验）。

空 pair 早退：能量 `0.0`，力 `zeros((n_atoms,3))`。仅依赖 numpy，对 core 零导入。

## Files to create or modify

- `src/molpy/potential/pair/thole.py` (new) — `Thole`
- `src/molpy/potential/pair/tang_toennies.py` (new) — `TangToennies`
- `src/molpy/potential/pair/__init__.py` — 导出 + `__all__`
- `tests/test_potential/test_pair/test_thole.py`、`test_tang_toennies.py` (new)

## Tasks

- [x] Write failing tests for Thole evaluator
- [x] Implement Thole in src/molpy/potential/pair/thole.py (damping factor, calc_energy, calc_forces)
- [x] Write failing tests for TangToennies evaluator
- [x] Implement TangToennies in src/molpy/potential/pair/tang_toennies.py (damping factor, calc_energy, calc_forces)
- [x] Register Thole and TangToennies in src/molpy/potential/pair/__init__.py and __all__
- [x] Add Google-style docstrings with units and DOI refs for both classes
- [x] Verify Thole T_ij and TT f_n against closed form, force vs finite-difference, and r->inf / r->0 limits
- [x] Run full check + test suite

## Testing strategy

闭式复现（Thole 默认 a=2.6；TT b=4.5,c=1.0,n=4，容差 1e-10）；解析力 == 能量有限差分（rtol 1e-5）；长程极限 `T,f -> 1`（1e-6）；短程单调强阻尼（`r->0` 时 `-> 0`）；空 pair 早退；导入性。

## Out of scope

- Drude 数据模型与 virtual-site 装配（第 01 段）；scaleLJ（第 03 段）；导出 / LAMMPS 写出；极化率数据的读取或推导（第 01 段提供）。
