---
title: CL&Pol Phase 01 — Virtual-Site Data Model, Augmentation Transform, Drude Polarizer
status: code-complete
created: 2026-06-10
---

# CL&Pol Phase 01 — Virtual-Site Data Model, Augmentation Transform, Drude Polarizer

## Summary

本阶段在 molpy 内部建立虚位点（virtual site）的数据模型、通用的"按规则给结构增补虚位点"变换基类，以及具体的 Drude 极化器。完成后，用户可以把一个经 CL&P 类型化（依赖 clp-typifier-02-forcefield）的 `Atomistic` 转换成一个 Drude 可极化体系：每个可极化的重原子获得一个 `DrudeParticle` 壳层和一根 core-shell 谐振键，电荷守恒、离子总电荷保持整数。本阶段全部工作发生在 molpy 内部，**不涉及任何导出 / engine 发射**。同时交付一个 `Tip4pBuilder` 以证明增补基类不是 Drude 专用，可服务刚性几何位点（TIP4P M 位点 / 孤对电子）。

## Domain basis

CL&Pol Drude 振子模型（取自 scientist agent 核验，逐字保留）：

- Drude 关系：`alpha = q_D^2 / (4*pi*eps0 * k_D)`（molpy 单位）。CL&Pol 固定 `k_D`，由输入 `alpha` 反推 `q_D = sqrt(4*pi*eps0 * k_D * alpha)`，`4*pi*eps0 = 0.0007197587 e^2/(kJ/mol·Å)`。
- 规范常数（paduagroup/clandpol `alpha.ff`）：`k_D = 4184.0 kJ/mol/Å^2`，`U = (1/2) k_D r^2` 形式。Drude 电荷取负号（壳层 `-|q_D|`）。`m_D = 0.4 u`，从核扣除（`m_core = m_atom - m_D`）。核电荷守恒 `q_core = q_atom - q_D`，离子总电荷保持整数（±1）。
- **FACTOR-OF-2（已核验）**：`harmonic.py:56` 为 `0.5*k*(r-r0)^2`，molpy 谐振键即 `U=1/2 K r^2`；Drude 弹簧直接写 `K = k_D = 4184`（不除 2）。
- 只有可极化的重原子获得 Drude（`alpha.ff` 中 `k_D > 0`）；氢原子 `k_D = 0`（虽带 `alpha=0.323` 仅供 Thole），故选择判据是 `k_D > 0` 而非 `alpha > 0`。
- Drude 弹簧复用现有 `BondHarmonic`，不新写弹簧势。
- CL&Pol 不缩放部分电荷（完整整数离子电荷 + 显式极化）。
- TIP4P：M 位点放在 HOH 角平分线上的无质量位点，O 负电荷转移到 M，无弹簧（刚性几何放置）——与 Drude（共置壳层 + 弹簧）对比；基类同时容纳两类。

参考：Goloviznina, Canongia Lopes, Costa Gomes, Pádua, JCTC 2019, 15, 5858, DOI 10.1021/acs.jctc.9b00689；Lamoureux & Roux, JCP 2003, 119, 3025, DOI 10.1063/1.1589749。

## Design

- **Core**：`VirtualSite(Atom)`（持久 `vsite` 标记字段；molrs 重新实例化为普通 `Atom`，故身份靠字段而非子类，经 `Atom.is_virtual` 读取），`DrudeParticle(VirtualSite)`、`MasslessSite(VirtualSite)`，数据-only。
- **Builder**：`VirtualSiteBuilder`(ABC) 模板方法 `apply = copy->select->build_sites->redistribute`，返回新结构不改入参；`DrudeBuilder`（CL&Pol 极化器）、`Tip4pBuilder`（证明基类通用）。
- **数据**：`alpha.ff`（CL&P 类型 → m_D/q_D 符号/k_D/alpha/a_thole），转录自 paduagroup/clandpol，经 `get_forcefield_path` 解析。

## Files to create or modify

- `src/molpy/core/atomistic.py` — VirtualSite/DrudeParticle/MasslessSite + Atom.is_virtual
- `src/molpy/core/__init__.py`、`src/molpy/__init__.py` — 导出
- `src/molpy/builder/virtualsite.py` (new) — VirtualSiteBuilder/DrudeBuilder/Tip4pBuilder/load_polarizability
- `src/molpy/builder/__init__.py` — 导出
- `src/molpy/data/forcefield/alpha.ff` (new)
- `tests/test_core/test_virtualsite.py`、`tests/test_builder/test_virtualsite.py` (new)

## Tasks

- [x] Write failing tests for VirtualSite/DrudeParticle/MasslessSite data model and molrs add_atom injection
- [x] Implement VirtualSite/DrudeParticle/MasslessSite as Atom subclasses and export
- [x] Verify Atomistic.add_atom / _spawn_entity accepts a VirtualSite subclass view and round-trips the virtual marker
- [x] Verify molpy BondHarmonic energy convention (U = 1/2 K r^2) and record Drude spring K = k_D = 4184 (no division by 2)
- [x] Transcribe paduagroup/clandpol alpha.ff to src/molpy/data/forcefield/alpha.ff resolvable via get_forcefield_path
- [x] Write failing tests for DrudeBuilder and Tip4pBuilder
- [x] Implement VirtualSiteBuilder ABC (apply = copy->select->build_sites->redistribute, no input mutation)
- [x] Implement DrudeBuilder: derive q_D from alpha, add DrudeParticle + core-shell BondHarmonic, conserve charge and mass
- [x] Implement Tip4pBuilder: emit MasslessSite on HOH bisector, move O charge to M, no spring; export both builders
- [x] Run full check + test suite

## Testing strategy

Atom typing/charge/mass conservation on CL&P-typed [C4C1im]+; Drude count == polarizable-heavy count (k_D>0), H excluded; core-shell spring K=4184; alpha recovered from q_D,k_D; input not mutated; Tip4p M-site on bisector. Full ruff/ty/pytest gate.

## Out of scope

- Thole / Tang–Toennies 势（phase 02）；scaleLJ（phase 03）；任何 LAMMPS/OpenMM 导出；电荷缩放变体；CL&P 首版范围以外离子族。
