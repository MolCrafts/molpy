---
title: scaleLJ — CL&Pol SAPT-derived Lennard-Jones epsilon scaling
status: code-complete
created: 2026-06-10
---

# scaleLJ — CL&Pol SAPT-derived Lennard-Jones epsilon scaling

## Summary

为已有的非极化 CL&P/OPLS-AA 力场提供 `scaleLJ` 操作：按片段对 (fragment pair) 用 SAPT 推导的 CL&Pol 因子 `k_ij` 缩小 Lennard-Jones 阱深 `epsilon`，以扣除在显式 Drude 极化加入后会被重复计入的诱导能。这是 CL&Pol 三段链的第 03 段，作用于 clpol-01 产出的（待极化）体系；本段纯属数据模型/力场参数操作，不涉及任何导出。调用后返回一份新的 `ForceField`，其 `PairType` 的 `epsilon` 被缩放，`sigma`、原子电荷与离子整数电荷保持不变；输入力场不被修改。

## Domain basis

逐片段对的 LJ epsilon 缩放因子（严格采用 paduagroup/clandpol `scaleLJ` 源码形式，而非 README 中宽松写法）：

```
k_ij = 1 / [ 1 + C0 * r_ij^2 * q^2 / alpha  +  C1 * mu^2 / alpha ]
```

- `q^2/alpha`（电荷-诱导偶极）项带有 COM 距离前因子 `r_ij^2`；`mu^2/alpha`（偶极-诱导偶极）项**不带** `r_ij^2` 前因子（对 README 的修正，以 `scaleLJ` 源码为准）。
- 常数（`scaleLJ` 源码）：`C0 = 0.254952`，`C1 = 0.106906`。
- 诱导项对片段对中**非极化**的那个片段求和（非对称：每个非极化片段各加一项）。
- 应用方式：`epsilon_ij_new = k_ij * epsilon_ij`，`k_ij <= 1`，减小阱深；`sigma` 不缩放（源码提供一个可选 flag 将 `sigma` 乘 0.985 用于特例，默认关闭）。

每个片段的标度数据（`fragment.ff`）：电荷 `q` (e)、偶极矩 `mu` (Debye)、极化率 `alpha` (Å³)；片段间 COM 距离 `r_COM` (Å) 在运行时由结构几何计算。

物理依据 (SAPT)：非极化 CL&P/OPLS-AA 的 `epsilon` 是对凝聚相总相互作用拟合得到的，隐含诱导能；显式 Drude 极化会重新引入诱导能，故 `epsilon` 必须减小。

参考：paduagroup/clandpol `scaleLJ` + `fragment.ff`；理论依据 Goloviznina et al., J. Chem. Theory Comput. 15 (2019) 5858，DOI 10.1021/acs.jctc.9b00689。

## Design

新增 `core/ops/scale_lj.py`，提供公共操作 `scale_lj(...)`（builder/op 层的纯函数式算子）。

- **片段数据加载**：`load_fragment_scaling_data(path) -> dict[str, FragmentScaling]`，从提交的 `src/molpy/data/forcefield/clpol_fragments.ff` 转录 clandpol `fragment.ff`，每个片段记录 `q`、`mu`、`alpha`、`polarizable: bool`。`FragmentScaling` 为 frozen dataclass。
- **片段划分（v1 设计决策，需固定）**：采用**用户提供的映射**或**按 Atomistic per-molecule** 的片段映射 —— `fragments: Mapping[fragment_label, Sequence[Atom]]`（或 per-molecule 默认）。不做自动片段感知。每个 fragment_label 必须能在片段标度数据中查到。
- **k_ij 计算**：`compute_k_ij(frag_i, frag_j, r_ij, data) -> float`，对片段对中每个非极化片段累加诱导项，按上式返回 `k_ij ∈ (0, 1]`。`r_ij` 取两片段原子坐标的 COM 距离。
- **epsilon 写回**：`scale_lj(ff, struct, fragments, *, scale_sigma=False) -> ForceField`。先深拷贝 `ff`（因 `Style.copy()` 不拷贝其 `types`，必须对每个 `PairStyle` 重建并 `PairType.copy()` 后逐一缩放，保证输入 FF 不被修改），按片段对查得对应 `PairType`，将 `epsilon` 乘以 `k_ij` 写回拷贝。`sigma`、`charge` 及任何原子电荷不动（`scale_sigma=True` 时仅将 `sigma` 乘 0.985）。
- **不变量**：原子整数离子电荷绝不改动；本算子只触碰 `PairType` 的 `epsilon`（与可选 `sigma`）。

## Files to create or modify

- `src/molpy/core/ops/scale_lj.py` (new) — `FragmentScaling`、`load_fragment_scaling_data`、`compute_k_ij`、`scale_lj`。
- `src/molpy/core/ops/__init__.py` — 导出 `scale_lj`、`load_fragment_scaling_data`、`compute_k_ij`、`FragmentScaling`。
- `src/molpy/data/forcefield/clpol_fragments.ff` (new) — 从 clandpol `fragment.ff` 转录的每片段 `q`、`mu`、`alpha`、`polarizable` 数据。
- `tests/test_core/test_ops/test_scale_lj.py` (new) — 算子与科学验证测试。

## Tasks

- [x] Write failing tests for scale_lj operator (tests/test_core/test_ops/test_scale_lj.py)
- [x] Transcribe clandpol fragment.ff into src/molpy/data/forcefield/clpol_fragments.ff (new) with per-fragment q, mu, alpha, polarizable
- [x] Implement FragmentScaling dataclass and load_fragment_scaling_data in src/molpy/core/ops/scale_lj.py
- [x] Implement compute_k_ij in src/molpy/core/ops/scale_lj.py (mu^2 term carries no r^2 prefactor; C0=0.254952, C1=0.106906)
- [x] Implement scale_lj in src/molpy/core/ops/scale_lj.py (deep-copy ff, scale PairType epsilon, leave sigma and charges unchanged)
- [x] Export scale_lj, compute_k_ij, load_fragment_scaling_data, FragmentScaling from src/molpy/core/ops/__init__.py
- [x] Add Google-style docstrings with units (q in e, mu in Debye, alpha in Å^3, r in Å, epsilon in kcal/mol)
- [x] Verify k_ij against closed form and clandpol reference for a known fragment pair
- [x] Run full check + test suite

## Testing strategy

- Happy path：构造含两已知片段的小 `Atomistic` + `ForceField`（带 `PairStyle`/`PairType`），调用 `scale_lj`，断言返回的 FF 中目标 `PairType.epsilon` 等于 `k_ij * 原 epsilon`。
- Copy 不变量：断言返回 FF 与输入 FF 非同一对象，输入 FF 所有 `PairType.epsilon`/`sigma` 未变；任一原子电荷未变。
- 缩放范围：断言所有 `k_ij ∈ (0, 1]`，即 epsilon 严格不增。
- sigma 行为：默认 `scale_sigma=False` 时 sigma 不变；`scale_sigma=True` 时 sigma == 0.985 * 原 sigma。
- Edge cases：片段对全极化（无诱导项 → `k_ij == 1`，epsilon 不变）；片段 label 缺失于标度数据 → 明确报错；`alpha` 缺失/为零 → 明确报错（避免除零）。
- 域验证：(a) 对已知 `q, mu, alpha, r` 解析复算 `k_ij` 在容差内重合；(b) **显式测试** `mu^2` 项不含 `r^2` 前因子 —— 改变 `r` 时，仅 `q^2` 项随 `r^2` 变化，`mu^2` 项的贡献不变；(c) 若可取得 clandpol 参考值，对一个完全非极化→极化的已知体系比对 epsilon 缩放结果在容差内。

## Out of scope

- Drude 数据模型（phase 01）。
- Thole / Tang-Toennies 势（phase 02）。
- 任何力场/结构导出。
- 超出简单 per-molecule / 用户提供映射的自动片段感知。
- 超出 CL&P 首版范围的离子族。
