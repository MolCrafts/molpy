---
title: Repoint Python + molpy + bench consumers onto raw-compute + fit composition
status: approved
created: 2026-06-18
---

# Repoint Python + molpy + bench consumers onto raw-compute + fit composition

## Summary

把 phase 01 在 molrs 里新增的「裸 compute + 显式 fit」组合通过 PyO3 暴露到 Python，并把 molpy 的 compute 包装层和 bm-molrs-molpy 的 freud-parity 基准全部重指到这条新路径上——全程保持**非破坏性**：旧的自由函数绑定（`dielectric_einstein_helfand_conductivity`、`transport_green_kubo_conductivity`、power/ir/raman 等）原样保留，仅加 `DeprecationWarning` 与弃用 docstring，真正删除留给 phase 03。用户可见的 Python API（如 `conductivity()` 便捷函数）尽量不变，但内部实现改为「裸 compute → 显式 fit」的组合。本阶段的核心验收是**数值回归锁**：新路径产出的 σ / D / 谱必须在既有容差内复现旧路径的结果。

## Domain basis

本阶段不引入新物理，沿用 phase 01 已建立并经 `.claude/notes/science.md` 约束的传输/介电公式与单位（时间 fs、频率 cm⁻¹、温度 K、电导率沿用 molrs real units）。回归锁所依据的等价关系：

- Einstein–Helfand 电导率：σ 由平动偶极 MSD 的线性段斜率经 `LinearFit` 给出，等价于旧 `einstein_helfand_conductivity` 内联的 `fit_start_frac`/`fit_end_frac` 线性拟合。
- Green–Kubo 电导率：σ 为电流 ACF 的时间积分（`RunningIntegral` → `Plateau`），等价于旧 `green_kubo_conductivity` 的 `sigma_running` 末值。
- Einstein / Green–Kubo 自扩散：D 经同样的「裸 MSD/VACF + LinearFit/RunningIntegral」组合得到。
- 振动谱（VDOS / IR / Raman）：`PowerSpectrum`/`IRSpectrum`/`RamanSpectrum` 变换等价于旧自由函数 `power_spectrum`/`ir_spectrum`/`raman_spectrum`（频率换算常数见 `molrs/src/compute/spectra/mod.rs:52` `ANGULAR_FREQ_TO_CM1`）。

参考实现交叉校验（bench 侧，沿用既有库）：freud（结构/扩散）、scipy（谱/积分）、MDAnalysis（传输）、LAMMPS（电导率参考输出）。freud 内部为 float32——**保留既有双预算断言（two-budget floors），不收紧为 value-range atol**（见 MEMORY: freud-parity floors）。

## Design

本阶段是**接线层**改动，不动 molrs Rust 数值核心（phase 01 已落地 `compute::fit`）。三个被重指的消费层：

- **molrs-python（PyO3）**：为 phase 01 的裸 compute 结构（VACF、EinsteinDiffusion、GreenKuboDiffusion、EinsteinConductivity、GreenKuboConductivity、DebyeRelaxation）与 fit 类（LinearFit、RunningIntegral、Plateau、DebyeFit、PowerSpectrum/VDOS、IRSpectrum、RamanSpectrum）新增 `#[pyclass]` 绑定，注册到根 `molrs` 模块（沿用 `lib.rs` 现有 `register_*` 模式）。**所有现存自由函数绑定保留**，但在其 docstring 标注弃用并在调用时 `warnings.warn(..., DeprecationWarning)`。弃用 warning 的发射位置以不改变返回值/dict 形状为前提。
- **molpy（thin wrapper）**：把 compute 包装类改为薄继承 / 委托到新 molrs 裸 compute + fit 类（遵循 core-sinks-to-molrs 规则，molpy 不得自带拟合/谱数学）。把 molpy 自带的 ad-hoc `DebyeFit`（`molpy/src/molpy/compute/result.py:168`）改为委托 molrs `DebyeFit`。对外便捷 helper（如 `conductivity()`）保持签名，内部实现为「裸 compute + fit」的组合。
- **bm-molrs-molpy（pytest-benchmark）**：把电导率/扩散/谱基准改为先调裸 compute 再 fit；每个 kernel 仍配一条对参考库的等价检查；保留 freud-parity 的双预算 floor。

数据契约不变：Python 侧仍以 numpy 数组进/出，dict / 标量出参形状与旧路径一致（回归锁要求）。precision 沿用 molrs 全程 f64（注意 `molrs-python/src/lib.rs` 顶部 docstring 关于 f32 的描述已过时，绑定实际走 f64；本阶段不修该 docstring，留给 docs）。

新增公开 Python 符号（在 `molrs` 顶层命名空间）：`VACF`、`EinsteinDiffusion`、`GreenKuboDiffusion`、`EinsteinConductivity`、`GreenKuboConductivity`、`DebyeRelaxation`、`LinearFit`、`RunningIntegral`、`Plateau`、`DebyeFit`、`PowerSpectrum`、`IRSpectrum`、`RamanSpectrum`。命名与 phase 01 的 method-first OOP compute 命名对齐。

## Files to create or modify

- `molrs-python/src/transport.rs` — 新增 EinsteinDiffusion / GreenKuboDiffusion / EinsteinConductivity / GreenKuboConductivity / VACF 的 `#[pyclass]` 绑定 + fit 类委托；旧自由函数加弃用。
- `molrs-python/src/dielectric.rs` — DebyeRelaxation 的 `#[pyclass]` 绑定；旧 `dielectric_*` 自由函数加弃用 docstring + `DeprecationWarning`。
- `molrs-python/src/compute_extra.rs` — fit 类（LinearFit、RunningIntegral、Plateau、DebyeFit、PowerSpectrum/VDOS、IRSpectrum、RamanSpectrum）的 `#[pyclass]` 绑定。
- `molrs-python/src/validate.rs` — 若 phase 01 的裸 compute + fit 改变了校验入参的来源，调整 validate 绑定以接受新路径产物（仅在必要时；否则保持不变）。
- `molrs-python/src/lib.rs` — 在 `#[pymodule] fn molrs` 注册新 pyclass / register 调用。
- `molpy/src/molpy/compute/dielectric.py` — 重指到 molrs 裸 compute + fit。
- `molpy/src/molpy/compute/jacf.py` — 重指 Green–Kubo 电导率到 VACF/电流 ACF 裸 compute + RunningIntegral/Plateau。
- `molpy/src/molpy/compute/msd.py` — 重指 Einstein 扩散/电导率到裸 MSD + LinearFit。
- `molpy/src/molpy/compute/onsager.py` — 重指到 molrs onsager 裸 compute + fit。
- `molpy/src/molpy/compute/mcd.py` — 重指到新路径。
- `molpy/src/molpy/compute/persist.py` — 重指到新路径（若涉及拟合）。
- `molpy/src/molpy/compute/result.py` — 把 ad-hoc `DebyeFit`（:168）改为委托 molrs `DebyeFit`。
- `molpy/src/molpy/compute/decomposition.py` — 重指到新路径。
- `bm-molrs-molpy/` — 电导率/扩散/谱基准改为 raw-compute → fit 组合，保留双预算 floor 与参考库等价检查（具体基准文件由 bench 套件现有布局确定，逐 kernel 修改）。

## Tasks

- [ ] Write failing parity tests for new PyO3 raw-compute + fit bindings (molrs-python/tests/test_fit_repoint.py)
- [ ] Implement EinsteinConductivity/GreenKuboConductivity/EinsteinDiffusion/GreenKuboDiffusion/VACF/DebyeRelaxation pyclass bindings in molrs-python/src/{transport.rs,dielectric.rs}
- [ ] Implement LinearFit/RunningIntegral/Plateau/DebyeFit/PowerSpectrum/IRSpectrum/RamanSpectrum pyclass bindings in molrs-python/src/compute_extra.rs and register in lib.rs
- [ ] Add DeprecationWarning + deprecated docstrings to existing free-function bindings in molrs-python/src/{dielectric.rs,transport.rs,compute_extra.rs} without changing return shapes
- [ ] Rebuild maturin wheel and verify import of new molrs classes
- [ ] Write failing regression tests pinning molpy wrapper outputs to pre-migration values (molpy/tests/compute/test_fit_repoint.py)
- [ ] Repoint molpy compute wrappers in molpy/src/molpy/compute/{dielectric.py,jacf.py,msd.py,onsager.py,mcd.py,persist.py,decomposition.py} to delegate to molrs raw-compute + fit
- [ ] Migrate molpy ad-hoc DebyeFit (molpy/src/molpy/compute/result.py:168) to delegate to molrs DebyeFit
- [ ] Repoint bm-molrs-molpy conductivity/diffusion/spectra benches to raw-compute + fit, preserving two-budget freud-parity floors
- [ ] Verify new-path sigma/D/spectrum reproduce old-path values within documented tolerance and reference-library equality holds
- [ ] Run full check + test suite

## Testing strategy

- **Happy path (bindings)**: every new pyclass (`EinsteinConductivity`, `GreenKuboConductivity`, `EinsteinDiffusion`, `GreenKuboDiffusion`, `VACF`, `DebyeRelaxation`, and all fit classes) constructs, runs `compute()` (raw curve), then is composed with the matching fit to yield a coefficient; output dtype is float64 and shapes match the legacy dict fields.
- **Regression lock (core)**: for a fixed input fixture, the new explicit pipeline (raw compute → fit) reproduces the old bundled coefficient (σ / D / spectrum) bit-for-bit where the arithmetic is identical, or within the documented float tolerance where ordering differs. Asserted on both the molrs-python layer and the molpy wrapper layer.
- **Deprecation**: calling each legacy free-function binding emits exactly one `DeprecationWarning` and returns the unchanged dict/scalar (caught via `pytest.warns`).
- **molpy thin-inheritance**: each migrated wrapper's public output equals its pre-migration output within existing tolerance; no fitting/spectra math remains implemented in molpy (delegation only); `DebyeFit` results match between molpy-delegated and direct molrs `DebyeFit`.
- **Wheel prerequisite**: maturin wheel is rebuilt after the Rust binding changes; molpy integration tests run green against the freshly built wheel (stale wheel must not mask the repoint).
- **Domain validation (reference libraries)**: bench equality checks — Green–Kubo / Einstein conductivity vs LAMMPS reference output; diffusion vs MDAnalysis / freud; VDOS/IR/Raman vs scipy. freud-parity benches still pass their existing two-budget floors (freud is float32 — budgets unchanged, not tightened to value-range atol).
- **Edge cases**: empty / single-frame input still raises the same error as the legacy path; a raw compute with no fit returns only the curve (no coefficient) and does not silently fabricate σ/D.

## Out of scope

- Removing the deprecated bindings, bundled-result fields, or legacy free functions — that is phase 03.
- Adding any new physics, compute, or fit kernel — that landed in phase 01.
- Tightening freud-parity tolerances to value-range atol (the float32 two-budget floors are deliberate; see MEMORY note).
- Fixing the stale f32 precision note in `molrs-python/src/lib.rs` docstring (docs task, not this repoint).
- Cross-repo release / version bumps of molrs, molpy, or the bench package.
