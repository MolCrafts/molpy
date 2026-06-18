---
title: 计算与拟合分离 — compute::fit 模块 + 原始输出 (Phase 01)
status: approved
created: 2026-06-18
---

# 计算与拟合分离 — compute::fit 模块 + 原始输出 (Phase 01)

## Summary

把 molrs 的「计算」（compute）与「拟合/平滑/谱变换」（fit）彻底解耦，本阶段以**非破坏式新增**落地第一步：在 `molrs/src/compute/` 下新增与 `Compute` 并列的 `Fit` trait 和 `compute::fit` 子模块，并新增一批**只返回原始曲线**的 OOP 计算结构（`VACF`、`EinsteinDiffusion`、`GreenKuboDiffusion`、`EinsteinConductivity`、`GreenKuboConductivity`、`DebyeRelaxation`）。`Fit` 消费上游计算结果（`Array1` 曲线 / 原始 ACF），而非帧；用户显式调用 `LinearFit { window }.fit(msd)` 等步骤获得标量或谱。现有的自由函数（`einstein_helfand_conductivity`、`green_kubo_conductivity`、`power_spectrum`/`ir_spectrum`/`raman_spectrum`、`dielectric` 谱函数）本阶段一律保留、并行共存，不删除任何字段或函数（移除属 Phase 03）。本阶段不动 Python/bench（属 Phase 02）。

> 与已批准的 `vibrational-spectra` spec 的关系（仅说明，不修改该 spec）：该 spec 的 `power_spectrum`/`ir_spectrum`/`raman_spectrum` 当前把「窗+FFT+物理前因子」与「ACF 计算」捆在一个函数里。本阶段把窗+FFT 抽成可分离的谱变换 `Fit`（输入是原始 ACF），从而让「谱 = 原始 ACF 经窗/FFT 变换」成为一条独立、用户参数化的步骤。该 spec 的旧函数本阶段仍原样保留并继续作为回归基准；其谱输出契约的正式收敛（原始计算只返回 ACF）留待 Phase 03。

## Domain basis

原始/派生分离对下列所有路线在物理上**无损**，当且仅当满足三条不变量（编码为验收标准 (a)/(b)/(c)）：

- **(a) 延迟充分性**：原始计算的 `max_lag` 必须 ≥ 下游 fit/变换所消费的延迟范围；截断到更短的 ACF/曲线会丢信息。
- **(b) 非归一化契约**：原始 ACF 必须**非归一化**返回（或「曲线 + 零延迟方差」配套）。归一化的 Debye Φ(t) 只给出弛豫形状/τ，**给不出**幅度 `ε0 − ε∞`——幅度来自 ⟨M²⟩（Neumann/Kirkwood 涨落）。
- **(c) 标量元数据随结果携带**：体积 V、温度 T、Ewald 边界条件、电流归一化约定（总电流 `J = Σ qᵢ vᵢ` vs 密度）必须作为标量元数据挂在原始结果上；缺了它们派生步骤就是错的。

物理前因子（编码为 `Fit` 实现内部常量与验收）：

- Einstein 扩散：`D = slope / (2d)`，`d = 3 → slope/6`。
- Green–Kubo 扩散：`D = (1/d) ∫ VACF dt`。
- Einstein–Helfand 电导：`σ = slope / (2d·V·k_B·T)`，`d = 3 → slope/6`。
- Green–Kubo 电导：`σ = (1/(d·V·k_B·T)) ∫ ⟨J(0)·J(t)⟩ dt`，`d = 3 → 1/3`；`J = Σ qᵢ vᵢ` 总电流约定。
- Nernst–Einstein：`σ_NE = (1/(V·k_B·T)) Σᵢ Nᵢ qᵢ² Dᵢ`，消费每种自扩散标量 `Dᵢ`——它是「派生之派生」，建模为**消费 Dᵢ 标量**的 `Fit`，而非消费原始曲线。（注：`NernstEinsteinConductivity` 本身属 Phase 后续 OOP 命名收口，本阶段只需 `Fit` trait 的签名能容纳该形态，不要求实现该具体 Fit。）
- Debye：`ε*(ω) = ε∞ + (ε0 − ε∞)/(1 + iωτ)`，且 `(ε*(ω) − ε∞)/(ε0 − ε∞) = 1 − iω ∫ Φ(t) e^{−iωt} dt`。
- 谱：窗在 FFT 前作用于 ACF，是可分离的泄漏抑制步骤；`S(ω) = FT[w(t)·C(t)]` 完全由原始 ACF + 窗/FFT 参数决定。
- Raman：各向同性 (1/15) + 各向异性前因子，谐振量子因子 `Q_HA = ℏω / [k_B T (1 − exp(−ℏω/k_B T))]`。

参考文献：

- France-Lanord & Grossman, *Phys. Rev. Lett.* **122**, 136001 (2019); arXiv:1812.04772 —— Einstein D、Einstein–Helfand σ、Nernst–Einstein。
- Thomas, Brehm, Fligg, Vöhringer, Kirchner, *Phys. Chem. Chem. Phys.* **15**, 6608 (2013); DOI 10.1039/c3cp44302g —— IR/Raman ACF→FT、Bose 因子、窗-先于-FFT。
- *J. Chem. Phys.* **159**, 134505 (2023); DOI 10.1063/5.0166788 —— Debye 弛豫。
- Frenkel & Smit, *Understanding Molecular Simulation*, 2nd ed., §4.4 —— GK/Einstein。
- Hansen & McDonald, *Theory of Simple Liquids*, 4th ed., Ch.10 —— 电流-电流关联。

## Design

### `Fit` trait（与 `Compute` 并列）

新增 `Fit` trait：消费上游计算结果（一条 `Array1<f64>` 曲线、或一个带元数据的原始结果），产出拟合/变换结果。签名约为：

```rust
pub trait Fit {
    /// 上游输入：原始曲线或原始结果引用（非帧）。
    type Input<'a>;
    /// 拟合/变换输出。
    type Output: ComputeResult + Clone + Send + Sync + 'static;
    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError>;
}
```

- `&self` 是不可变参数包（window、prefactor、d 维度等），与 `Compute` 同约定：相同输入 + 相同 `&self` ⇒ 相同输出，无隐藏可变状态。
- `Input` 用 GAT 携带借用，以便消费 `&Array1<f64>` 或 `&JacfResult` 之类的上游结果。
- 复用现有 `ComputeError`；不新增错误枚举（degenerate 窗用 `OutOfRange`，维度不匹配用 `DimensionMismatch`，空输入用 `EmptyInput`）。

### `compute::fit` 子模块（一文件一 fit 实现，镜像 `compute/{rdf,msd,...}` 布局）

数值拟合：

- `LinearFit { window: (f64, f64) }` → `LinearFitResult { slope, intercept, r2, fit_start, fit_end }`。OLS 斜率逻辑**抽取自** `compute/dielectric.rs` 当前内联实现（`einstein_helfand_conductivity` 的 `fit_start..=fit_end` 最小二乘块，约 dielectric.rs:1029–1047）；`window` 为 `(start_frac, end_frac)`，语义沿用现有 `fit_start_frac/fit_end_frac`，并补算 `r2`。
- `RunningIntegral`（梯形）→ 累积积分曲线 `Array1<f64>`。梯形逻辑**抽取自** `compute/jacf.rs` 当前内联实现（jacf.rs:159–166 的 running 梯形积分）。
- `Plateau { window: (f64, f64) }` → 窗内平台均值标量（含样本数 / 标准差），用于 GK 积分平台读取。
- `DebyeFit` → 指数/Debye 弛豫拟合（τ + 幅度），消费归一化 Φ(t) 给 τ、配合 ⟨M²⟩ 元数据给幅度；**统一收口** molpy 侧 ad-hoc 的 `DebyeFit`（sibling 仓 `molpy/src/molpy/compute/result.py` 中的实现，作为参照，不在本仓内）。

谱变换拟合（输入为原始 ACF，**组合** `molrs::signal` 的 window/acf/grid 原语，绝不重写窗函数）：

- `PowerSpectrum`（即 VDOS）、`IRSpectrum`、`RamanSpectrum` 的 `Fit` 实现：对原始 ACF 施加 窗 + FFT(+ 物理前因子，Raman 含 Bose/谐振量子因子)。变换体**抽取自** `compute/spectra/mod.rs::window_and_fft`（及其内部 `acf_to_spectrum`/`acf_to_intensities`/`cosine_sq_window`/`bose_factor`）与 `compute/dielectric.rs::{windowed_acf_spectrum, acf_to_spectrum}`。这些谱 `Fit` 必须能**逐位复现**对应旧函数在「同一 ACF + 同窗/FFT 参数」下的输出（回归锁）。

> 为了让谱 `Fit` 与旧自由函数共享同一段窗+FFT 代码而非复制，本阶段把 `window_and_fft` / `acf_to_spectrum` 系列下沉为 `compute::fit` 内的共享私有 helper，旧的 `spectra::*` 与 `dielectric::*` 自由函数改为调用这些 helper（纯内部重构，公开签名与数值不变）。

新的原始可观测 OOP 计算（实现 `Compute`，**只返回原始曲线**，旧自由函数本阶段不动）：

- `VACF` → 原始非归一化速度 ACF；**抽取自** `compute/spectra/power_spectrum.rs` 内部的 `acf_sum`（逐 DOF 去均值 ACF 求和、归一化前的那一段）。
- `EinsteinDiffusion` → 原始自 MSD；委托现有 `compute/msd` 的 MSD 原语（`MSD::windowed()`），不重算。
- `GreenKuboDiffusion` → 原始 VACF（与 `VACF` 同曲线，按扩散语义包装 / 复用）。
- `EinsteinConductivity` → 原始集体电荷-偶极 MSD（即 `ConductivityResult` 当前捆绑的 `lag_times + msd` 原始部分）。
- `GreenKuboConductivity` → 原始电流 ACF（即 `JacfResult` 当前捆绑的 `lag_times + jacf` 原始部分）。
- `DebyeRelaxation` → 原始偶极 ACF **外加** 零延迟方差 ⟨M(0)²⟩ 与 V/T/Ewald-BC 元数据（Debye 幅度所需，落实不变量 (b)/(c)）。

### 归一化归属

`ComputeResult::finalize` 内的**归一化仍属原始计算**，不是拟合：RDF 的 g(r)、密度的体积归一、COM 的质量加权都不是 fitting，保持在各自 `Compute` 内。`Fit` 只做斜率/积分/平台/谱变换。

### 生命周期与所有权

所有 `Fit` 与新 `Compute` 实现均无状态：`&self` 只读参数包，输入只读借用，输出新分配并满足 `ComputeResult + Clone + Send + Sync + 'static`。`compute::fit` 子模块在 `compute/mod.rs` 中 `pub mod fit;` 并 re-export 公共类型；`Fit` trait 在 `compute/traits.rs` 与 `Compute` 并列定义（或新增 `compute/fit/traits.rs`，由实现者择一，re-export 一致）。

> 注：本阶段不刷新 librarian 蓝图（`/mol:map`），蓝图刷新延后处理。

## Files to create or modify

- `/Users/roykid/work/molcrafts/molrs/molrs/src/compute/traits.rs` — (modify) 新增 `Fit` trait（与 `Compute` 并列）
- `/Users/roykid/work/molcrafts/molrs/molrs/src/compute/fit/mod.rs` (new) — `compute::fit` 子模块定义、共享 window+FFT/OLS/梯形 私有 helper、re-exports
- `/Users/roykid/work/molcrafts/molrs/molrs/src/compute/fit/linear_fit.rs` (new) — `LinearFit` + `LinearFitResult`（OLS 抽取自 dielectric）+ 单元测试
- `/Users/roykid/work/molcrafts/molrs/molrs/src/compute/fit/running_integral.rs` (new) — `RunningIntegral`（梯形抽取自 jacf）+ 单元测试
- `/Users/roykid/work/molcrafts/molrs/molrs/src/compute/fit/plateau.rs` (new) — `Plateau` + 平台均值结果 + 单元测试
- `/Users/roykid/work/molcrafts/molrs/molrs/src/compute/fit/debye_fit.rs` (new) — `DebyeFit`（收口 molpy ad-hoc DebyeFit）+ 单元测试
- `/Users/roykid/work/molcrafts/molrs/molrs/src/compute/fit/spectral.rs` (new) — `PowerSpectrum`/`IRSpectrum`/`RamanSpectrum` 谱变换 `Fit`（组合 signal 原语，复现旧输出）+ 单元测试
- `/Users/roykid/work/molcrafts/molrs/molrs/src/compute/fit/raw_computes.rs` (new) — `VACF`/`EinsteinDiffusion`/`GreenKuboDiffusion`/`EinsteinConductivity`/`GreenKuboConductivity`/`DebyeRelaxation` 原始 `Compute` + 原始结果类型 + 单元测试
- `/Users/roykid/work/molcrafts/molrs/molrs/src/compute/spectra/mod.rs` — (modify) `window_and_fft`/`acf_to_spectrum` 系列改调 `compute::fit` 共享 helper（数值不变；旧函数保留）
- `/Users/roykid/work/molcrafts/molrs/molrs/src/compute/dielectric.rs` — (modify) `windowed_acf_spectrum`/`acf_to_spectrum`/OLS 改调 `compute::fit` 共享 helper（数值不变；旧函数保留）
- `/Users/roykid/work/molcrafts/molrs/molrs/src/compute/mod.rs` — (modify) `pub mod fit;` + re-export `Fit`、各 `Fit`/原始 `Compute` 类型

## Tasks

- [ ] Write failing tests for `Fit` trait + a trivial in-test impl (`/Users/roykid/work/molcrafts/molrs/molrs/src/compute/traits.rs`)
- [ ] Implement `Fit` trait alongside `Compute` (`/Users/roykid/work/molcrafts/molrs/molrs/src/compute/traits.rs`)
- [ ] Write failing tests for numeric fits `LinearFit`/`RunningIntegral`/`Plateau`/`DebyeFit` against known synthetic curves (`/Users/roykid/work/molcrafts/molrs/molrs/src/compute/fit/{linear_fit,running_integral,plateau,debye_fit}.rs`)
- [ ] Implement `LinearFit`/`RunningIntegral`/`Plateau`/`DebyeFit` reusing OLS from dielectric.rs and trapezoid from jacf.rs (`/Users/roykid/work/molcrafts/molrs/molrs/src/compute/fit/{linear_fit,running_integral,plateau,debye_fit}.rs`)
- [ ] Write failing regression tests for spectral `Fit` impls reproducing old `power_spectrum`/`ir_spectrum`/`raman_spectrum` + `dielectric` spectrum output bit-for-bit on a shared ACF (`/Users/roykid/work/molcrafts/molrs/molrs/src/compute/fit/spectral.rs`)
- [ ] Implement `PowerSpectrum`/`IRSpectrum`/`RamanSpectrum` `Fit` composing molrs::signal window/acf/grid via shared helper in `/Users/roykid/work/molcrafts/molrs/molrs/src/compute/fit/spectral.rs` and refactor `spectra/mod.rs` + `dielectric.rs` to call it
- [ ] Write failing tests asserting raw computes equal the raw portion of today's bundled results (EinsteinConductivity.msd == ConductivityResult.msd, GreenKuboConductivity.jacf == JacfResult.jacf, VACF == power_spectrum acf_sum) and invariants (a)/(b)/(c) (`/Users/roykid/work/molcrafts/molrs/molrs/src/compute/fit/raw_computes.rs`)
- [ ] Implement `VACF`/`EinsteinDiffusion`/`GreenKuboDiffusion`/`EinsteinConductivity`/`GreenKuboConductivity`/`DebyeRelaxation` returning raw-only curves + metadata (`/Users/roykid/work/molcrafts/molrs/molrs/src/compute/fit/raw_computes.rs`)
- [ ] Wire `pub mod fit;` + re-exports in `/Users/roykid/work/molcrafts/molrs/molrs/src/compute/mod.rs` and `compute/fit/mod.rs`
- [ ] Add rustdoc per doc.style with units on every new `Fit`/`Compute`/result type (`/Users/roykid/work/molcrafts/molrs/molrs/src/compute/fit/*.rs`, `traits.rs`)
- [ ] Run full check + test suite (`cargo fmt --all --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-features`)

## Testing strategy

### 快乐路径

- `LinearFit`：对合成线性数据 `y = a·x + b + 0` 恢复已知 `slope=a`、`intercept=b`、`r2=1.0`，容差 < 1e-12；窗 `(0.0, 1.0)` 用全程，`fit_start/fit_end` 索引正确。
- `RunningIntegral`：对常数曲线 `c` 在步长 `dt` 上，累积积分在第 k 点等于 `c·k·dt`（容差 < 1e-12）；对已知三角形信号恢复解析积分。
- `Plateau`：对带平台的曲线，窗内均值等于平台值；样本数与窗边界一致。
- 谱 `Fit`：把谱 `Fit` 施加到旧函数内部用的同一条原始 ACF，输出 `frequencies_cm1`/`intensities`（及 Raman 的 iso/aniso/parallel/perp）与旧 `power_spectrum`/`ir_spectrum`/`raman_spectrum` 逐位相等（回归锁）。
- 原始 `Compute`：`EinsteinDiffusion` 返回的 MSD 与 `MSD::windowed()` 一致；`EinsteinConductivity.msd == ConductivityResult.msd`、`GreenKuboConductivity.jacf == JacfResult.jacf`、`VACF` 曲线 == `power_spectrum` 的 `acf_sum`（容差 < 1e-12）。

### 边界情况

- 退化拟合窗（窗内所有 x 相等 / `start >= end`）→ `ComputeError::OutOfRange`。
- 维度不符（谱 `Fit` 收到非 1D ACF、Raman 非 6 分量）→ `ComputeError::DimensionMismatch`。
- 空 / 过短输入（< 2 帧、`max_lag` 小于下游所需）→ `ComputeError::EmptyInput` / `OutOfRange`，落实不变量 (a)。
- `DebyeRelaxation`：缺 V/T/Ewald-BC 元数据时——结果结构强制携带这些字段（编译期保证），ACF 必须非归一化（含非零零延迟方差 ⟨M²⟩）落实不变量 (b)/(c)。

### 科学验证

- Einstein D：`LinearFit` 斜率经 `slope/6` 与已知扩散系数相符。
- GK D：`RunningIntegral` ∫VACF 经 `1/d` 与已知值相符。
- Einstein–Helfand σ：`slope/(6·V·k_B·T)` 复现 `einstein_helfand_conductivity` 旧 σ（同输入）。
- GK σ：`(1/(3·V·k_B·T))∫⟨JJ⟩` 复现 `green_kubo_conductivity` 旧 σ（同输入）。
- 谱回归：谱 `Fit` 对正弦速度/偶极信号的峰值位置与 `vibrational-spectra` 既有科学验收一致（峰位容差 < 2 cm⁻¹），且与旧函数逐位相等。
- 不变量 (a)/(b)/(c) 各自有独立断言用例。

## Out of scope

- 删除任何现有字段 / 函数（`ConductivityResult.sigma/slope/fit_*`、`JacfResult.sigma_running/sigma`、`spectra::*`/`dielectric::*` 自由函数）——属 Phase 03。
- Python 绑定与 bench 的重新指向 / 新增——属 Phase 02。
- `NernstEinsteinConductivity` 具体 `Fit` 的实现（本阶段仅要求 `Fit` 签名能容纳「消费 Dᵢ 标量」形态）；其 OOP 命名收口随后续阶段。
- 已退役的命名替换 / 旧自由函数的废弃标注（`#[deprecated]`）——留待 Phase 03 统一处理。
- librarian 蓝图刷新（`/mol:map`）——延后。
