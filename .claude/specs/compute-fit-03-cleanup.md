---
title: compute-fit-03-cleanup — remove legacy bundled-derived results and free-function compute paths (BREAKING)
status: approved
created: 2026-06-18
---

# compute-fit-03-cleanup — remove legacy bundled-derived results and free-function compute paths (BREAKING)

## Summary

本阶段是「RAW compute 与 FITTING 分离」三段链的收尾，承接 compute-fit-01-module（新增 `compute::fit` 模块与 method-first 的纯 raw OOP compute）与 compute-fit-02-repoint（已把 Python 绑定、molpy 包装、freud-parity benches 重指到 raw-compute + fit 组合并标注旧路径为 deprecated）。在 01/02 的新路径全部就位后，本阶段执行 BREAKING 删除：剥掉 raw 结果体里的 derived 字段（`ConductivityResult` 的 `sigma/slope/fit_start/fit_end`、`JacfResult` 的 `sigma/sigma_running`），把残留的 free function 全部转成实现 `Compute` 的 OOP 结构体或删除（改由 raw-compute + `compute::fit` 组合替代），删掉随之死亡的内联 window+FFT/OLS/梯形积分辅助代码，移除已废弃的 PyO3 绑定与 molpy 过渡 shim，并记录 `molcrafts-molrs` 的 breaking SemVer 跳版要求。完成后：每个公开 compute 入口都是 `impl Compute` 的 OOP 结构体；所有 fit/smoothing/transform 都在 `compute::fit`；spectra computes 返回未加窗的 raw ACF；signal window 复用而非重复实现；全仓（molrs / molrs-python / molpy / benches）对被删字段与函数零引用。

## Domain basis

物理量与公式不变，本阶段只搬运计算职责（raw 观测量 vs. 拟合后导出量），不引入新物理。涉及的既有关系与单位（LAMMPS *real* 单位）原样保留：

- 静态离子电导率（Einstein–Helfand）：`σ = lim_{t→∞} (1/(6·V·k_B·T)) · d/dt ⟨|M_J(t) − M_J(0)|²⟩`，3-D Einstein 因子 `1/(2d)`，`d = 3`。参考：Frenkel & Smit, *Understanding Molecular Simulation*, 2nd ed. (2002) §4.4.2；理想极限退化为 Nernst–Einstein `σ = n·q²·D/(k_B·T)`。
- Green–Kubo 离子电导率：`σ = 1/(3·V·k_B·T) · ∫₀^∞ ⟨J(0)·J(t)⟩ dt`。参考：Hansen & McDonald, *Theory of Simple Liquids*, Eq. 7.7.20。
- 介电谱 EH / GK 路径：Caillol, Levesque & Weis, *J. Chem. Phys.* **85**, 6645 (1986)；静态极限 Neumann, *Mol. Phys.* **50**, 841 (1983)。
- Onsager 系数：`L_ij(τ) = ⟨ΔP_i(τ)·ΔP_j(τ)⟩_t`（raw 互相关曲线，长时线性拟合属拟合层）。
- 振动谱（VDOS/IR/Raman）：raw 量是速度/偶极通量/极化率导数的 ACF；加窗 + FFT + Bose/cross-section 校正属 transform 层。参考：Dickey & Paskin, *Phys. Rev.* **188**, 1407 (1969)。

单位换算常数（`e²`, `Å→m`, `ps→s`, `k_B`）来自 `molrs::units::constants`，删除后保持同一来源，物理输出逐值不变（由 scientific regression criteria 守住）。

## Design

「分离」的终态约束 raw-compute 只产出可观测原始曲线（ACF / MSD / 互相关 / 谱前 raw ACF），所有拟合（OLS 斜率、梯形/running 积分、plateau、Debye 拟合）与变换（加窗、one-sided FT、Bose 因子）都归 `compute::fit`（01 已建）。本阶段删除违反该约束的存量代码：

实体改动：

- `compute/dielectric.rs`
  - 删 `ConductivityResult` 的 derived 字段 `sigma/slope/fit_start/fit_end`，仅保留 raw `{lag_times, msd}`；若 01 已提供 `EinsteinConductivity`（raw `{lag_times, msd}`）且无其它引用，直接删 `ConductivityResult` 与 free function `einstein_helfand_conductivity`，由 `EinsteinConductivity` + `compute::fit::LinearFit` 组合替代。
  - 把谱 free function `einstein_helfand_spectrum` / `green_kubo_spectrum` 转成 raw-compute OOP 结构体（产出 raw ACF / 导数 ACF）或删除，改由 raw ACF compute + `compute::fit` transform 组合；删内联 transform 辅助 `windowed_acf_spectrum`、`acf_to_spectrum`、`windowed_acf_derivative_spectrum`（其职责已在 `compute::fit`）。
  - 保留纯 raw 的 `compute_dipole_moment` / `compute_current_density` / `decompose_current` / `static_dielectric_constant` / `static_dielectric_constant_components`（fluctuation 公式属定义量，非拟合）。
- `compute/jacf.rs`
  - 删 `JacfResult` 的 derived 字段 `sigma/sigma_running`，仅保留 raw `{lag_times, jacf}`；若 01 已提供 `GreenKuboConductivity`（raw `{lag_times, jacf}`）且无其它引用，删 `JacfResult` 与 free function `green_kubo_conductivity`，由 `GreenKuboConductivity` + `compute::fit::RunningIntegral` 组合替代；删内联梯形积分代码。
- `compute/onsager.rs`
  - 把 free function `onsager_correlation` 转成 `OnsagerCorrelation` 结构体 `impl Compute`（raw collective 互相关曲线，`OnsagerResult` 保持 raw `{lag_times, correlation}` 不变）。
- `compute/spectra/`
  - 把 free function `power_spectrum` / `ir_spectrum` / `raman_spectrum` 转成 raw-compute 结构体 `impl Compute`，返回 raw（未加窗、未变换）ACF；删除其 baked-in 加窗输出。删 `compute/spectra/mod.rs` 的内联 `window_and_fft`、`acf_to_spectrum`、`acf_to_intensities`（迁移/复用 `compute::fit` 与 `compute::signal` 后死亡的部分）；`cosine_sq_window`/`bose_factor` 若已迁入 `compute::fit` 则一并删除其重复定义，window 一律复用 `molrs::signal`。
- `compute/mod.rs`：更新 re-export，去掉被删符号，加入新 OOP 结构体的 re-export，保持 raw compute + `compute::fit` 的公开面一致。

生命周期 / 归属：被删 derived 字段的拥有者从 raw 结构体转移到调用方组合（raw compute → `compute::fit`），与 01/02 已建立的组合约定一致。`ComputeResult::finalize` 的归一化（RDF g(r)、density 体积归一、COM 质量加权）不是拟合，留在 raw computes；`gaussian_density` / diffraction Gaussian smear 是定义量、`local_density` diameter taper 是估计器参数——全部不动。

绑定 / 下游：

- `molrs-python/src/dielectric.rs`、`molrs-python/src/transport.rs` 删掉调用已删 free function / 已删 derived 字段的 deprecated PyO3 函数；`lib.rs` 的 `register_dielectric` / `register_transport` 注册随之收敛。`molrs-python/python/molrs/dielectric.py`、`molrs-python/python/molrs/transport.py` 删掉对应过渡 shim（02 引入的 raw + fit 入口保留）。
- molpy 侧（独立仓 / 已装 wheel）删掉 02 保留的过渡 shim wrappers（transport / dielectric 包装中指向旧绑定的 fallback 路径）。
- molcrafts-molrs 走 breaking SemVer 跳版；下游 pin（molpack、molpy exact-pin）须更新——本阶段只记录要求，不实际发布。

## Files to create or modify

- `molrs/src/compute/dielectric.rs` — 删 `ConductivityResult` derived 字段（或删整个 bundled 类型 + `einstein_helfand_conductivity`）；删/转 `einstein_helfand_spectrum`、`green_kubo_spectrum`；删 `windowed_acf_spectrum`、`acf_to_spectrum`、`windowed_acf_derivative_spectrum` 内联辅助及内联 OLS 代码。
- `molrs/src/compute/jacf.rs` — 删 `JacfResult` 的 `sigma/sigma_running`（或删 `JacfResult` + `green_kubo_conductivity`）；删内联梯形积分代码。
- `molrs/src/compute/onsager.rs` — `onsager_correlation` free fn → `OnsagerCorrelation` 结构体 `impl Compute`。
- `molrs/src/compute/spectra/mod.rs` — 删 `window_and_fft`、`acf_to_spectrum`、`acf_to_intensities`（及迁移后重复的 `cosine_sq_window`/`bose_factor`）；spectra 结果体改为 raw ACF。
- `molrs/src/compute/spectra/power_spectrum.rs` — `power_spectrum` free fn → raw-compute 结构体，返回 raw ACF。
- `molrs/src/compute/spectra/ir_spectrum.rs` — `ir_spectrum` free fn → raw-compute 结构体，返回 raw ACF。
- `molrs/src/compute/spectra/raman_spectrum.rs` — `raman_spectrum` free fn → raw-compute 结构体，返回 raw ACF（iso/aniso）。
- `molrs/src/compute/mod.rs` — 更新 re-export：删被删符号，加新 OOP 结构体。
- `molrs-python/src/dielectric.rs` — 删 deprecated PyO3 绑定。
- `molrs-python/src/transport.rs` — 删 deprecated PyO3 绑定（`transport_green_kubo_conductivity` 的 `sigma/sigma_running` set_item 等）。
- `molrs-python/src/lib.rs` — 收敛 `register_dielectric` / `register_transport` 注册到剩余函数。
- `molrs-python/python/molrs/dielectric.py` — 删过渡 shim（保留 02 raw + fit 入口）。
- `molrs-python/python/molrs/transport.py` — 删过渡 shim（保留 02 raw + fit 入口）。
- `molrs/benches/compute/dielectric.rs` — 重指到 raw-compute + fit（去掉对已删 free fn 的 import）。
- `molrs/benches/compute/spectra.rs` — 重指到 raw-compute 结构体调用。
- `Cargo.toml`（`molrs/Cargo.toml` 或 workspace 根，含 `molcrafts-molrs` version）— 记录 breaking SemVer 跳版（不发布）。

## Tasks

- [ ] Write failing tests for raw `ConductivityResult`/`JacfResult` field removal + `EinsteinConductivity`/`GreenKuboConductivity` + `LinearFit`/`RunningIntegral` composition (molrs/src/compute/dielectric.rs, molrs/src/compute/jacf.rs)
- [ ] Remove derived fields / bundled types + inline OLS, trapezoidal, and transform helpers from dielectric.rs and jacf.rs (molrs/src/compute/dielectric.rs, molrs/src/compute/jacf.rs)
- [ ] Write failing tests for OnsagerCorrelation and spectra raw-compute structs returning unwindowed ACFs (molrs/src/compute/onsager.rs, molrs/src/compute/spectra/power_spectrum.rs, molrs/src/compute/spectra/ir_spectrum.rs, molrs/src/compute/spectra/raman_spectrum.rs)
- [ ] Convert onsager_correlation, power_spectrum, ir_spectrum, raman_spectrum free functions to Compute structs and delete window_and_fft/acf_to_spectrum/acf_to_intensities from spectra/mod.rs (molrs/src/compute/onsager.rs, molrs/src/compute/spectra/mod.rs, molrs/src/compute/spectra/power_spectrum.rs, molrs/src/compute/spectra/ir_spectrum.rs, molrs/src/compute/spectra/raman_spectrum.rs)
- [ ] Update compute re-exports for removed symbols and new structs (molrs/src/compute/mod.rs)
- [ ] Remove deprecated PyO3 bindings and converge registration (molrs-python/src/dielectric.rs, molrs-python/src/transport.rs, molrs-python/src/lib.rs)
- [ ] Remove transition shims from molpy and molrs Python wrappers (molrs-python/python/molrs/dielectric.py, molrs-python/python/molrs/transport.py)
- [ ] Repoint dielectric and spectra benches to raw-compute + fit composition (molrs/benches/compute/dielectric.rs, molrs/benches/compute/spectra.rs)
- [ ] Record breaking SemVer bump and downstream pin-update requirement (Cargo.toml; do not publish)
- [ ] Rebuild maturin wheel and run full check + test suite (cargo test --all-features, cargo clippy --all-targets --all-features -D warnings, cargo fmt, molpy + bench suites)

## Testing strategy

- Happy path — raw computes return only raw observables: `ConductivityResult`/`EinsteinConductivity` → `{lag_times, msd}`; `JacfResult`/`GreenKuboConductivity` → `{lag_times, jacf}`; `OnsagerCorrelation` → `{lag_times, correlation}`; spectra computes → raw unwindowed ACF. Existing hand-checkable unit tests (3-frame x-ramp MSD = {0, 2.5, 9}; constant-current JACF C(τ)=1; collective-MSD self-correlation L(τ)=τ²) are preserved against the raw structs.
- Edge cases — invalid input contracts (non-`(_,3)` shape, < 2 frames, non-positive dt/volume/temperature, NaN/inf) still error identically through the OOP `Compute` entry points; resolution/max_lag clamping unchanged.
- Domain validation (scientific) — composing `EinsteinConductivity` + `compute::fit::LinearFit` reproduces the Nernst–Einstein conductivity within the existing ≤0.13 ensemble tolerance; composing `GreenKuboConductivity` + `compute::fit::RunningIntegral` reproduces the previous `sigma`/`sigma_running` values; EH/GK dielectric spectra recover the Neumann static limit and the Debye-equivalence; power-spectrum sine peak lands at ~333.56 cm⁻¹. All physics outputs are bit-for-bit/within-prior-tolerance unchanged vs. pre-cleanup (no new physics).
- Grep-clean — no remaining references anywhere in molrs / molrs-python / molpy / benches to `sigma`/`slope`/`fit_start`/`fit_end` on the old result types, to `einstein_helfand_conductivity`/`green_kubo_conductivity`/`einstein_helfand_spectrum`/`green_kubo_spectrum`/`onsager_correlation`/`power_spectrum`/`ir_spectrum`/`raman_spectrum` as free functions, or to `window_and_fft`/`windowed_acf_spectrum`/`acf_to_spectrum`/`sigma_running`.
- Cross-binding — `cargo test --all-features` green; `cargo clippy --all-targets --all-features -- -D warnings` clean; `cargo fmt --all --check` clean; rebuilt maturin wheel installed; molpy + freud-parity bench suites green with parity floors intact.

## Out of scope

- Adding new physics, new fit types, or new spectral transforms (those landed in compute-fit-01-module).
- Touching `gaussian_density` / diffraction Gaussian smear (defined quantities) or `local_density` diameter taper (estimator parameter).
- Altering `ComputeResult::finalize` normalization (RDF g(r), density volume norm, COM mass-weighting) — that is normalization, not fitting.
- Actually publishing the breaking release or bumping downstream pins in molpack/molpy — only the bump requirement is recorded here.
