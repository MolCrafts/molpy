---
title: molpy OPLS typifier + 估计器退成 molrs 薄壳（消费侧 rewire）
status: approved
created: 2026-06-18
---

# molpy OPLS typifier + 估计器退成 molrs 薄壳

> molpy 消费侧 spec，对应 molrs 实现链 `opls-typifier-01-typing` / `-02-assign` / `-03-parity` + `ff-parameter-estimator`（均在 `/Users/roykid/work/molcrafts/molrs/.claude/specs/`）。**前置硬门**：molrs `opls-typifier-03-parity` 绿（逐原子 type 100% 一致、成键参数容差内）后才删 molpy Python 分型代码。

## Summary

OPLS-AA 原子分型 + 成键参数赋值 + 缺参估计已整体下沉 molrs（Rust，复用 MMFF/GAFF typifier 范式）。本 spec 把 molpy `typifier/` 退成薄壳：`OplsTypifier` 委托/再导出 `molrs.OplsTypifier`（镜像现有 `MMFFTypifier` 的消费方式），删除 molpy 侧 Python 分型实现（`_OplsAtomTypifier`、`ForceField{Bond,Angle,Dihedral}Typifier`、`graph.SMARTSGraph` 引擎等），并把 molrs 依赖 pin 到含 OPLS typifier 的新版本。这反转了 molrs `opls-ef-01-kernels-seam` 的"typifier 不下沉（B线）"——分型不再在 molpy。

## Domain basis

正确性 = 行为等价。删除 molpy Python 分型的前提是 molrs `opls-typifier-03-parity` 已证明 molrs `OplsTypifier` 与 molpy 旧实现逐原子一致、成键参数容差内（bond r₀ 0.02 Å、angle θ₀ 3°、力常数 rtol 0.10）。释放链路（见 molpy 记忆 `feedback_release_order_and_pinning`）：molrs 先发布含 OPLS typifier 的版本 → molpy `pyproject.toml` pin 该精确版本 → molpy 删 Python 分型 + 改委托。

## Design

**1. 委托而非重写**

molpy `typifier.OplsTypifier` 改为薄包装 `molrs.OplsTypifier`（PyO3 暴露的 `PyOplsTypifier`，由 molrs 链随 parity 通过后绑定）。公开 API/返回类型对 molpy 下游保持稳定（返回 typed `Frame` 或 labeled `Atomistic`，与现状一致）。估计器：molpy 不再有 `ParameterEstimator` Python 实现（那份 spec 已 superseded）；开启估计走 molrs `OplsTypifier` 的 estimator 选项。

**2. 删除清单（parity 绿后）**

- `typifier/atomistic.py`：`_OplsAtomTypifier`、`ForceFieldBondTypifier`、`ForceFieldAngleTypifier`、`ForceFieldDihedralTypifier`、`ForceFieldAtomTypifier`、`_sequence_score`/`_end_score`/`_build_type_class_layer` 等仅服务 OPLS 分型的逻辑。
- `typifier/graph.py`（Python SMARTS 引擎 SMARTSGraph）—— 若仅被 OPLS/GAFF 分型使用则删；若 parser 层有其它消费方则保留 parser、删分型用法。
- `typifier/matcher.py`、`layered_engine.py`、`dependency_analyzer.py`、`pair.py`、`bond.py`/`angle.py`/`dihedral.py`（未导出旧件）—— 按实际依赖逐一评估删/留。
- CL&P：`ClpTypifier(OplsTypifier)` 继承链相应改为基于 molrs（CL&P 专属表 molrs v2；本 spec 仅保证 OPLS 路径，CL&P 委托随后）。

**3. 依赖 pin**

`pyproject.toml` 的 `molcrafts-molrs` pin 到含 OPLS typifier 的发布版本（开发期 editable，发布期精确 pin，对齐 `feedback_release_order_and_pinning` / `feedback_version_alignment`）。

**4. 测试迁移**

molpy `tests/test_typifier/` 中验证 OPLS 分型数值的测试改为：要么调薄壳验证委托正确，要么标记为 molrs 侧 parity 覆盖后移除；保留 molpy 公开 API 契约测试。

## Files to create or modify

- `src/molpy/typifier/atomistic.py` — 删 OPLS 分型实现，`OplsTypifier` 改委托 `molrs.OplsTypifier`。
- `src/molpy/typifier/__init__.py` — 导出对齐（`OplsTypifier` 仍可导入；移除已删符号）。
- `src/molpy/typifier/{graph,matcher,layered_engine,dependency_analyzer,pair,bond,angle,dihedral}.py` — 按依赖评估删/留。
- `src/molpy/typifier/clp.py` — CL&P 继承链改基于 molrs（或留待 CL&P 委托 spec）。
- `pyproject.toml` — pin `molcrafts-molrs` 到含 OPLS typifier 的版本。
- `tests/test_typifier/` — 迁移/精简 OPLS 分型测试为委托契约测试。

## Tasks

- [ ] Gate: confirm molrs opls-typifier-03-parity is green (per-atom 100% + params within tol) before any deletion
- [ ] Pin molcrafts-molrs to the release exposing OplsTypifier in pyproject.toml; rebuild editable
- [ ] Rewrite molpy OplsTypifier as a thin delegate over molrs.OplsTypifier (stable public API/return types)
- [ ] Remove molpy Python OPLS typification (_OplsAtomTypifier, ForceField*Typifier, _sequence_score/_end_score/_build_type_class_layer) once delegate verified
- [ ] Evaluate and remove now-dead typifier modules (graph/matcher/layered_engine/dependency_analyzer/pair/bond/angle/dihedral) per actual dependency
- [ ] Migrate tests/test_typifier OPLS tests to delegate-contract tests; keep public API contract tests
- [ ] Run ruff format --check, ruff check, ty check, pytest -m "not external"

## Testing strategy

- **Contract** — `from molpy.typifier import OplsTypifier` still works; typifying a known molecule returns the same public shape as before; delegates to molrs.
- **Regression** — molpy public API consumers (builder/io paths using OplsTypifier) unaffected.
- **Cleanup** — grep confirms removed symbols have zero remaining references in molpy.
- **Build smoke** — ruff/ty/pytest (not external) green.

## Out of scope

- **molrs 侧实现** → molrs 链 `opls-typifier-01/02/03` + `ff-parameter-estimator`。
- **GAFF 分型 molpy rewire** — GAFF 走 molrs `gaff-typifier-*` 链；molpy GAFF 委托另立。
- **CL&P/CL&Pol 完整委托** — CL&P 专属表 molrs v2；本 spec 仅保证 OPLS 路径。
- **删除前的删除** — 任何删除都以 molrs parity 门绿为前提。
