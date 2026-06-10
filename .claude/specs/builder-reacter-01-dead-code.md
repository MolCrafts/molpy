---
title: Builder/Reacter Dead-Code Consolidation (Phase 01)
status: done
created: 2026-06-10
---

# Builder/Reacter Dead-Code Consolidation (Phase 01)

## Summary
本规格是 builder/reacter 五段清理链的第 01 段：纯删除与去重，零行为变更。`builder/polymer` 下存在多组重复或死代码——重复的 `PolymerBuilder` 分叉、两套发散的异常体系、约 1000 行无人引用的随机生长子系统、一个死文件 `selectors.py`、以及一个仅被单测拖活的 `AmberPolymerBuilder` 分叉。本规格逐项删除并收敛到唯一实现，使 `from molpy.builder.polymer import PolymerBuilder` 导出的就是 `core.PolymerBuilder`，连接器抛出的异常可被 `except AssemblyError` 统一捕获。删除完成后净减约 1100+ 行，全测试套件（`-m "not external"`）保持绿色，`ruff`/`ty` 通过，公共 `__init__` 导出表无断裂导入。本段不改任何 API 形态、不动 `core/`，为后续 02–05 段提供干净基线。

## Domain basis
不适用。本规格为纯删除/去重的代码维护工作，不引入或修改任何物理模型、方程或单位；G-BigSMILES 的生产路径（`compiler.py` → `system.py` + `distributions.py` + `sequences.py`）保持不变，被删除的是从未接入该路径的孤立随机生长副本。

## Design
收敛原则——每个重复族保留"被生产路径引用的那一份"，删除分叉：

- **PolymerBuilder 唯一化**：`core.py:120` 的 `PolymerBuilder` 是活实现（被 `dsl.py:137`、`compiler.py:309` 使用，含端口生命周期方法）。`polymer_builder.py:39` 是仅被 `test_polymer_builder.py` 拖活的陈旧分叉。将 `PolymerBuildResult` 的定义从 `polymer_builder.py` 迁入 `core.py`，删除 `core.py:112` 对 `polymer_builder` 的回引导入，删除 `polymer_builder.py`，并令 `polymer/__init__.py` 改从 `core` 重导出 `PolymerBuilder` 与 `PolymerBuildResult`。
- **异常单源**：`errors.py` 为唯一权威来源。修正其分叉 bug——`errors.py:50` `GeometryError(Exception)` 改为 `GeometryError(AssemblyError)`，与 `core.py:93` 的基类对齐，消除"连接器抛 `errors.GeometryError` 却逃过 `except AssemblyError`"的静默漏捕。删除 `core.py:51-108` 的全部重复异常类，改为 `from .errors import (...)`。`connectors.py` 已从 `.errors` 导入，无需改动其导入源。
- **死随机子系统清除**：`growth_kernel.py`、`stochastic_generator.py` 是从 `stochastic.py` 抽出的副本，三者互相引用但无任何外部导入者。整体删除两文件及 `stochastic.py`（其数据类型经核实仅被 `__init__` 与这两个死兄弟引用，属孤立），同步清理 `polymer/__init__.py` 的相关导出。
- **死文件清除**：`selectors.py` 的 `process_port_markers` 无任何引用，`LeavingGroupSelector` 别名与 `reacter.selectors` 重复；`find_neighbors` 的活路径已是 `reacter.selectors`。整文件删除。
- **Amber 分叉清除**：保留已导出的 `ambertools/amber_builder.py:AmberPolymerBuilder`，删除未导出、仅被 `test_polymer_amber_sequence.py` 拖活的 `ambertools/polymer_amber.py`；其唯一消费者 `residue_manager.py` 随之孤立，一并删除（`amber_builder.py` 不使用它）。
- **虚假文档串清理**：`builder/__init__.py:3-5` 宣称 `PolymerBuilder` 已移除并指向不存在的 notebook，两项均为假。改写为指向 `polymer()` / `polymer_system()` 入口（完整 API 重组属第 04 段，此处仅删假述）。

## Files to create or modify
- src/molpy/builder/polymer/errors.py — 修正 `GeometryError` 基类为 `AssemblyError`，确立为异常唯一来源
- src/molpy/builder/polymer/core.py — 删除重复异常类（51-108 行），改从 `.errors` 导入；内联 `PolymerBuildResult` 定义，移除对 `.polymer_builder` 的回引导入（112 行）
- src/molpy/builder/polymer/polymer_builder.py — 删除（陈旧 PolymerBuilder 分叉）
- src/molpy/builder/polymer/growth_kernel.py — 删除（死副本）
- src/molpy/builder/polymer/stochastic_generator.py — 删除（死副本）
- src/molpy/builder/polymer/stochastic.py — 删除（孤立随机生长类型与生成器）
- src/molpy/builder/polymer/selectors.py — 删除（死文件，重复别名）
- src/molpy/builder/polymer/ambertools/polymer_amber.py — 删除（未导出的 Amber 分叉）
- src/molpy/builder/polymer/residue_manager.py — 删除（删 polymer_amber 后孤立）
- src/molpy/builder/polymer/__init__.py — 改从 `core` 重导出 `PolymerBuilder`/`PolymerBuildResult`；移除 growth_kernel、stochastic 及其数据类型的导出
- src/molpy/builder/__init__.py — 改写顶部 docstring，删除虚假声明，命名 `polymer()`/`polymer_system()` 入口
- tests/test_builder/test_polymer_core.py — 新增导出一致性与异常单源回归测试；吸收 `test_polymer_builder.py` 中仍适用于活类的行为测试
- tests/test_builder/test_polymer_builder.py — 删除/合并（合并后移除该文件）
- tests/test_builder/test_polymer_amber_sequence.py — 重指向 `ambertools.amber_builder` 或并入 `test_amber_polymer_builder.py`
- tests/conftest.py — 更新对 `test_polymer_builder.py` 的陈旧 external 标记特例（合并后移除该分支）

## Tasks
- [x] Write failing regression tests for export identity and exception single-source (tests/test_builder/test_polymer_core.py)
- [x] Consolidate exceptions: fix GeometryError base to AssemblyError in errors.py and repoint core.py to import from errors, deleting duplicate classes
- [x] Inline PolymerBuildResult into core.py, remove the polymer_builder back-import, delete polymer_builder.py, and re-export PolymerBuilder/PolymerBuildResult from polymer/__init__.py
- [x] Delete dead stochastic subsystem (growth_kernel.py, stochastic_generator.py, stochastic.py) and clean their exports from polymer/__init__.py
- [x] Delete selectors.py and confirm find_neighbors resolves only via reacter.selectors
- [x] Delete ambertools/polymer_amber.py and residue_manager.py; the amber-sequence test exercised only the deleted fork's private helper (absent from amber_builder), so it was removed rather than repointed
- [x] Rewrite the false docstring in builder/__init__.py to name polymer()/polymer_system() entry points
- [x] Merge test_polymer_builder.py behavior tests into test_polymer_core.py and update the stale conftest.py external-marker special-case
- [x] Run full check + test suite
- [x] Hygiene cleanup (/mol:simplify: 3 fixes — dead `Any` import, stale comment, stale bytecode; suite green)

## Testing strategy
- Happy path: `python -c "import molpy.builder; import molpy.builder.polymer; import molpy.reacter"` 全部成功，证明导出表无断裂导入。
- 导出一致性（回归，RED-first）：`from molpy.builder.polymer import PolymerBuilder` 与 `from molpy.builder.polymer import core` 满足 `PolymerBuilder is core.PolymerBuilder`；`PolymerBuilder(library=..., reacter=...)` 可构造；类 docstring 中记录的活签名示例可成功构造。
- 异常单源（回归，RED-first）：`issubclass(GeometryError, AssemblyError)` 为真；由 `connectors.py` 抛出的 `AmbiguousPortsError` 可被 `except AssemblyError` 捕获。
- 边界/删除验证：在 `src/` 全树 grep `growth_kernel`、`stochastic_generator`、`GrowthKernel`、`StochasticChainGenerator`、`process_port_markers` 均无命中；被删文件在磁盘上确实不存在；`core.py` 中无 `from .polymer_builder` 残留。
- 死代码确认（删除前）：对 `stochastic.py` 数据类型与 `residue_manager.py` 再次 grep，确认除 `__init__` 与待删兄弟外无其它引用，方可删除。
- 域验证：不适用（零行为变更）。
- 回归套件：`pytest tests/ -m "not external" -v` 全绿；`ruff format --check`、`ruff check`、`ty check src/molpy/` 通过；Amber 相关合并测试仍带 `@pytest.mark.external`，本地按需跳过。

## Out of scope
- API 表面重组（`polymer()`/`polymer_system()` 入口体系的完整重构）——属第 04 段。
- 序列化/数据搬移——属第 02 段。
- 任何行为变更、性能优化——分别属第 05 段及后续。
- `core/`（`Atomistic` 等）的任何改动——由草案规格 molgraph-ecs-03-molpy 负责，本段绝不触碰。
- 新增文档页面——第 04 段负责。
