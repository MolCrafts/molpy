---
title: Builder/Reacter API consolidation, terminology unification, and documentation repair
status: done
created: 2026-06-10
---

# Builder/Reacter API 整合、术语统一与文档修复

## Summary

本规格是 builder-reacter 五段链的第 4 段（依赖 01–03 已落地），不改变任何运行时行为，只收敛公共 API 表面、统一术语、修复已验证会崩溃的文档示例，并建立"文档示例永不腐烂"的冒烟测试机制。完成后：`polymer()` / `polymer_system()` 成为唯一被文档化的一站式建链入口，`PolymerBuilder` + `Connector` 作为进阶分步 API 保留导出，编译器/规划器等内部件从 `__all__` 中降级；`ReactionPresets` / `ReactionPresetSpec` 升级为公开扩展点；`mp.builder` / `mp.reacter` / `mp.pack` / `mp.compute` 可以作为 `molpy` 顶层属性访问；`anchor_selector_*` 成为全库统一的锚点选择器术语；`docs/api/builder.md`、`docs/api/reacter.md`、`reacter/base.py` 类文档中所有复制即崩溃的示例被替换为可执行版本，并由新增的文档抽取测试与 doctest 持续守护；每个建链/反应目标各有一个可运行的 `examples/` 脚本。依项目记忆规则，ToolRegistry / Tool 类属于 AI agent 基础设施，迁入 agent 专用模块且绝不出现在任何面向用户的文档、docstring 或导出中。

## Design

**A. 入口收敛（builder 层）**

- `builder/polymer/dsl.py`（738 行）拆分：`PrepareMonomer` / `BuildPolymer` / `PlanSystem` / `BuildSystem` / `BuildPolymerAmber` 五个 Tool 类（dsl.py:35–405）整体迁至新模块 `builder/polymer/tools.py`（agent-only，**不**从 `builder` 或 `builder.polymer` 的 `__init__` 再导出）；`dsl.py` 只保留纯函数 `polymer()`（:406）、`polymer_system()`（:501）、`prepare_monomer()`（:547）、`generate_3d()`（:589）、`_detect_notation()`（:626）及 `_build_*` 私有助手，模块 docstring 删除 ToolRegistry 提及（dsl.py:6）。
- `builder/polymer/__init__.py`（现导出 45 名）重排 `__all__`：入口函数与 `PolymerBuilder` / `PolymerBuildResult` / `Connector` / `ConnectorContext` / Placer 族保留；新增导出 `ReactionPresets` + `ReactionPresetSpec`（presets.py:43，`reaction_preset` 字符串 kwarg 的唯一扩展点，文档化 `register()` 作为自定义化学的扩展入口）；`GBigSmilesCompiler` / `SystemPlanner` / `PolydisperseChainGenerator` 等内部件移出 `__all__`（仍可导入），docstring 标注 internal。`builder/__init__.py` 同步移除五个 Tool 类导出。
- `src/molpy/__init__.py` 通过模块级 `__getattr__` 惰性暴露 `builder` / `reacter` / `pack` / `compute` 子模块（README 教学 `mp.*` 风格；惰性以避免 RDKit 等重依赖拖慢 `import molpy`），并补入 `__all__`。

**B. 术语统一（锚点 = anchor）**

- `ReactionPresetSpec.site_selector_left/right` 改名 `anchor_selector_left/right`（presets.py:27,36,92–93,141,154 的静默重映射随之消失），同步更新 `tests/test_builder/test_polymer_presets.py` 等调用点；实验阶段不留弃用垫片，改名记入新建 `CHANGELOG.md`。
- `reacter/__init__.py:84–85` 的 `find_port = find_port_atom` 别名对只保留 `find_port`：`selectors.py` 中函数本体改名为 `find_port`，删除别名行，`__all__` 移除 `find_port_atom`，更新 `tests/test_reacter/` 各调用点（`find_port_atom_by_node` 不在本次范围）。
- `builder/polymer/core.py:133` Connector 示例 docstring 的 `rules=` 修正为实际 kwarg `port_map=`。
- `bond_react.py` 中 'initiator' 仅在 LAMMPS map 文件段名处保留（LAMMPS 方言），docstring 注明这一点。

**C. 文档表面修复（全部为已验证的崩溃点）**

- `docs/api/builder.md:34–35`：`polymer(..., reacter=rxn)` → 实际签名 `reaction_preset: str`；`result.polymer` → `polymer()` 直接返回 `Atomistic`。
- `docs/api/reacter.md:10,15,45`：`TemplateReacter` → `BondReactReacter`；`select_nothing` → `select_none`；`result.product_info.product` → `result.product`。
- `reacter/base.py:79–96` Reacter 类 docstring 示例：`select_port_atom`（不存在）→ `select_port`；`port_selector_left` → `anchor_selector_left`；`run(structA, structB, port_L="1", ...)` → 实际 `run(left, right, port_atom_L: Entity, port_atom_R: Entity)`；以 `docs/user-guide/02_polymer_stepwise.ipynb` 中可运行示例为蓝本重写。
- `docs/user-guide/04_crosslinking.ipynb` 删除手写 `select_hydroxyl_group`，改用 `molpy.reacter` 的同名导出（reacter/__init__.py:70）。
- 检查 `docs/api/builder.md` 不再渲染 `molpy.builder._tool` / ToolRegistry 任何痕迹（记忆规则）。

**D/E. 防腐机制与示例**

- 新建 `tests/test_docs/test_doc_examples.py`：抽取 `docs/api/builder.md` 与 `docs/api/reacter.md` 的 python 代码块逐块 `exec`（轻量 fixture，重依赖缺失时 skip）；对 `reacter/base.py` 与 `builder/polymer/dsl.py` 跑定向 doctest；调用两个 examples 脚本的 `main()`（RDKit 缺失时 skip）；全部标记非 external。
- 新建 `examples/polymer_build.py`（`polymer()` 线性链 + `polymer_system()` 多分散体系）与 `examples/reacter_bond_react_templates.py`（`BondReactReacter` → `molpy.io.write_lammps_bond_react_system`，镜像 04_crosslinking.ipynb 最小路径）。

**F. Docstring 审计**

存活公共表面（`polymer`、`polymer_system`、`PolymerBuilder`、`Connector`、`ReactionPresets`、`Reacter`、`BondReactReacter`、`BondReactTemplate`、`find_port`、关键 selectors）补齐 Google 风格 docstring（含类型，涉及物理量处标单位，如 `CovalentSeparator.buffer` 的 Å）；`builder/polymer/__init__.py`、`builder/__init__.py`、`reacter/__init__.py` 模块 docstring 重写为"从这里开始"导览（入口函数最先、进阶 API 其次、内部件不列出）。

## Files to create or modify

- src/molpy/builder/polymer/tools.py (new)
- src/molpy/builder/polymer/dsl.py
- src/molpy/builder/polymer/__init__.py
- src/molpy/builder/__init__.py
- src/molpy/builder/polymer/presets.py
- src/molpy/builder/polymer/core.py
- src/molpy/builder/polymer/polymer_builder.py
- src/molpy/builder/polymer/connectors.py
- src/molpy/reacter/__init__.py
- src/molpy/reacter/base.py
- src/molpy/reacter/selectors.py
- src/molpy/reacter/bond_react.py
- src/molpy/__init__.py
- docs/api/builder.md
- docs/api/reacter.md
- docs/user-guide/04_crosslinking.ipynb
- tests/test_docs/test_doc_examples.py (new)
- tests/test_builder/test_polymer_presets.py
- tests/test_reacter/test_selectors.py
- tests/test_reacter/test_base.py
- tests/test_reacter/test_basic.py
- tests/test_reacter/test_bond_react.py
- examples/polymer_build.py (new)
- examples/reacter_bond_react_templates.py (new)
- CHANGELOG.md (new)

## Tasks

- [x] Write failing tests for doc-example smoke + API surface (tests/test_docs/test_doc_examples.py): exec docs/api/builder.md 与 docs/api/reacter.md 的 python 代码块、定向 doctest reacter/base.py 与 builder/polymer/dsl.py、断言 `hasattr(mp, "builder"/"reacter"/"pack"/"compute")`、断言 Tool 类不在 `molpy.builder.__all__` / `molpy.builder.polymer.__all__`、调用两个 examples 脚本（RDKit 缺失 skip）
- [x] Implement agent-only tools module (builder/polymer/tools.py)：迁入五个 Tool 类（自 dsl.py:35–405），dsl.py 仅保留 polymer/polymer_system/prepare_monomer/generate_3d/_detect_notation/_build_* 并清除 ToolRegistry 提及
- [x] Implement export consolidation：builder/polymer/__init__.py 移除 Tool 类与 GBigSmilesCompiler/SystemPlanner/PolydisperseChainGenerator（出 `__all__`、docstring 标 internal），新增导出 ReactionPresets/ReactionPresetSpec；builder/__init__.py 同步移除 Tool 类
- [x] Implement lazy submodule access in src/molpy/__init__.py：模块级 `__getattr__` 暴露 builder/reacter/pack/compute 并补入 `__all__`
- [x] Implement terminology unification：presets.py 字段 site_selector_* → anchor_selector_*（删除 :92–93 静默重映射）；selectors.py find_port_atom 改名 find_port，reacter/__init__.py 删除别名行并更新 `__all__`；core.py:133 docstring `rules=` → `port_map=`；同步更新调用点测试；新建 CHANGELOG.md 记录两处改名
- [x] Implement doc fixes in docs/api/builder.md（:34–35 改 reaction_preset 签名与直接返回 Atomistic、删除 _tool/ToolRegistry 渲染）与 docs/api/reacter.md（:10,15,45 改 BondReactReacter/select_none/result.product）
- [x] Implement docstring/notebook fixes：reacter/base.py:79–96 以 02_polymer_stepwise.ipynb 可运行示例重写；docs/user-guide/04_crosslinking.ipynb 改用 molpy.reacter.select_hydroxyl_group 导出；bond_react.py 注明 'initiator' 仅为 LAMMPS map 段名
- [x] Implement examples scripts (examples/polymer_build.py, examples/reacter_bond_react_templates.py)：各含 `main()`，后者走 BondReactReacter → write_lammps_bond_react_system 最小路径
- [x] Add Google-style docstrings（类型 + 物理量单位）on polymer/polymer_system/PolymerBuilder/Connector/ReactionPresets/Reacter/BondReactReacter/BondReactTemplate/find_port/关键 selectors；重写 builder、builder.polymer、reacter 三个 `__init__.py` 模块 docstring 为"从这里开始"导览
- [x] Run full check + test suite（`ruff format src tests && ruff check src tests && ty check src/molpy/ && pytest tests/ -m "not external" -v`）
- [x] Hygiene cleanup (/mol:simplify: 7 fixes — phantom-class docstring, agent-infra leak in Related: sections, dead pragmas, stale comments; suite green)
- [x] Deviation note: `molpy.builder.polymer` 属性是 polymer() 函数（star-import 覆盖子模块属性），验收脚本经 importlib.import_module 寻址模块；ac-011 的 CHANGELOG.md 建于仓库根（指向 docs/changelog.md）

## Testing strategy

- **Happy path**：`tests/test_docs/test_doc_examples.py` 逐块 exec 两个 API 文档的 python 代码块全部通过；定向 doctest（reacter/base.py、builder/polymer/dsl.py）通过；两个 examples 脚本 `main()` 跑通（本地、非 external）。
- **API 表面**：`import molpy as mp` 后 `mp.builder` / `mp.reacter` / `mp.pack` / `mp.compute` 可用；Tool 类仅可经 `molpy.builder.polymer.tools` 导入且不在任何 `__all__`；`ReactionPresets` / `ReactionPresetSpec` 可从 `molpy.builder.polymer` 导入。
- **改名回归**：现有 tests/test_reacter 与 tests/test_builder/test_polymer_presets.py 在 anchor_selector_* / find_port 新名下全绿；`grep` 确认 src/ 内无 `site_selector_`、无 `find_port_atom`（`find_port_atom_by_node` 除外）。
- **边界**：RDKit 缺失环境下 examples 测试 skip 而非 fail；`import molpy` 不因惰性子模块新增而显著变慢（惰性 `__getattr__` 保证不在 import 时加载）。
- **领域验证**：本规格无新物理行为；以"零行为变化"为验证目标——除改名调用点外，既有测试断言一律不改。

## Out of scope

- 任何运行时行为变更（改名不提供弃用垫片，项目处 experimental 阶段；`ReactionPresetSpec.site_selector_*` 改名仅记 CHANGELOG）。
- 序列化（本链 02）、正确性修复（03）、性能（05）。
- `find_port_atom_by_node` 等其余 selector 命名整理。
- `docs/user-guide/` 其余 notebook 的内容重写（仅 04_crosslinking.ipynb 的 selector 替换）。
- 为全部 docs/ markdown 建立通用 doctest 框架（仅覆盖 builder.md / reacter.md 两个已知腐烂面）。
