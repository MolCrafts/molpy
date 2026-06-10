---
title: REACTER 模板序列化下沉至 io 层并统一类型映射
status: code-complete
created: 2026-06-10
---

# REACTER 模板序列化下沉至 io 层并统一类型映射

## Summary

当前 LAMMPS fix bond/react 模板的序列化逻辑分裂在两个层之间：`BondReactTemplate.write()`（reacter 层）使用按模板对（pre/post）局部编号的类型映射并手写 `.map` 文件，而 `molpy.io.write_lammps_bond_react_system`（io 层）重新实现了全系统统一编号，且通过 `tpl._assign_atom_ids()`、`tpl.write_map()` 侵入模板私有方法。两条路径产出互不兼容的文件——路径 A 的模板局部类型 ID 与系统 data 文件不一致，对真实 fix bond/react 运行是静默错误。本规格将全部序列化职责（`.map` 写出 + 类型统一）下沉到新模块 `src/molpy/io/data/lammps_bond_react.py`，`BondReactTemplate` 退化为纯数据对象，`write_lammps_bond_react_system` 保持唯一用户出口且对该路径字节级保持现有输出（golden-file 验证）。同时消除 `reacter → io` 的层违规导入（`bond_react.py:19` 的顶层 `import molpy as mp`）与 `io → reacter` 私有方法访问。`BondReactTemplate.write()`/`write_map()` 直接删除，不留弃用垫片（项目处于 experimental 阶段）。本规格为 builder-reacter 五段链的第 2 段，依赖第 1 段（dead-code 清理）先落地；模板内容修正（impropers/edge 校验）留给第 3 段。

## Domain basis

本规格不涉及物理方程，约束来自 LAMMPS fix bond/react 的文件格式契约：

- 模板 `.mol` 文件中的数值类型 ID 必须与系统 data 文件的类型 ID 完全一致，否则 fix bond/react 无法匹配反应位点（LAMMPS 文档：https://docs.lammps.org/fix_bond_react.html）。
- `.map` 文件包含 header 计数（`N equivalences` / `N edgeIDs` / `N deleteIDs`）与 `InitiatorIDs` / `EdgeIDs` / `DeleteIDs` / `Equivalences` 四节，原子 ID 为模板内 1-based 序号。
- REACTER 方法学参考：
  - Gissinger, J. R.; Jensen, B. D.; Wise, K. E. *Modeling chemical reactions in classical molecular dynamics simulations.* Polymer 128, 211–217 (2017). DOI: 10.1016/j.polymer.2017.09.038
  - Gissinger, J. R.; Jensen, B. D.; Wise, K. E. *REACTER: A Heuristic Method for Reactive Molecular Dynamics.* Macromolecules 53(22), 9953–9961 (2020). DOI: 10.1021/acs.macromol.0c02012
  - reacter.org（官方工具站）

## Design

**`BondReactTemplate`（reacter 层）退化为纯数据对象。** 其 dataclass 公有字段（`pre`、`post`、`initiator_atoms`、`edge_atoms`、`deleted_atoms`、`pre_react_id_to_atom`、`post_react_id_to_atom`）已经覆盖序列化所需的全部信息，无需新增访问器。变更点：

1. `_assign_atom_ids()` 改名为公有方法 `assign_atom_ids()`：确定性地按 `pre.atoms` / `post.atoms` 迭代顺序赋 1-based `id`，docstring 明确该确定性约定。io 层只调用公有 API。
2. 删除 `write()`（bond_react.py:66-99）、`write_map()`（bond_react.py:101-167）、模块级 `_unify_type_mappings`（bond_react.py:472-516）与 `_ensure_typified`（bond_react.py:519-527，已验证其唯一调用方是被删除的 `write()` 路径，见 grep：仅 bond_react.py:87-88 调用）。
3. 删除 `bond_react.py:19` 的顶层 `import molpy as mp`（该导入使 `import molpy.reacter` 急切初始化 io/engine/adapter 全包）。
4. `reacter/base.py:15` 对 `TypifierBase` 的运行时导入降为 `TYPE_CHECKING` 守卫（typifier 实例由调用方传入，仅注解需要；需补 `from __future__ import annotations`）。处理后 `src/molpy/reacter/` 内对其他 molpy 子包的运行时依赖仅剩 `core`（typifier 协议依赖降为类型注解），符合 reacter 仅依赖 core 的层规则。
5. 更新 `BondReactTemplate` 类 docstring 与 `BondReactReacter` docstring 示例（bond_react.py:223 当前演示 `result.template.write(...)`），改为指向 `molpy.io.write_lammps_bond_react_system`。

**新 io 模块 `src/molpy/io/data/lammps_bond_react.py`**，与 `io/data/lammps_molecule.py` 并列（沿用既有 per-format io 模块模式）：

- `write_bond_react_map(template, path)` — 由 bond_react.py:101-167 迁入：用 react_id 建 pre/post 1-based 索引、校验 pre/post 原子集合一致、写 header 计数与四节。仅读模板公有字段 + 调用 `assign_atom_ids()`。
- 统一类型映射辅助函数 — 合并 bond_react.py:472-516（pre/post 双帧版）与 writers.py:369-394（全系统版）为单一实现：对任意帧序列按 section 收集字符串类型名、排序、生成 `{name: 1-based id}` 映射；系统路径行为以 writers.py 现实现为准（跳过纯数字/None 类型、丢弃未识别类型行并 warn）。
- Google 风格 docstring，标注文件格式参考（LAMMPS fix bond/react 文档 + reacter.org）。

**`write_lammps_bond_react_system`（io/writers.py:320-449）保持唯一用户出口**，内部改为委托新模块：`tpl._assign_atom_ids()`（writers.py:365）→ `tpl.assign_atom_ids()`；`tpl.write_map(...)`（writers.py:447）→ `write_bond_react_map(tpl, ...)`；类型统一段（writers.py:369-394）→ 新模块辅助函数。该路径产出（`.data`/`.ff`/`*_pre.mol`/`*_post.mol`/`*.map`）与重构前字节级一致，由 golden-file 测试约束。`write_bond_react_map` 同时从 `molpy.io` 导出（与 `write_lammps_molecule` 同列，io/__init__.py:138-139、181-182 处）。

**破坏性变更**：`BondReactTemplate.write()`/`write_map()` 移除。experimental 阶段不做弃用垫片；在 `docs/changelog.md` 写迁移说明。已验证 `docs/user-guide/04_crosslinking.ipynb` 仅使用 `write_lammps_bond_react_system`（ipynb:312），无需改动，仅在任务中复核；`tests/test_reacter/test_bond_react.py:569,713` 两处 `template.write()` 调用需迁移。

## Files to create or modify

- src/molpy/io/data/lammps_bond_react.py (new)
- src/molpy/io/writers.py
- src/molpy/io/__init__.py
- src/molpy/reacter/bond_react.py
- src/molpy/reacter/base.py
- tests/test_io/test_data/test_lammps_bond_react.py (new)
- tests/test_io/test_data/golden/bond_react/ (new，golden 输出夹具目录)
- tests/test_reacter/test_import_hygiene.py (new)
- tests/test_reacter/test_bond_react.py
- docs/changelog.md

## Tasks

- [x] Write golden-file test for `write_lammps_bond_react_system` (tests/test_io/test_data/test_lammps_bond_react.py)：在测试内确定性构建小型 frame + forcefield + BondReactTemplate，用重构前 HEAD 代码生成输出并提交为夹具（tests/test_io/test_data/golden/bond_react/），测试断言重构后输出与夹具逐文件字节相等（注：sys.ff 因 TypeBucket set 序不定按行多重集比较，其余 4 文件字节级；夹具用 4849ada 代实现生成）
- [x] Write failing tests for `write_bond_react_map` (tests/test_io/test_data/test_lammps_bond_react.py)：header 计数行、InitiatorIDs/EdgeIDs/DeleteIDs/Equivalences 四节、1-based ID、pre/post 原子集合不一致时抛 ValueError（RED：ModuleNotFoundError）
- [x] Write failing import-hygiene tests (tests/test_reacter/test_import_hygiene.py)：子进程 `python -c "import sys, molpy.reacter; assert 'molpy.io' not in sys.modules"`；AST 断言 src/molpy/reacter/ 内无顶层 molpy 导入（RED：molpy/__init__ 急切导入 io + bond_react.py:19）
- [x] Implement `write_bond_react_map` + 统一类型映射辅助函数 in src/molpy/io/data/lammps_bond_react.py（合并 bond_react.py:101-167,472-516 与 writers.py:369-394），含 Google docstring 与 LAMMPS/reacter.org 格式参考
- [x] Implement delegation in src/molpy/io/writers.py：`write_lammps_bond_react_system` 改调新模块与公有 `assign_atom_ids()`，移除 `tpl._assign_atom_ids()`/`tpl.write_map()` 私有访问；在 src/molpy/io/__init__.py 导出 `write_bond_react_map`
- [x] Implement pure-data `BondReactTemplate` in src/molpy/reacter/bond_react.py：`_assign_atom_ids` → 公有 `assign_atom_ids()`，删除 `write`/`write_map`/`_unify_type_mappings`/`_ensure_typified` 与顶层 `import molpy as mp`，更新类与示例 docstring 指向 io 写出器
- [x] Implement TYPE_CHECKING-only `TypifierBase` import in src/molpy/reacter/base.py（补 `from __future__ import annotations`）
- [x] Migrate tests/test_reacter/test_bond_react.py:569,713 from `template.write()` to `molpy.io` 写出路径，保留原有断言意图
- [x] Add migration note to docs/changelog.md（`BondReactTemplate.write/write_map` 移除 → `mp.io.write_lammps_bond_react_system` / `mp.io.write_bond_react_map`）；复核 docs/user-guide/04_crosslinking.ipynb 仍仅使用系统写出器并在 PR 描述中声明
- [x] Run full check + test suite（`ruff format src tests && ruff check src tests && ty check src/molpy/ && pytest tests/ -m "not external" -v`）
- [x] Hygiene cleanup (/mol:simplify: 4 fixes — dangling Sequence ref, stale comment, List→list, import grouping; suite green)
- [x] Deviation note: molpy/__init__.py 改为 PEP 562 惰性子模块加载（spec 文件清单之外的必要改动——molpy/__init__ 原本急切导入 io，使 ac-003 在 reacter 侧任何修改下都不可满足；变更已记入 changelog）

## Testing strategy

- **Happy path**：golden-file 测试覆盖 `write_lammps_bond_react_system` 全部产物（`.data`、`.ff`、`*_pre.mol`、`*_post.mol`、`*.map`）字节级等于重构前输出；`write_bond_react_map` 单元测试覆盖四节内容与 1-based 编号。
- **Edge cases**：pre/post 原子集合不一致抛 `ValueError`；模板含未识别类型行时丢弃并 `warn`（沿用 writers.py:425-436 现行为）；空 edge/delete 列表时 header 计数为 0 且节为空。
- **回归/层级**：导入卫生测试（`molpy.reacter` 不触发 `molpy.io` 加载）+ grep 层断言；迁移后的 test_bond_react.py 两个写出测试保持原断言（文件存在、.map 节标题、pre/post 类型映射统一）。
- **领域验证**：golden `.map`/`.mol` 夹具人工对照 LAMMPS fix bond/react 文档格式一次（节名、计数行、ID 基准），结论记录在测试 docstring 中。

## Out of scope

- 模板内容正确性修正（impropers 缺失、edge 原子校验）→ 链中第 3 段（builder-reacter-03）。
- 公共 API 改名 → 第 4 段（builder-reacter-04）。
- 性能优化（类型映射循环向量化等）→ 第 5 段（builder-reacter-05）。
- `Path A` 单模板独立写出的"修复"：不修复其类型编号语义，直接删除该路径；独立写 `.map` 的需求由 `write_bond_react_map` 满足。
- 弃用垫片 / DeprecationWarning（experimental 阶段约定不做）。
