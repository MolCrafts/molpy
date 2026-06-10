---
title: REACTER 模板科学正确性与变异卫生修复
status: code-complete
created: 2026-06-10
---

# REACTER 模板科学正确性与变异卫生修复

## Summary

修复 `BondReactReacter` 生成的 LAMMPS `fix bond/react` 模板中的科学正确性缺陷：post 模板目前完全丢失 improper（`_build_post` 只重建 bonds/angles/dihedrals，而 pre 模板经 `extract_subgraph` 完整克隆了 improper），导致 fix bond/react 在反应区域删除全部 improper，对 OPLS-AA/GAFF 的 sp2 中心产生错误的 post 拓扑。同时修复四类伴随缺陷：InitiatorIDs 由 set 构建导致顺序不确定且不校验"恰好 2 个"；edge 原子的 pre/post 类型与电荷一致性从不验证（`_incremental_typify` 会重打反应位点邻居的类型，是 fix bond/react 拒绝 map 的常见原因）；`BondReactReacter.run` 在 `super().run()` 复制之前就对调用方持有的 `left`/`right` 原结构打上 `react_id`，违反"helper 不得意外变异调用方结构"的项目准则；以及缺少 pre/post 电荷守恒检查。完成后，生成的模板满足 REACTER 文献与 LAMMPS 文档要求的全部 map 不变量，且 `run()` 不再污染调用方输入。

## Domain basis

- **REACTER 协议**：Gissinger, Jensen, Wagner, *Modeling chemical reactions in classical molecular dynamics simulations*, Polymer 128 (2017) 211–217, DOI: 10.1016/j.polymer.2017.06.038；Gissinger et al., Macromolecules 53 (2020) 9953–9961, DOI: 10.1021/acs.macromol.0c02012。模板要求：
  - Equivalences 必须是 pre→post 原子的双射；
  - InitiatorIDs 恰好 2 个（成键的两个 anchor），顺序确定；
  - EdgeIDs 原子在 pre/post 必须**完全一致**（类型与电荷不变），且 initiator 的第一近邻壳层必须包含在模板内（initiator 不得落在模板边界上）；
  - DeleteIDs 原子在 post 中保留但被标记删除。
- **LAMMPS fix bond/react 文档**：https://docs.lammps.org/fix_bond_react.html — post 模板拓扑替换 pre 模板拓扑，post 缺失的 improper 即被物理删除。
- **Improper 的物理意义**：OPLS-AA（Jorgensen et al., J. Am. Chem. Soc. 118 (1996) 11225, DOI: 10.1021/ja9621760）与 GAFF（Wang et al., J. Comput. Chem. 25 (2004) 1157, DOI: 10.1002/jcc.20035）用 improper 维持 sp2 中心（酰胺、乙烯基、芳环）的平面性；丢失 improper 导致平面基团出现非物理的锥化。
- **电荷守恒**：反应模板 pre/post 总电荷之差应为 0（单位：元电荷 e）；数值容差取模块常量 `CHARGE_CONSERVATION_TOL = 1e-6` e，超出时以 warning 级别日志提示（电荷由固定点电荷力场给定，模板不重新分配总电荷）。

## Design

- **TopologyDetector 增加 improper 枚举**（`topology_detector.py`）：
  - `_remove_topology_with_removed_atoms` 扩展为同时移除涉及被删原子的 `Improper`（与现有 Angle/Dihedral 同模式，返回值元组增加 `removed_impropers`）；
  - 新增 `_generate_impropers_around_atoms(assembly, atoms)`：对每个受影响原子（新键端点及其近邻，复用 `_get_affected_atoms`），若其键合近邻恰好为 3 个，生成候选 `Improper(center, n1, n2, n3)`（center 为 i 位，与 `core.atomistic.Improper` 约定一致）；
  - 新增 `_deduplicate_impropers`：以 center 为锚、近邻集合无序比较去重；
  - `detect_and_update_topology` 返回值扩展为包含 `new_impropers` / `removed_impropers`，调用方（`base.py` 的 `ReactionResult` 装配处）同步接收。
- **`_build_post` 增加 improper 分支**（`bond_react.py:416-462`）：`_add_topo` 新增 `"improper"` 类型走 `post.def_improper`，沿用既有 deleted-atom 守卫（端点含 removed react_id 即跳过）；遍历 `reactants.impropers`（覆盖未受反应影响、由 merge 携带的 improper）与 `product.impropers`（覆盖 TopologyDetector 新生成的 improper）。
- **`_incremental_typify` improper 分支**（`base.py`）：若 typifier 暴露 `improper_typifier` 则对新 improper 及涉及 modified atoms 的既有 improper 重打类型（与 Step 4/5 同模式）；否则 improper 携带原 data 原样复制，不报错。
- **确定性 InitiatorIDs**（`bond_react.py:131-136` 的 `write_map`）：`initiator_atoms` 在 `_generate_template` 中已是有序列表 `[anchor_L, anchor_R]`；`write_map` 改为按该顺序输出（左 anchor 在前），不再经过 set；断言恰好 2 个且均在 `pre_rid_to_idx` 中，否则抛 `ValueError` 并点名落在模板半径之外的 anchor；若任一 initiator 同时出现在 `edge_atoms` 中（第一近邻壳层不在模板内），同样抛错并建议增大 `radius`。
- **Edge 原子 pre/post 一致性验证**（`bond_react.py` 的 `_generate_template` 末尾）：对每个 edge 原子的 react_id，比较 pre 与 post 原子的 `type` 与 `charge`；不一致即抛 `ValueError`，消息列出原子、两侧取值，并建议增大 `radius`。默认抛错（不提供静默冻结）。
- **变异卫生修复**（`bond_react.py:247-293`）：`BondReactReacter.run` 在打 `react_id` 之前先 `left.copy()` / `right.copy()`，按位置对应（同 `base.py:142-160` 的 `_prepare_reactants` 模式）把 `port_atom_L/R` 解析到副本中，再对副本打 `react_id` 并将副本传给 `super().run()`（基类内部的二次 copy 会深拷贝 data dict，`react_id` 随之保留）。调用方的 `left`/`right` 原子在 `run()` 之后不得出现 `react_id` 键。
- **电荷守恒检查**（`bond_react.py`）：模块常量 `CHARGE_CONSERVATION_TOL = 1e-6`；模板构建完成后比较 pre/post 等价原子总电荷，`|Δq| > CHARGE_CONSERVATION_TOL` 时通过 `logging` 输出 warning（不抛错），缺失 `charge` 字段的原子按 0 处理并在消息中注明。

## Files to create or modify

- src/molpy/reacter/topology_detector.py — improper 枚举、去重、删除清理；`detect_and_update_topology` 返回值扩展
- src/molpy/reacter/bond_react.py — `_build_post` improper 分支、确定性 InitiatorIDs、edge 验证、电荷守恒、`run` 变异修复、`CHARGE_CONSERVATION_TOL`、docstring 更新
- src/molpy/reacter/base.py — `detect_and_update_topology` 新返回值的接收；`_incremental_typify` 可选 improper 分支
- tests/test_reacter/test_topology_detector.py — improper 生成/删除/去重单元测试
- tests/test_reacter/test_bond_react.py — improper 平移、InitiatorIDs 确定性、edge 验证、变异卫生、电荷守恒及 REACTER map 不变量科学验证测试
- docs/user-guide/04_crosslinking.ipynb — 模板有效性保证小节

## Tasks

- [x] Write failing tests for improper regeneration in TopologyDetector and post template (tests/test_reacter/test_topology_detector.py, tests/test_reacter/test_bond_react.py)
- [x] Implement improper enumeration, deduplication, and removed-atom cleanup in TopologyDetector (src/molpy/reacter/topology_detector.py), wiring new return values through Reacter.run (src/molpy/reacter/base.py)
- [x] Implement improper section in BondReactReacter._build_post mirroring bonds/angles/dihedrals incl. deleted-atom guard (src/molpy/reacter/bond_react.py)
- [x] Write failing tests for deterministic InitiatorIDs, edge type/charge validation, charge conservation warning, and caller-input mutation hygiene (tests/test_reacter/test_bond_react.py)
- [x] Implement ordered InitiatorIDs (left anchor first) with exactly-2 assertion and initiator-on-edge error naming the offending anchor (src/molpy/reacter/bond_react.py)
- [x] Implement edge-atom pre/post type+charge validation raising ValueError, and charge-conservation warning with module constant CHARGE_CONSERVATION_TOL = 1e-6 (src/molpy/reacter/bond_react.py)
- [x] Fix BondReactReacter.run to stamp react_id on internal copies only, resolving port atoms positionally so caller-owned left/right stay untouched (src/molpy/reacter/bond_react.py)
- [x] Verify REACTER map invariants with a scientific validation test on an sp2-center reaction: bijective equivalences, byte-identical .map across two runs, improper parity under equivalence mapping, edge type/charge parity (tests/test_reacter/test_bond_react.py)
- [x] Add docstring per google style with units and literature refs (Gissinger 2017/2020, LAMMPS fix bond/react docs) to BondReactReacter, _build_post, TopologyDetector, plus a template-validity subsection in docs/user-guide/04_crosslinking.ipynb
- [x] Run full check + test suite
- [x] Hygiene cleanup (/mol:simplify: 5 fixes — dead helpers, stale aliases/import/noqa, docstring sync; suite green)

## Testing strategy

- **Happy path**：现有 `tests/test_reacter/` 全量回归通过；含 improper 的反应物经 `run()` 后 pre/post 模板均含 improper 区段，`write_map` 输出格式不变（仅顺序确定化）。
- **单元测试**：
  - TopologyDetector：3 近邻受影响原子生成 improper；4 近邻/2 近邻不生成；与既有 improper 去重；涉及被删原子的 improper 被移除；
  - `_build_post`：deleted-atom 守卫对 improper 生效（端点含被删原子的 improper 不进入 post）；
  - InitiatorIDs：左 anchor 恒在前；anchor 落在半径外或 initiator 即 edge 原子时抛 `ValueError`，消息含 anchor 标识与增大 `radius` 的建议；
  - 变异卫生：`run()` 后调用方 `left`/`right` 的所有原子 `"react_id" not in atom.data`（沿用 CLAUDE.md 的 `.copy()` 隔离测试模式）；
  - 电荷守恒：构造 `|Δq| > 1e-6` 的模板触发 warning 日志（`caplog` 断言），守恒时无 warning。
- **边界情形**：edge 原子类型或电荷 pre/post 不一致 → `ValueError` 且消息可操作；pre/post 原子集不一致仍按既有路径报错。
- **领域验证（scientific）**：在 sp2 中心反应（基于 `tests/test_reacter/test_bond_react.py` 既有 fixture 构造的乙烯基/酰胺型小体系）上断言 REACTER 不变量：Equivalences 双射；pre 中不涉及被删原子的每个 improper 经等价映射在 post 中存在对应 improper，未受反应触及的 improper 数目 pre/post 相等；连续两次以等价新输入运行产生字节相同的 `.map` 文件；edge 原子 type/charge 严格相等。项目未配置 bench 仓库，科学判据全部以本地 pytest + 文献推导断言表达。

## Out of scope

- 模板序列化位置与统一 writer 归属（本链 02 spec）。
- 公共 API 命名调整（本链 04 spec）。
- 性能优化（本链 05 spec）。
- LAMMPS map 的 `Constraints` / `ChiralIDs` 区段支持（文档化为已知限制）。
- edge 原子"冻结回 pre 值"的可选 flag（本 spec 选定默认抛错；如有需求另立 spec）。
- `utils.create_atom_mapping` 的位置回退容差问题（主路径未使用，LOW，留待后续）。
