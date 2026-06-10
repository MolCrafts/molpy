---
title: builder/reacter 构建循环性能优化（行为保持）
status: code-complete
created: 2026-06-10
---

# builder/reacter 构建循环性能优化（行为保持）

## Summary

聚合物构建循环目前随聚合度（DP）呈 O(N²) 退化：每次单体连接都对已累积的整条链做 O(N) 级别的全量工作——`Reacter.run` 对合并结构最多做 3 次完整深拷贝、`find_neighbors` 每次调用全量扫描所有键且每根新键被调用 6 次以上、端口查询与实体映射反查每次连接都重扫整个产物。本规格按 optimizer 评审的 7 项发现逐项消除这些热点，使 DP=100 量级链的构建时间显著下降，同时保证**零行为变化**：固定种子下优化前后构建产物的图结构、原子数、键/角/二面角计数完全一致，全测试套件保持绿色。本规格是 builder-reacter 五段链的最后一段，依赖 01（已删除死代码的 stochastic 路径——本规格**不**优化已删除代码）与 03（`_assign_react_ids` 位置修复）。

## Domain basis

本规格不引入新物理模型，属于行为保持的性能重构；正确性以"优化前后构建产物逐项一致"为唯一科学判据，无需文献依据。坐标变换向量化（`(coords - pivot) @ R.T + pivot + t`）与逐原子 `R @ (x - pivot) + pivot + t` 数学恒等。

## Design

**反应器拷贝语义（发现 1，HIGH）** — `Reacter.run`（`reacter/base.py:331,376,379,380`）中：

- `merged_copy`（`:380`）仅在 `record_intermediates=True` 分支被消费，改为在该分支内才创建；
- `merged_reactants_before_reaction`（`:379`）仅服务于模板生成（`BondReactReacter._generate_template` 读取 `result.reactants`，`bond_react.py:302`）。引入类级钩子 `_needs_reactants_snapshot: bool`（基类 `Reacter` 为 `False`，`BondReactReacter` 为 `True`），普通 `Reacter` 路径不再做该拷贝，`ReactionResult.reactants`（已是 `Atomistic | None`）在普通路径返回 `None`。这是唯一一处可观察的契约收紧，在 docstring 中文档化；`tests/test_reacter/test_base.py:48` 的 `result.reactants is merged` 断言相应更新。构建产物本身不受影响。

**邻接表复用（发现 2 + 7，HIGH/LOW）** — `detect_and_update_topology`（`topology_detector.py:386`）顶部一次性构建 `atom → list[neighbor]` 邻接 dict，贯穿 `_generate_angles_around_bond` / `_generate_dihedrals_from_bond`（`:272,275,281,289` 的嵌套 `find_neighbors` 调用）。`utils.py` 的 `find_neighbors`（`:49`）、`get_bond_between`（`:87`）、`count_bonds`（`:111`）增加可选 `adjacency` 参数：传入时 O(degree)，缺省时保持现有全扫描回退（向后兼容）。`_deduplicate_angles` / `_deduplicate_dihedrals` 所需的既有 tuple 集合（`:315-321,359-364,421-422`）从同一缓存派生，不再每次连接从头重建。

**端口查询局部化（发现 3 + 5，HIGH/MEDIUM）** — `get_all_ports` 的全部调用方在 stochastic 模块内，已由 01 删除；本规格优化**存活的** graph 构建路径：`_connect_monomers`（`core.py:331-332` 的 `get_ports_on_node` 全原子扫描）、`_transfer_unused_ports`（`:417` 的 `set(product.atoms)`）、`_preserve_node_ids`（`:448` 的全量反向实体映射）、`_cleanup_stale_ports`（`:469` 全产物扫描 + 逐原子 `find_neighbors`）全部限定到**当前反应 entity map 的定义域**（新加入单体的原子 + 锚点邻域），不随链长增长。`get_ports` / `get_all_ports`（`port_utils.py:41,57`）保留用于初始扫描，docstring 标注 O(N) 复杂度。

**分组映射（发现 4，MEDIUM）** — `_build_from_graph`（`core.py:259-282`）的逐边 `[nid for nid, mon in monomers.items() if mon is current_monomer]` 身份扫描（两处 + 环闭合扫描）替换为 union-find / group-id 映射，每条边摊还近 O(1)。

**坐标变换向量化（发现 6，MEDIUM）** — `placer.py:496-525` `_apply_transform` 的逐原子 Python 循环替换为 (N,3) NumPy 批量运算：`(coords - pivot) @ rotation.T + pivot + translation`，一次性读出、一次性写回 x/y/z。

**性能判据的落地方式** — `mol_project.bench` 仓库未配置，因此所有性能验收一律表达为**本地 pytest 断言**，并优先采用计数型断言（monkeypatch 统计 `Atomistic.copy` 调用次数、邻接表构建次数、每次连接访问的原子数）而非易抖动的 wall-clock；仅 DP=200 规模冒烟测试使用宽松的时间上限并标记 `@pytest.mark.slow`。

## Files to create or modify

- src/molpy/reacter/base.py
- src/molpy/reacter/bond_react.py
- src/molpy/reacter/utils.py
- src/molpy/reacter/topology_detector.py
- src/molpy/builder/polymer/core.py
- src/molpy/builder/polymer/port_utils.py
- src/molpy/builder/polymer/placer.py
- tests/test_reacter/test_perf_copy_semantics.py (new)
- tests/test_reacter/test_utils.py
- tests/test_reacter/test_topology_detector.py
- tests/test_reacter/test_base.py
- tests/test_builder/test_polymer_build_perf.py (new)
- tests/test_builder/test_polymer_placer.py
- docs/developer/performance-notes.md (new)

## Tasks

- [x] Write failing instrumentation tests for Reacter.run copy counts (tests/test_reacter/test_perf_copy_semantics.py)：monkeypatch 计数 `Atomistic.copy`，断言 `record_intermediates=False` 时合并结构拷贝次数——普通 `Reacter` 为 0、`BondReactReacter` 为 1；`record_intermediates=True` 路径中间态仍完整
- [x] Implement copy gating in Reacter.run（src/molpy/reacter/base.py、src/molpy/reacter/bond_react.py）：`merged_copy` 移入 `record_intermediates` 分支；`_needs_reactants_snapshot` 类钩子控制 reactants 快照；更新 tests/test_reacter/test_base.py 的 `result.reactants` 断言；Google 风格 docstring 写明拷贝语义
- [x] Write failing tests for adjacency-based neighbor queries（tests/test_reacter/test_utils.py、tests/test_reacter/test_topology_detector.py）：`find_neighbors`/`get_bond_between`/`count_bonds` 带预构建邻接表时结果与全扫描逐项一致；`detect_and_update_topology` 单次调用内全键扫描次数为 0（计数器断言）
- [x] Implement adjacency map in topology detection（src/molpy/reacter/utils.py、src/molpy/reacter/topology_detector.py）：`detect_and_update_topology` 顶部构建一次邻接 dict 并贯穿；utils 三函数加可选 `adjacency` 参数；去重用既有 angle/dihedral tuple 集合从同一缓存派生
- [x] Write failing determinism, scaling and placer-equivalence tests（tests/test_builder/test_polymer_build_perf.py、tests/test_builder/test_polymer_placer.py）：DP≥50 固定种子链 fixture 的 atom/bond/angle/dihedral 计数与节点连接图对照黄金快照；每次连接访问原子数有界（不随 DP 增长）；`_apply_transform` 向量化与参考逐原子实现 `allclose(atol=1e-10)`；DP=200 冒烟测试 `@pytest.mark.slow` + 宽松时间上限
- [x] Implement localized port queries in graph build loop（src/molpy/builder/polymer/core.py、src/molpy/builder/polymer/port_utils.py）：`_connect_monomers`/`_transfer_unused_ports`/`_preserve_node_ids`/`_cleanup_stale_ports` 限定到当前反应 entity map 定义域；`get_ports`/`get_all_ports` 保留用于初始扫描并标注复杂度
- [x] Implement union-find group map in _build_from_graph（src/molpy/builder/polymer/core.py:259-282）：替换逐边身份扫描与环闭合扫描，补充分支图（V≥30）分组正确性单元测试
- [x] Implement vectorized _apply_transform（src/molpy/builder/polymer/placer.py）：(N,3) 批量 `(coords - pivot) @ R.T + pivot + t`，一次读出/写回
- [x] Add docstrings and developer note（src/molpy/reacter/utils.py、src/molpy/reacter/base.py、docs/developer/performance-notes.md）：公共 helper 复杂度标注（`find_neighbors` 注明 adjacency 参数）、`Reacter.run` 拷贝语义、开发者文档中一段构建循环复杂度预期
- [x] Run full check + test suite：`ruff format src tests && ruff check src tests && ty check src/molpy/ && pytest tests/ -m "not external" -v`
- [x] Hygiene cleanup (/mol:simplify: 9 fixes — dead params, docstring Args sync, strict zip; suite green)
- [x] Note: 黄金快照的 inter-monomer 节点 ID 改为按首现顺序归一化（parser 节点 ID 来自进程级全局计数器，原快照对测试顺序敏感——经 stash A/B 证实该缺陷在优化前代码同样存在，归一化后顺序无关；DP=200 冒烟 30.5s → 14.9s）

## Testing strategy

- **快乐路径（行为保持）**：固定种子 DP≥50 链构建，优化前后 atom/bond/angle/dihedral 计数与单体连接图逐项一致（黄金快照，在实现首个优化前生成）；全套件 `pytest tests/ -m "not external"` 优化前后均绿。
- **边界情况**：环闭合（`left is right`）路径的拷贝门控与 entity map；`record_intermediates=True` 时中间态记录不回归；`find_neighbors` 不传 adjacency 的回退路径；空 port、单单体（无连接）构建；`_apply_transform` 显式 pivot 与默认质心两种分支。
- **性能验收（本地 pytest 计数优先，bench 仓库未配置）**：`Atomistic.copy` 每次连接调用计数；邻接表每次 `detect_and_update_topology` 构建恰一次、全键扫描为 0；每次连接访问原子数以单体尺寸为界；DP=200 `@pytest.mark.slow` 冒烟仅设宽松 wall-clock 上限。
- **契约收紧验证**：普通 `Reacter` 的 `result.reactants is None`，`BondReactReacter` 的 `result.reactants` 仍为可生成模板的完整反应物快照（模板生成测试不回归）。

## Out of scope

- placement 的算法重设计（placer 无成对碰撞检测，不需要 cell list）；
- 并行化、Rust offload；
- stochastic 死代码路径（已由 01 删除，不做任何优化）；
- `_assign_react_ids` 的位置问题（由 03 处理）；
- 任何公共 API 行为变更（`ReactionResult.reactants` 在普通路径返回 `None` 是唯一的、文档化的契约收紧）。
