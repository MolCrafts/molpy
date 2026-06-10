---
slug: builder-reacter-05-perf
criteria:
  - id: ac-001
    summary: record_intermediates=False 时合并结构拷贝有界（普通 0 / BondReact 1）
    type: performance
    evaluator_hint: "bench repo 未配置;以本地 pytest monkeypatch 计数代替外部基准"
    pass_when: |
      pytest tests/test_reacter/test_perf_copy_semantics.py -v 全绿:测试以
      monkeypatch 包装 Atomistic.copy 计数,断言单次 run(record_intermediates=False)
      中合并后结构的拷贝次数——普通 Reacter == 0、BondReactReacter == 1。
    status: pending
  - id: ac-002
    summary: reactants 快照由类钩子门控且契约文档化
    type: code
    pass_when: |
      pytest tests/test_reacter/test_perf_copy_semantics.py tests/test_reacter/test_base.py -v
      全绿:普通 Reacter 的 result.reactants is None;BondReactReacter 的
      result.reactants 为 Atomistic 且模板生成测试(tests/test_reacter/test_bond_react.py)不回归;
      Reacter.run docstring 含拷贝语义说明。
    status: verified
    last_checked: 2026-06-10
  - id: ac-003
    summary: detect_and_update_topology 邻接表构建一次、无全键扫描
    type: performance
    evaluator_hint: "计数型断言,非 wall-clock"
    pass_when: |
      pytest tests/test_reacter/test_topology_detector.py tests/test_reacter/test_utils.py -v
      全绿:单次 detect_and_update_topology 调用内邻接 dict 恰构建一次、
      find_neighbors 全键扫描回退路径调用次数为 0(计数器断言);
      带 adjacency 参数的 find_neighbors/get_bond_between/count_bonds
      与全扫描结果逐项一致。
    status: pending
  - id: ac-004
    summary: DP≥50 固定种子构建产物与黄金快照逐项一致
    type: runtime
    pass_when: |
      pytest tests/test_builder/test_polymer_build_perf.py -v -k determinism 全绿:
      固定种子 DP>=50 链的 atom/bond/angle/dihedral 计数及单体节点连接图
      与优化前生成的黄金快照完全相等。
    status: verified
    last_checked: 2026-06-10
  - id: ac-005
    summary: 每次连接的原子访问量以单体尺寸为界,不随链长增长
    type: performance
    evaluator_hint: "instrumentation fixture 统计单次连接访问原子数"
    pass_when: |
      pytest tests/test_builder/test_polymer_build_perf.py -v -k bounded 全绿:
      DP=50 构建中最后一次连接所访问的原子数 <= 常数 k × 单体原子数
      (k 在测试中固定,与 DP=10 时的计数同阶,比值 < 1.5)。
    status: pending
  - id: ac-006
    summary: _build_from_graph 用 union-find 分组且分支图分组正确
    type: code
    pass_when: |
      pytest tests/test_builder/test_polymer_build_perf.py -v -k group 全绿:
      V>=30 的分支图(含环闭合)构建后所有节点 id 映射到同一最终结构,
      产物与逐边身份扫描版本计数一致。
    status: verified
    last_checked: 2026-06-10
  - id: ac-007
    summary: _apply_transform 向量化且与逐原子参考实现数值等价
    type: code
    pass_when: |
      pytest tests/test_builder/test_polymer_placer.py -v 全绿:含新增等价性测试,
      向量化结果与参考逐原子实现 np.allclose(atol=1e-10),
      覆盖显式 pivot 与默认质心两分支。
    status: verified
    last_checked: 2026-06-10
  - id: ac-008
    summary: DP=200 规模冒烟构建在宽松上限内完成
    type: performance
    evaluator_hint: "@pytest.mark.slow;宽松 wall-clock 仅防灾难性回归"
    pass_when: |
      pytest tests/test_builder/test_polymer_build_perf.py -v -m slow 全绿:
      DP=200 链构建在测试内设定的宽松上限(60 s)内完成且产物计数自洽。
    status: pending
  - id: ac-009
    summary: 全量检查与测试套件通过（零行为变化）
    type: runtime
    pass_when: |
      ruff format --check src tests && ruff check src tests &&
      ty check src/molpy/ && pytest tests/ -m "not external" -v 全部通过。
    status: verified
    last_checked: 2026-06-10
  - id: ac-010
    summary: 复杂度与拷贝语义文档化
    type: docs
    pass_when: |
      find_neighbors docstring 标注 adjacency 参数与缺省 O(bonds) 复杂度;
      get_ports/get_all_ports docstring 标注 O(atoms);Reacter.run docstring
      含拷贝语义;docs/developer/performance-notes.md 存在且含构建循环
      复杂度预期(每次连接工作量以单体尺寸为界)一段。
    status: pending
---

# Acceptance criteria

性能判据（ac-001、ac-003、ac-005、ac-008）全部以本地 pytest 落地：`mol_project.bench` 仓库未配置，故不引用外部基准，优先计数型断言（拷贝次数、邻接构建次数、访问原子数），仅 ac-008 使用宽松 wall-clock 上限防灾难性回归。行为保持由 ac-004（黄金快照）与 ac-009（全套件）双重约束；唯一可观察契约变化（普通 `Reacter` 的 `result.reactants` 为 `None`）由 ac-002 绑定并要求文档化。
