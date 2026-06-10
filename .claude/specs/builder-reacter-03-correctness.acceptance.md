---
slug: builder-reacter-03-correctness
criteria:
  - id: ac-001
    summary: Post template carries impropers matching pre under equivalence mapping
    type: scientific
    evaluator_hint: "pytest tests/test_reacter/test_bond_react.py -k improper -v"
    pass_when: |
      On an sp2-center reaction fixture, every improper in template.pre whose
      endpoints exclude deleted atoms has a corresponding improper in
      template.post under the react_id equivalence map, and the count of
      impropers untouched by the reaction is equal pre vs post; the test
      passes via `pytest tests/test_reacter/test_bond_react.py -k improper -v`.
    status: pending
  - id: ac-002
    summary: TopologyDetector enumerates, dedups, and cleans up impropers
    type: code
    evaluator_hint: "pytest tests/test_reacter/test_topology_detector.py -k improper -v"
    pass_when: |
      Unit tests assert: a 3-neighbor affected atom yields Improper(center, n1, n2, n3);
      2- and 4-neighbor atoms yield none; candidates duplicate to existing impropers
      are dropped; impropers crossing removed atoms are removed by
      _remove_topology_with_removed_atoms. All pass via
      `pytest tests/test_reacter/test_topology_detector.py -k improper -v`.
    status: verified
    last_checked: 2026-06-10
  - id: ac-003
    summary: .map file is deterministic with ordered InitiatorIDs
    type: runtime
    evaluator_hint: "pytest tests/test_reacter/test_bond_react.py -k determinis -v"
    pass_when: |
      Running BondReactReacter.run + write_map twice on equivalent fresh inputs
      produces byte-identical .map files; the InitiatorIDs section contains
      exactly 2 entries with the left anchor first. Verified by
      `pytest tests/test_reacter/test_bond_react.py -k determinis -v`.
    status: verified
    last_checked: 2026-06-10
  - id: ac-004
    summary: Invalid initiator configuration raises actionable ValueError
    type: code
    evaluator_hint: "pytest tests/test_reacter/test_bond_react.py -k initiator -v"
    pass_when: |
      When an anchor falls outside the template radius, or an initiator atom is
      also an edge atom, ValueError is raised and the message names the offending
      anchor and suggests increasing radius. Verified by
      `pytest tests/test_reacter/test_bond_react.py -k initiator -v`.
    status: verified
    last_checked: 2026-06-10
  - id: ac-005
    summary: Edge atoms validated for identical pre/post type and charge
    type: scientific
    evaluator_hint: "pytest tests/test_reacter/test_bond_react.py -k edge -v"
    pass_when: |
      Templates from the validation fixture have every edge atom with strictly
      equal type and charge pre vs post; a constructed mismatch raises ValueError
      whose message lists the atom, both values, and suggests a larger radius.
      Verified by `pytest tests/test_reacter/test_bond_react.py -k edge -v`.
    status: pending
  - id: ac-006
    summary: run() never mutates caller-owned left/right structures
    type: code
    evaluator_hint: "pytest tests/test_reacter/test_bond_react.py -k mutat -v"
    pass_when: |
      After BondReactReacter.run(left, right, ...), no atom of the caller's
      left or right has a 'react_id' key in atom.data, and atom counts of the
      inputs are unchanged. Verified by
      `pytest tests/test_reacter/test_bond_react.py -k mutat -v`.
    status: verified
    last_checked: 2026-06-10
  - id: ac-007
    summary: Charge conservation check warns beyond named tolerance constant
    type: code
    evaluator_hint: "pytest tests/test_reacter/test_bond_react.py -k charge -v"
    pass_when: |
      Module constant CHARGE_CONSERVATION_TOL == 1e-6 exists in
      src/molpy/reacter/bond_react.py; a template with |sum(q_post) - sum(q_pre)|
      above the tolerance emits a logging warning (asserted via caplog), and a
      conserving template emits none. Verified by
      `pytest tests/test_reacter/test_bond_react.py -k charge -v`.
    status: verified
    last_checked: 2026-06-10
  - id: ac-008
    summary: REACTER map invariants hold on scientific validation fixture
    type: scientific
    evaluator_hint: "pytest tests/test_reacter/test_bond_react.py -k invariant -v"
    pass_when: |
      On the sp2-center validation reaction: equivalences are a bijection between
      pre and post atom sets; exactly 2 initiators; edge and initiator ID sets are
      disjoint; deleted atoms appear in DeleteIDs. Verified by
      `pytest tests/test_reacter/test_bond_react.py -k invariant -v`.
    status: pending
  - id: ac-009
    summary: Docstrings and tutorial document template-validity invariants
    type: docs
    pass_when: |
      Google-style docstrings of BondReactReacter, BondReactReacter._build_post,
      and TopologyDetector reference Gissinger 2017 (10.1016/j.polymer.2017.06.038),
      Gissinger 2020 (10.1021/acs.macromol.0c02012), and the LAMMPS fix bond/react
      docs URL, and state the edge/initiator/improper invariants;
      docs/user-guide/04_crosslinking.ipynb gains a subsection on template
      validity guarantees. Verified by grep for the DOIs/URL in the three sources
      and the notebook.
    status: pending
  - id: ac-010
    summary: Full check and local test suite pass
    type: runtime
    pass_when: |
      `ruff check src tests && ty check src/molpy/` exits 0 and
      `pytest tests/ -m "not external" -v` reports no failures.
    status: verified
    last_checked: 2026-06-10
---

# Acceptance criteria

- **ac-001 / ac-002** — improper 修复的核心科学判据与单元判据：post 模板不再丢失 improper，TopologyDetector 的枚举仅作用于受影响原子并正确清理被删原子的 improper。
- **ac-003 / ac-004** — InitiatorIDs 的确定性与防御：set 顺序不确定性消除，半径不足不再静默产出无效模板。
- **ac-005** — REACTER 的 edge 原子一致性硬性要求（fix bond/react 拒绝 map 的常见原因）成为构建期错误。
- **ac-006** — 变异卫生：与 `base.py` 先复制再操作的契约对齐，调用方输入零污染。
- **ac-007** — 电荷守恒为 warn 级（固定点电荷力场下模板不重分配电荷），容差为命名常量。
- **ac-008** — 综合不变量回归锚点，防止后续 spec（02 序列化、04 命名）破坏 map 语义。
- **ac-009 / ac-010** — 文档与全量门禁。

**Note on chain order**: 本 spec 依赖链中 01（清理）与 02（统一 io writer）先落地；改动假设 `write_map` 与 `BondReactTemplate.write` 仍为 map 输出入口。若 02 已迁移序列化位置，ac-003 的判据 command 不变，仅 `.map` 生成调用点随 02 调整。

**Bench note**: `mol_project` 未配置 bench 仓库，故全部 `type: scientific` 判据以本地 pytest + 文献推导断言（双射性、improper 计数守恒、字节级 map 可复现）表达，无需外部 bench。
