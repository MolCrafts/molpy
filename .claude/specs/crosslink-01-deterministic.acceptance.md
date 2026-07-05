---
slug: crosslink-01-deterministic
criteria:
  - id: ac-001
    summary: Crosslinker.apply 是 immutable 边界（copy-once，返回同子类新图）
    type: code
    pass_when: |
      out = DeterministicCrosslinker(rxn, cutoff=5.0).apply(g)：入参 g 的原子/键/拓扑计数前后一致；
      out is not g，type(out) is type(g)（Atomistic→Atomistic、CoarseGrain→CoarseGrain），含预期新交联键。
    status: verified
    last_checked: 2026-07-05

  - id: ac-002
    summary: 匹配/应用/图编辑全走 molrs，molpy 不自造引擎
    type: code
    pass_when: |
      Crosslinker 用 molrs.Reaction（reactant_patterns/forming_bonds/apply）+ molrs.SmartsPattern.find_matches_mapped
      + molrs.NeighborList + topo_distances + molrs Atomistic.copy；
      grep src/molpy/builder/crosslink/ 无自造 SMARTS 解析/子图同构、无自造键/拓扑生成、无 MatchSet/Block 存储、无 classmethod。
    status: verified
    last_checked: 2026-07-05

  - id: ac-003
    summary: cutoff 距离过滤（molrs 邻居表）+ xyz 可选
    type: code
    pass_when: |
      成键原子对部分在 cutoff 内/外：apply 后只有 ≤cutoff 成键（走 molrs.NeighborList，非手搓 O(N²)）；
      无坐标 + 不给 cutoff：拓扑全组合确定性配对；给 cutoff 但无坐标 → 清晰 ValueError。
    status: verified
    last_checked: 2026-07-05

  - id: ac-004
    summary: 穷尽 100%，无 RNG、无 conversion
    type: code
    pass_when: |
      2 组分各 N 个 1:1 位点全在 cutoff 内：apply 反应到无可配对，消耗 = 理论最大（每位点一次）；
      DeterministicCrosslinker 无 conversion/seed 参数。
    status: verified
    last_checked: 2026-07-05

  - id: ac-005
    summary: spacing 均匀交联点（纯拓扑，用 molrs topo_distances）
    type: code
    pass_when: |
      线性链每单体一位点、共 M，spacing=K：交联点沿主链等间隔（第 0,K,2K,…，按 topo_distances 排序），
      数 ≈ M/K；不同 K 线性改变密度；确定可复现；不读坐标（无坐标图也可选点）。
    status: verified
    last_checked: 2026-07-05

  - id: ac-006
    summary: 确定性复现 + 显式 pairs + exclude
    type: code
    pass_when: |
      同输入两次 apply 产物逐键一致（无 Math.random/无 seed）；
      pairs=[(0,2),(1,0)] 只成这两配对的键；exclude_same_molecule 无同分子对；
      A×A + exclude_same_match 无 occ_i==occ_j。
    status: verified
    last_checked: 2026-07-05

  - id: ac-007
    summary: 质量闸：ruff/ty/pytest 全绿
    type: runtime
    pass_when: |
      `ruff format --check src tests`、`ruff check src tests`、`ty check src/molpy/`、
      `pytest tests/ -m "not external" -v` 全部 exit 0。
    status: verified
    last_checked: 2026-07-05
---

# Acceptance criteria

- **ac-001**: 顶层 immutable 契约（copy-once、返回同子类、入参不动，对齐 `VirtualSiteBuilder`）。
- **ac-002**: **只编排不造引擎**——匹配/SMIRKS 应用/图编辑/距离全走 molrs，molpy 侧无 SMARTS 引擎、无 MatchSet/Block。
- **ac-003 / ac-004 / ac-005**: 距离配对（molrs 邻居表，xyz 可选）+ 穷尽 100% + spacing 均匀交联点。
- **ac-006**: 确定性可复现 + 显式 pairs + 排除规则。
- **ac-007**: molpy 质量闸。
