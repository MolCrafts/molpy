---
slug: incremental-typify-01-region
criteria:
  - id: ac-001
    summary: AffectedRegion 从 touched 原子按半径抽取（interior/boundary/entity_map）
    type: code
    pass_when: |
      对已知编辑，AffectedRegion._from(parent, touched, radius) 是 Atomistic 子类实例；
      interior=变化原子、boundary=有外部邻居的壳层原子、entity_map 把区域原子映回父图原子（往返正确）；
      radius=region_radius(typifier)=max(context_radius,4)。
    status: pending
  - id: ac-002
    summary: 结构 __hash__/__eq__（相同 junction 相等）
    type: code
    pass_when: |
      两份相同局部环境的区域 hash 相等且 ==（经 molrs structural_hash/is_isomorphic）；
      不同结构的区域 hash 一般不等且 !=。成员 Entity/Link 仍身份哈希（未破坏核心契约）。
    status: pending
  - id: ac-003
    summary: 是 MolGraph，可喂 AmberTools
    type: code
    pass_when: |
      isinstance(region, Atomistic) 为真；AmberTools 的 Atomistic→PDB 桥接受该区域（不改 wrapper）。
    status: pending
  - id: ac-004
    summary: 生产者构建区域
    type: code
    pass_when: |
      Crosslinker 用 molrs.Reaction.apply 返回的 touched handle 构建 AffectedRegion；
      Reacter.run 的 ReactionResult.region 被设置（取代 modified_atoms）。
    status: pending
  - id: ac-005
    summary: 质量闸：ruff/ty/pytest 全绿 + reacter 无回归
    type: runtime
    pass_when: |
      ruff format --check / ruff check / ty check / pytest -m "not external" 全 exit 0；
      pytest tests/test_reacter/ 无新增失败。
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002**: 区域抽取正确 + 结构哈希去重键（相同 junction 相等），且不破坏 Entity 身份哈希。
- **ac-003**: 区域即 MolGraph → 直接喂 AmberTools。
- **ac-004**: reacter/crosslink 作为生产者返回区域，取代扁平 modified_atoms。
- **ac-005**: molpy 质量闸 + reacter 无回归。
