---
slug: incremental-typify-02-cache
criteria:
  - id: ac-001
    summary: typify_region 的 interior 类型 == 整图分型
    type: code
    pass_when: |
      对小体系，typify_region(region) 给 interior 原子的类型，与整图 typify 给这些原子的类型一致
      （boundary 壳层提供完整环境）。RegionTypes 按 canonical order、无 live Entity 引用。
    status: verified
    last_checked: 2026-07-06
  - id: ac-002
    summary: 缓存命中——相同 junction 只调用底层 typifier 一次
    type: code
    pass_when: |
      对两份相同 junction 调 RetypeCache.retype：底层 typifier 只被调用一次（计数器/spy 验证），
      第二次为哈希命中 + is_isomorphic 确认；write-back 经 entity_map + canonical order。
    status: verified
    last_checked: 2026-07-06
  - id: ac-003
    summary: 增长成本 O(#distinct junction) 而非 O(N)
    type: code
    pass_when: |
      构造 N-junction 规则链，随 N 增大，typifier 调用次数有界（≈distinct junction 数），
      不随 N 线性增长——O(N²) retype 消除。
    status: verified
    last_checked: 2026-07-06
  - id: ac-004
    summary: 与旧路径结果一致 + fallback
    type: code
    pass_when: |
      region+cache 路径分型的产物类型 == 旧 _incremental_typify 整图路径；
      region 为 None 时回退旧路径，tests/test_reacter/ 无回归。
    status: verified
    last_checked: 2026-07-06
  - id: ac-005
    summary: 质量闸：ruff/ty/pytest 全绿
    type: runtime
    pass_when: |
      ruff format --check / ruff check / ty check / pytest -m "not external" 全 exit 0。
    status: verified
    last_checked: 2026-07-06
---

# Acceptance criteria

- **ac-001**: 区域分型正确（interior 类型等于整图，靠 boundary 壳层）。
- **ac-002 / ac-003**: 按结构哈希去重——相同 junction 只分型一次，增长成本 O(#distinct) 而非 O(N²)。
- **ac-004**: 与旧路径等价 + region 缺失时回退无回归。
- **ac-005**: molpy 质量闸。
