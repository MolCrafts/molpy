---
slug: incremental-typify-03-integration
criteria:
  - id: ac-001
    summary: polymer builder 全程共享 RetypeCache，junction 去重
    type: code
    pass_when: |
      N-单体规则链增长：底层 FF typifier 调用次数有界（≈distinct junction 数）、不随 N 线性；
      产物类型 == 整链 typify 基线；PolymerBuilder 产物除分型更快/缓存外不变。
    status: verified
    last_checked: 2026-07-06
  - id: ac-002
    summary: crosslink 可选 typifier 局部重分型
    type: code
    pass_when: |
      Crosslinker(rxn, typifier=ff).apply(g)：每个交联点 interior 原子带正确类型；
      不传 typifier 时图未分型（纯拓扑行为不变）。
    status: verified
    last_checked: 2026-07-06
  - id: ac-003
    summary: AmberTools 区域路径 + 哈希缓存
    type: code
    pass_when: |
      （external）AmberTools 对一个 AffectedRegion 经 antechamber 分型；
      重复的相同区域为缓存命中（不再起子进程）；GAFF 类型经 entity_map 写回父图。
      whole-molecule 路径不变。
    status: verified
    last_checked: 2026-07-06
  - id: ac-004
    summary: 质量闸 + builder/crosslink 无回归
    type: runtime
    pass_when: |
      ruff/ty/pytest -m "not external" 全 exit 0；
      pytest tests/test_builder/ -m "not external" 前后全绿。
    status: verified
    last_checked: 2026-07-06
---

# Acceptance criteria

- **ac-001**: 这是 O(N²)→O(#distinct) 的落地——共享缓存让 junction 跨整条链去重。
- **ac-002**: 交联获得可选局部重分型（默认仍纯拓扑）。
- **ac-003**: 第三方 typifier（AmberTools）也消费区域 + 按哈希缓存。
- **ac-004**: molpy 质量闸 + builder/crosslink 无回归。
