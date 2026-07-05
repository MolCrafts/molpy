---
slug: crosslink-02-random
criteria:
  - id: ac-001
    summary: 转化率控制精确
    type: code
    pass_when: |
      RandomCrosslinker(rxn, conversion=0.5, seed=1).apply(g)：已反应位点 / 限制反应物总位点 ≈ 0.5（±1 步）。
    status: verified
    last_checked: 2026-07-05

  - id: ac-002
    summary: seed 可复现
    type: code
    pass_when: |
      同 seed 两次 apply 产物逐键一致；存在一对不同 seed 使产物不同。
    status: verified
    last_checked: 2026-07-05

  - id: ac-003
    summary: conversion=1.0 无候选时提前停，不死循环
    type: code
    pass_when: |
      conversion=1.0 且有无法再配对的剩余位点：apply 候选耗尽即终止，返回已达最大转化网络，不无限循环。
    status: verified
    last_checked: 2026-07-05

  - id: ac-004
    summary: cutoff / exclude / max_per_molecule
    type: code
    pass_when: |
      cutoff：只在 cutoff 内随机配对（继承基类 molrs 邻居表）；
      exclude_same_molecule 无同分子对；A×A + exclude_same_match 无 occ_i==occ_j；
      max_per_molecule=2 任一分子被消耗位点 ≤ 2。
    status: verified
    last_checked: 2026-07-05

  - id: ac-005
    summary: 只重写 select，匹配/距离/改图全继承基类（→ molrs）
    type: code
    pass_when: |
      RandomCrosslinker 仅实现 select；候选/距离/binding/改图（molrs.Reaction.apply）均继承 01 基类；
      grep src/molpy/builder/crosslink/_random.py 无自造距离双重循环、无自造键/拓扑生成、无 SMARTS 引擎、无 classmethod。
    status: verified
    last_checked: 2026-07-05

  - id: ac-006
    summary: 质量闸：ruff/ty/pytest 全绿
    type: runtime
    pass_when: |
      `ruff format --check src tests`、`ruff check src tests`、`ty check src/molpy/`、
      `pytest tests/ -m "not external" -v` 全部 exit 0。
    status: verified
    last_checked: 2026-07-05
---

# Acceptance criteria

- **ac-001 / ac-002 / ac-003**: 随机成网核心——转化率精度、seed 复现、`conversion=1.0` 不死循环。
- **ac-004**: 距离 + 排除 + 每分子上限。
- **ac-005**: 只重写 `select`，其余全继承基类（→ molrs），不重造。
- **ac-006**: molpy 质量闸。
