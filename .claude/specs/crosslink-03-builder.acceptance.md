---
slug: crosslink-03-builder
criteria:
  - id: ac-001
    summary: PortMatcher 把 atom["port"] 读成 occurrence（与 molrs 匹配同形）
    type: code
    pass_when: |
      BigSMILES 建带 `<`/`>` port 的结构，PortMatcher().find_matches_mapped(g, ">") 返回 list[dict[int,int]]，
      命中所有 atom["port"]==">" 的原子、{1:handle} 解引用正确；结构与 molrs.SmartsPattern.find_matches_mapped 一致。port_key 可配。
    status: verified
    last_checked: 2026-07-05

  - id: ac-002
    summary: PortMatcher occurrence 可作为 Crosslinker 选点前端
    type: code
    pass_when: |
      用 PortMatcher 产出的 occurrence 喂给 Crosslinker（注入/覆写取 occurrence 步）正常交联，返回新图；
      与 molrs SMARTS 前端走同一 molrs.Reaction.apply 编辑路径。
    status: verified
    last_checked: 2026-07-05

  - id: ac-003
    summary: PEO 均匀网络 recipe 端到端可运行
    type: code
    pass_when: |
      线性 PEG × N：DeterministicCrosslinker(rxn, spacing=K, cutoff=r).apply(peg) 产网，交联键数符合 spacing 预期（±容差）；
      产物可 minimize、可 io.write LAMMPS data（不报错）。
    status: verified
    last_checked: 2026-07-05

  - id: ac-004
    summary: PEO 随机网络 recipe 端到端可运行
    type: code
    pass_when: |
      RandomCrosslinker(rxn, conversion=0.7, seed=42, cutoff=r).apply(peg)：转化率 ≈ 0.7（±1 步）、同 seed 复现；
      产物 minimize 收敛、可写出。
    status: verified
    last_checked: 2026-07-05

  - id: ac-005
    summary: 未改 PolymerBuilder，builder 测试零回归
    type: runtime
    pass_when: |
      本 spec 不修改 src/molpy/builder/polymer/（git 确认 core.py/connectors.py/presets.py 未改）；
      pytest tests/test_builder/ -m "not external" 引入前后结果完全一致（全绿）。
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

- **ac-001 / ac-002**: `PortMatcher` 把建模标好的 port 收敛成与 molrs 匹配同形的 occurrence，可作为
  `Crosslinker` 选点前端——"标好的位点"与"SMARTS 找的位点"合流到同一 molrs 编辑引擎。
- **ac-003 / ac-004**: 端到端 PEO 凝胶配方（均匀 `spacing` / 随机 `conversion`）可运行、可松弛、可导出。
- **ac-005**: 收敛只做分析 + 桥接，**不动能跑的 `PolymerBuilder`**，builder 测试零回归。
- **ac-006**: molpy 质量闸。
