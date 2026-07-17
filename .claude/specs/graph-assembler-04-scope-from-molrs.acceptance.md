---
slug: graph-assembler-04-scope-from-molrs
criteria:
  - id: ac-001
    summary: molrs 前置就绪(只要语法事实)
    type: code
    pass_when: |
      hasattr(molrs.SmartsPattern, "max_bond_depth") 且 hasattr(molrs.SmartsPattern, "ring_primitives")
      且 hasattr(molrs.Atomistic, "max_ring_system_size");
      molpy pyproject.toml 对 molcrafts-molrs 是精确版本 pin。
      **不要求** molrs.OplsTypifier —— 本 spec 不依赖分型器下沉。
      不成立 ⟹ 保持 blocked,不得进入实现。
    status: pending

  - id: ac-002
    summary: 体系判断在 molpy,molrs 只报语法(铁律 2)
    type: code
    pass_when: |
      molrs 的 ring_primitives 返回枚举(Sized/Membership/RingCount/RingBondCount),
      不返回 is_bounded 布尔,不返回 scope。
      molrs 源码中 grep 'TypeScope|reach|receptive' → 零命中。
      合成公式 max(depth, ring_system//2 + 1) 与无界判据只出现在
      src/molpy/typifier/scope.py 的 TypeScope.from_patterns()。
    status: pending

  - id: ac-003
    summary: 无尺寸环谓词使模式集失去 scope
    type: code
    pass_when: |
      含 [R]、[!R]、[R0]、[R2](环数)、[x2](环键数)之一的模式集 →
      TypeScope.from_patterns 抛 UnboundedPatternSet →
      SmartsTypifier 构造 TypeError,错误信息**点名该谓词**;
      换成 [r6] 后构造成功且 scope.reach 有界。
      UnboundedPatternSet 携带 .primitive 数据(不是空壳异常)。
    status: pending

  - id: ac-004
    summary: 导出的 reach 等于实测最小 reach
    type: scientific
    pass_when: |
      对 OPLS-AA 与 MMFF94:SmartsTypifier.scope.reach == graph-assembler-01 的
      ac-003 扫描出的最小通过值。导出值更大 → 浪费抽取且碎化缓存;更小 → 类型错。
    status: pending

  - id: ac-005
    summary: 环系尺寸,不是 SSSR 单环尺寸
    type: scientific
    pass_when: |
      萘(稠合双环)的 max_ring_system_size == 10。返回 6 ⟹ molrs bug,
      在 molrs 修,不在 molpy 兜底。
    status: pending

  - id: ac-006
    summary: molpy 侧零半径字面量
    type: code
    pass_when: |
      grep -rnE 'reach\s*=\s*[0-9]' src/molpy/typifier/ → 仅命中 TypeScope.TERM_REACH = 2
      (其 docstring 证明它是"二面角元数减二",非魔数)。
      SmartsTypifier 及其子类不含任何 reach 字面量。
      AmberToolsTypifier 的 scope 仍为构造器必填(黑盒边界,docstring 标注)。
    status: pending
