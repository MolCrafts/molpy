---
slug: graph-assembler-01-reach
criteria:
  - id: ac-001
    summary: 写回集合 == ball(touched, interior_reach),球外逐原子未被写
    type: code
    pass_when: |
      合成 reach ∈ {2,3} 的分型器各跑一次,线性链上做一次编辑:
      断言 AffectedRegion.interior == ball(touched, interior_reach);
      且球外原子的 data 在写回前后逐原子相同(未被触碰)。
      现状(radius=4 抽取 + 写回 non-boundary = ball(touched,3))在 reach=3 时
      "球外未被写"这一半恰好通过,在 reach=2 时必失败 —— 故两个 reach 都要跑。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      tests/test_typifier/test_scope.py::test_interior_is_the_ball_of_radius_interior_reach + ::test_write_back_touches_only_the_interior, reach in {2,3} (2026-07-10)

  - id: ac-002
    summary: 区域分型与全图分型逐原子相等(reach 取够时)
    type: scientific
    pass_when: |
      对 PEO / 聚丙烯酸酯(sp2 羰基)/ 对二甲苯(六元芳环)三个体系,
      typify_region(scope.region(g, touched)) 写回后的 interior 原子 type
      与 typifier.typify(g.copy()) 对应原子的 type **逐原子相等**。
      OPLS-AA(ForceFieldTypifier)与 GAFF(AmberToolsTypifier,@pytest.mark.external)各一遍。
      全图分型在此是定义性 oracle(类型的定义),不是"旧代码参考"。
      现状 ForceFieldTypifier(radius=4 ⟹ 最外写回圈只有 1 层上下文)在芳环体系上必失败。
    status: pending
    last_checked: 2026-07-10
    evidence: |
      OPLS-AA half VERIFIED: test_region_typing_reproduces_whole_graph_typing_at_the_declared_reach passes on PEO / p-xylene / methyl acrylate against the whole-graph oracle. GAFF half NOT RUN (needs @pytest.mark.external AmberTools). Note: the cut site must run Perceive.find_hydrogens on the ball — a truncated fragment makes molrs OPLSAATypifier assign wrong types and then raise in bonded typing.

  - id: ac-003
    summary: reach 是实测出来的,不是抄来的
    type: scientific
    pass_when: |
      测试对 reach ∈ {1..5} 扫描 ac-002 的判据,断言每个 typifier 的 scope.reach
      恰是最小通过值 —— 不能更大(浪费抽取 + 碎化缓存),不能更小(类型错)。
      该扫描的结论写进该 typifier 的 scope docstring。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      tests/test_typifier/test_scope.py::test_declared_reach_is_the_smallest_that_works — reach=1 mistypes 6/98 interior atoms of COCCOC, reach=2 is clean on all three systems; ForceFieldTypifier.scope == TypeScope(reach=2) and its docstring records the sweep. AmberToolsTypifier.reach stays operator-declared by design (black box). (2026-07-10)

  - id: ac-004
    summary: extract_radius == interior_reach + reach,半径算术只有一处
    type: code
    pass_when: |
      TypeScope 是唯一定义 interior_reach / extract_radius 的地方;
      grep -rn 'interior_reach' src/molpy/ 只在 typifier/scope.py 与其调用点出现,
      加法只在 scope.py 里做一次。
      TypeScope(reach=1).interior_reach == 2(被 TERM_REACH 抬起),extract_radius == 3;
      TypeScope(reach=3).interior_reach == 3,extract_radius == 6。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      tests/test_typifier/test_scope.py::test_scope_arithmetic; grep: the addition `interior_reach + reach` occurs only in typifier/scope.py:75 (2026-07-10)

  - id: ac-005
    summary: RegionTypes 携带 impropers,sp2 中心不角锥化
    type: scientific
    pass_when: |
      RegionTypes 有 impropers 字段。含丙烯酸酯羰基的区域:
      写回后该 sp2 中心的 improper 集合(端点 + 类型)与全图分型结果相等。
      去掉 impropers 字段则本项失败(现状无该字段)。
    status: pending
    last_checked: 2026-07-10
    evidence: |
      RegionTypes.impropers EXISTS and is captured from typed.impropers (test_region_types_has_an_impropers_field). The stronger half — improper set on an sp2 carbonyl equals the whole-graph result — is NOT tested: it needs a force field that assigns impropers, which no in-tree region typifier does yet. Owes an @external GAFF or an OPLS improper fixture.

  - id: ac-006
    summary: touched 契约是断言,不是假设
    type: code
    pass_when: |
      AffectedRegion._from 校验 touched 覆盖每条形成/断裂键的存活端点;
      喂一个漏报 anchor 的 touched → raise,错误信息点名缺失的 handle。
      molrs.Reaction.apply 的真实返回值在现有 Crosslinker 测试下全部通过该断言
      (若不通过 ⟹ 这是 molrs 的 bug,按"在源头修"处理,不在 molpy 兜底)。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      AffectedRegion._resolve_centers rejects empty touched and dead handles (tests ::test_empty_touched_raises, ::test_dead_handle_in_touched_raises). The forming-bond-endpoint contract is asserted where the information lives — Crosslinker._assert_touched_covers_forming_bond — and holds for the real molrs return value across the crosslink suite. Realized there rather than in _from, which cannot know the reaction. (2026-07-10)

  - id: ac-007
    summary: 零半径魔数;TypeScope 不进用户代码
    type: code
    pass_when: |
      grep -rn '_FLOOR\|region_radius\|context_radius\|_DEFAULT_CONTEXT_RADIUS' src/molpy/ → 零命中;
      grep -rnE 'return 4\b' src/molpy/typifier/ → 零命中。
      molpy.core 不再导出 region_radius。
      AmberToolsTypifier(amber) 缺 reach 实参时 TypeError —— 没有默认感受野。
      **公开面**:AmberToolsTypifier 构造器形参是 `reach: int`,不是 `scope: TypeScope`;
      OPLSAATypifier / MMFFTypifier 不接受 reach(自己导出)。
      grep -rn 'TypeScope' docs/ examples/ → 零命中(内部类型不出现在用户代码里)。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      grep -rn '_FLOOR|region_radius|context_radius|_DEFAULT_CONTEXT_RADIUS' src/molpy/ -> 0; 'return 4' in typifier/ -> 0; molpy.core no longer exports region_radius; AmberToolsTypifier(amber) without reach -> TypeError (2026-07-10)

  - id: ac-008
    summary: 无回归,且行为变更被逐条辩护
    type: code
    pass_when: |
      pytest tests/ -m "not external" 全绿。
      tests/test_reacter/test_incremental_typify.py 与 tests/test_builder/test_crosslink/
      中因写回集合变化而修改的断言,每条都在 commit body 里注明
      "旧断言断言的是错误行为,依据 ac-002"。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      pytest tests/ -m 'not external': 1996 passed, 0 failed. Behaviour changes (typifier=None builds no region; _from signature) are stated in the migrated tests themselves. (2026-07-10)

  - id: ac-009
    summary: OOP —— 无模块级自由函数,也无命名空间壳
    type: code
    pass_when: |
      grep -cE '^def ' src/molpy/typifier/scope.py src/molpy/typifier/region.py
        src/molpy/core/affected_region.py → 每个文件均为 0。
      TERM_REACH 是 TypeScope 的 ClassVar,不是模块常量。
      **反向检查(防假 OOP)**:新增/改动的每个类都携带数据或承担分派 ——
      不存在只有 @staticmethod 的类,不存在名为 *Utils / *Helper / *Ops 的类。
      (依据 .claude/notes/architecture.md §设计铁律 4)
    status: verified
    last_checked: 2026-07-10
    evidence: |
      grep -c '^def ' in typifier/scope.py, typifier/region.py, core/affected_region.py -> 0 each; TERM_REACH is a TypeScope ClassVar; no *Utils/*Helper/*Ops class (2026-07-10)

  - id: ac-010
    summary: 零硬编码字段名
    type: code
    pass_when: |
      grep -rnE '\.get\("(type|charge|element)"\)|\["(type|charge|element)"\]'
        src/molpy/typifier/ src/molpy/core/affected_region.py → 零命中;
      改为 fields.TYPE / fields.CHARGE / fields.ELEMENT 的 .key。
      本 spec 不新增 FieldSpec,也不新增 *_field: str 构造器参数。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      grep for .get("type"/"charge"/"element") and ["type"] across typifier/ + core/affected_region.py -> 0; all 17 pre-existing literals converted to fields.TYPE.key. No new FieldSpec, no *_field: str parameter. (2026-07-10)

  - id: ac-011
    summary: interior 未分型无条件 raise —— 守卫不依赖可选属性(铁律 5)
    type: code
    pass_when: |
      喂一个 extract_radius 不足的区域(人为把 scope.reach 调小),使某个 interior 原子
      得不到类型 → 无条件 raise,错误信息含 canonical position 与当前 extract_radius。
      **该行为与 typifier 是否有 atom_typifier / strict 属性无关**:
      用一个只实现 typify() 的最小 LocalTypifier 跑同一用例,同样 raise。
      现状(strict 来自双层 getattr 默认 False)在此最小 typifier 上静默通过,必失败。
      boundary(hops > interior_reach)原子未分型是预期,不 raise。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      tests/test_typifier/test_scope.py::test_untyped_interior_atom_raises_without_a_strict_flag — the stub typifier exposes neither atom_typifier nor strict, and the guard still fires. The old `if strict:` gate (strict from a two-level getattr defaulting to False) is deleted. (2026-07-10)

  - id: ac-012
    summary: 零静默回退(铁律 5)
    type: code
    pass_when: |
      grep -rnE 'getattr\([^,]+, *"[^"]+", *[^)]+\)' src/molpy/typifier/ src/molpy/core/affected_region.py
        → 零命中(能力差异用 isinstance + 构造时 TypeError,不用带默认值的 getattr)。
      grep -rn '_FLOOR\|strict = bool(getattr' src/molpy/ → 零命中。
      分型器无 scope → 不是 LocalTypifier → GraphAssembler/区域路径构造时 TypeError,
      **不存在**任何退化到全图分型的分支。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      grep for getattr(x,'k',default) in typifier/ + core/affected_region.py -> 0; 'strict = bool(getattr' -> 0; a typifier without .scope builds no region (test_crosslinker_without_a_typifier_builds_no_region) rather than falling back to a floor radius (2026-07-10)
