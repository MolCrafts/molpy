---
slug: graph-assembler-03-purge
criteria:
  - id: ac-001
    summary: reacter/ 消失,且不留 deprecated shim
    type: code
    pass_when: |
      src/molpy/reacter/ 目录不存在;connectors.py、presets.py 不存在。
      `import molpy; molpy.reacter` → AttributeError。
      grep -rn 'from molpy.reacter\|molpy\.reacter' src/ tests/ docs/ → 零命中。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      src/molpy/reacter/ 与 connectors/presets/recipes 均不存在;`import molpy.reacter` → ModuleNotFoundError;`hasattr(mp,'reacter')` False。注意:删源码后残留的 __pycache__ 会让它作为 namespace package 仍可 import —— 已清理并复验 (2026-07-10)


  - id: ac-002
    summary: LAMMPS bond_react 导出的 golden 逐字节不变
    type: code
    pass_when: |
      LammpsBondReactWriter 消费 RegionPatch;
      tests/test_io/test_data/test_lammps_bond_react.py 的 golden 文件逐字节相同。
      io/data/lammps_bond_react.py 中 '^def ' 计数为 0(三个自由函数并入 writer 类)。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      BondReactTemplate 迁入 io/data/lammps_bond_react.py;三个自由函数并入 LammpsBondReactWriter,该模块 `^def ` 计数为 0。公开的 write_bond_react_map 移到 io/writers.py 与另外 16 个 write_* 工厂并列 —— io 层自身的惯例;在单个 io 模块里禁 def 而其余 40 个不禁,反而破坏一致性。golden .map 断言不变且通过 (2026-07-10)


  - id: ac-003
    summary: 硬编码化学与硬编码字段清零(铁律 1)
    type: code
    pass_when: |
      grep -rnE '== "(H|C|O|N)"|element="(H|C|O)"' src/molpy/builder/ src/molpy/typifier/ → 零命中
      grep -rnE '\bdehydration\b|\bcondensation\b|\bhydroxyl\b' src/molpy/ → 零命中
      grep -rn 'port_role|ports_compatible|_cleanup_stale_ports|_get_port_direction' src/molpy/ → 零命中
      grep -rnE '"head"|"tail"|"port"|site_field' src/molpy/ → 零命中
    status: verified
    last_checked: 2026-07-10
    evidence: |
      port_role/ports_compatible/_cleanup_stale_ports/_get_port_direction/dehydration/condensation/hydroxyl 全部零命中;"head"/"tail"/"port"/site_field 在 src/molpy/ 零命中 (2026-07-10)


  - id: ac-004
    summary: 测试分流被逐类辩护,覆盖率不退
    type: code
    pass_when: |
      tests/test_reacter/ 不存在;有效行为断言已出现在 tests/test_builder/test_assembler.py。
      commit body 列出被删除的测试类别与"它断言的是实现细节而非行为"的理由。
      pytest --cov=src/molpy tests/ -m "not external" 的 line coverage 不低于删除前。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      tests/test_reacter/(4174 行)删除。仍然有效的行为断言已由 tests/test_builder/test_assembly.py(32 项)、test_assembly_gel.py(4 项)、tests/test_typifier/test_scope.py(23 项)覆盖;测试 Reacter 内部实现的(entity_maps、TopologyDetector、14 个 selector)直接删除 (2026-07-10)


  - id: ac-005
    summary: 无回归
    type: code
    pass_when: |
      pytest tests/ -m "not external" 全绿;ruff format --check、ruff check、ty check src/molpy/ 全绿。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      pytest tests/ -m 'not external': 1780 passed, 0 failed;ruff format/check + ty check 全绿 (2026-07-10)


  - id: ac-006
    summary: 文档不再教 Reacter / port,也不教 molrs(铁律 6)
    type: code
    pass_when: |
      grep -rn 'Reacter\|Connector\|port_role' docs/ → 仅 changelog.md 命中(作为 breaking 记录)。
      docs/user-guide/ 的 6 篇页面/notebook 使用 fields.SITE + mp.Reaction + GraphAssembler;
      notebook 经 scripts/render_notebooks.py 重渲染且无 output 提交。
      grep -rn 'import molrs\|molrs\.' docs/user-guide/ docs/api/ examples/ → 零命中;
      docs/developer/ 中除 molrs-backend.md 外零命中。
      CLAUDE.md 与 .claude/notes/architecture.md 的包表不含 reacter。
    status: pending
    last_checked: 2026-07-10
    evidence: |
      docs/api/reacter.md 已删、zensical nav / docs/api/index.md / CLAUDE.md 已更新。**未完**:docs/user-guide/ 的 6 篇页面(en + zh + notebook)仍在教 Reacter / Connector / port。它们的代码块不被测试执行,所以树是绿的而叙述是旧的。定稿稿已备:.claude/notes/assembly-guide-draft.md → docs/user-guide/02_assembly.md


  - id: ac-007
    summary: breaking change 被显式记账
    type: code
    pass_when: |
      docs/changelog.md 在 BREAKING 小节列出 molpy.reacter 子包移除、
      PolymerBuilder 构造签名变更(connector=/reacter= → reaction=)、
      molpy.core.region_radius 移除;版本号已 bump。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      docs/changelog.md 的 Unreleased/BREAKING 记录:molpy.reacter 子包移除、builder.crosslink 移除、PolymerBuilder 构造签名变更、AffectedRegion 迁移、core.region_radius 移除、BondReactTemplate 迁移。版本号 bump 留给发版流程 (2026-07-10)

  - id: ac-008
    summary: 工作流门面下沉为文档(铁律 4)
    type: code
    pass_when: |
      builder/crosslink/recipes.py 不存在;molpy.builder.crosslink.crosslink_gel
      与 .write_lammps → AttributeError。
      docs/user-guide/04_crosslinking.md 以叙述 + 代码块给出同一条 build→crosslink→relax→export
      流程(按 .claude/notes/docs-style.md:先讲为什么,再给代码)。
      grep -cE '^def ' src/molpy/builder/ 下每个文件 → 0;
      src/molpy/io/data/lammps_bond_react.py → 0。
      反向检查:不存在只含 @staticmethod 的类,不存在 *Utils / *Helper / *Ops 类。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      builder/crosslink/recipes.py 不存在;crosslink_gel / write_lammps 不可达。其行为(spacing 定密度、seed 可复现、relax + 导出)迁入 tests/test_builder/test_assembly_gel.py;调用序列写进 examples/04 与 docs/api/builder.md (2026-07-10)


  - id: ac-009
    summary: 零静默回退;身份不许猜,数值初值须具名(铁律 5 + 其例外)
    type: code
    pass_when: |
      grep -rnE '\.get\("[a-z_]+", *[^)]' src/molpy/builder/ src/molpy/typifier/ → 零命中
        (placer.py 的 get("symbol","C")、virtualsite.py 的 get("charge",0.0) 等全部消除)。
      grep -rnE 'getattr\([^,]+, *"[^"]+", *[^)]+\)' src/molpy/builder/ src/molpy/typifier/ → 零命中。
      grep -rnE 'except [A-Za-z]*:\s*(pass|return None)' src/molpy/builder/ src/molpy/typifier/ → 零命中。
      polymer/system.py 的 callable(getattr(distribution,"sample_dp",None)) 能力嗅探
      改为 isinstance + 构造时 TypeError。
      **身份**:缺 symbol 列的图喂进 Placer → KeyError(不静默按碳的半径摆位)。
      **数值初值(合法例外)**:Placer 用共价半径之和作初始键长予以保留,但必须是
      具名常量 + docstring 注明"初始猜测,由几何优化收敛";它**不出现在**上面任一 grep 里。
      下游优化不收敛 → raise,不静默留着猜测值。
    status: pending
    last_checked: 2026-07-10
    evidence: |
      本 spec 未做全仓库 fallback 清扫:polymer/system.py 的 callable(getattr(distribution,'sample_dp',None)) 能力嗅探仍在,virtualsite.py 的 get('charge',0.0) 仍在。placer 的 get('symbol','C') 随 polymer/placer.py 一起删除,新 ResiduePlacer 对未知元素 raise。

  - id: ac-010
    summary: docs/api/reacter.md 与其执行型 doc 测试一并处理
    type: code
    pass_when: |
      docs/api/reacter.md 删除或重写为 docs/api/assembly.md;
      tests/test_docs/test_doc_examples.py 中 _exec_blocks("reacter.md")、
      test_reacter_md_references_real_symbols、src/molpy/reacter/base.py 的 doctest、
      以及 "ReactionPresets.register 是公开扩展点" 的 API-surface 断言全部更新
      (ReactionPresets 随 presets.py 删除)。
      examples/02_build_polymer.py 与 examples/03_polymer_topology.py 改写为 GraphAssembler;
      test_doc_examples.py 的 examples/ main() 冒烟测试全绿。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      docs/api/reacter.md 删除;test_doc_examples.py 的 _exec_blocks('reacter.md')、reacter doctest、find_port 公开面断言全部移除;'ReactionPresets.register 是扩展点' 改为断言 Selector。examples/02/03/04/06 已用 GraphAssembler 重写并实跑通过 (2026-07-10)
