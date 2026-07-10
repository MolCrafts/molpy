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
    status: pending


  - id: ac-002
    summary: LAMMPS bond_react 导出的 golden 逐字节不变
    type: code
    pass_when: |
      LammpsBondReactWriter 消费 RegionPatch;
      tests/test_io/test_data/test_lammps_bond_react.py 的 golden 文件逐字节相同。
      io/data/lammps_bond_react.py 中 '^def ' 计数为 0(三个自由函数并入 writer 类)。
    status: pending


  - id: ac-003
    summary: 硬编码化学与硬编码字段清零(铁律 1)
    type: code
    pass_when: |
      grep -rnE '== "(H|C|O|N)"|element="(H|C|O)"' src/molpy/builder/ src/molpy/typifier/ → 零命中
      grep -rnE '\bdehydration\b|\bcondensation\b|\bhydroxyl\b' src/molpy/ → 零命中
      grep -rn 'port_role|ports_compatible|_cleanup_stale_ports|_get_port_direction' src/molpy/ → 零命中
      grep -rnE '"head"|"tail"|"port"|site_field' src/molpy/ → 零命中
    status: pending


  - id: ac-004
    summary: 测试分流被逐类辩护,覆盖率不退
    type: code
    pass_when: |
      tests/test_reacter/ 不存在;有效行为断言已出现在 tests/test_builder/test_assembler.py。
      commit body 列出被删除的测试类别与"它断言的是实现细节而非行为"的理由。
      pytest --cov=src/molpy tests/ -m "not external" 的 line coverage 不低于删除前。
    status: pending


  - id: ac-005
    summary: 无回归
    type: code
    pass_when: |
      pytest tests/ -m "not external" 全绿;ruff format --check、ruff check、ty check src/molpy/ 全绿。
    status: pending


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


  - id: ac-007
    summary: breaking change 被显式记账
    type: code
    pass_when: |
      docs/changelog.md 在 BREAKING 小节列出 molpy.reacter 子包移除、
      PolymerBuilder 构造签名变更(connector=/reacter= → reaction=)、
      molpy.core.region_radius 移除;版本号已 bump。
    status: pending

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
    status: pending


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
    status: pending
