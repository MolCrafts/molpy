---
slug: builder-reacter-04-api-docs
criteria:
  - id: ac-001
    summary: docs/api/builder.md 与 reacter.md 的 python 代码块可执行
    type: runtime
    pass_when: |
      pytest tests/test_docs/test_doc_examples.py -v 通过,且该文件确实
      逐块 exec docs/api/builder.md 与 docs/api/reacter.md 的 python 代码块
      (不允许整文件跳过)。
    status: verified
    last_checked: 2026-06-10
  - id: ac-002
    summary: Tool 类迁至 tools.py 且不在任何用户面 __all__
    type: code
    pass_when: |
      python -c "import molpy.builder as b, molpy.builder.polymer as p,
      molpy.builder.polymer.tools as t;
      names={'PrepareMonomer','BuildPolymer','PlanSystem','BuildSystem','BuildPolymerAmber'};
      assert names.isdisjoint(b.__all__) and names.isdisjoint(p.__all__);
      assert all(hasattr(t, n) for n in names)" 退出码 0。
    status: verified
    last_checked: 2026-06-10
  - id: ac-003
    summary: 内部件降级、ReactionPresets 升级为公开扩展点
    type: code
    pass_when: |
      python -c "import molpy.builder.polymer as p;
      assert {'GBigSmilesCompiler','SystemPlanner','PolydisperseChainGenerator'}.isdisjoint(p.__all__);
      from molpy.builder.polymer import ReactionPresets, ReactionPresetSpec" 退出码 0,
      且 docs/api/builder.md 文档化 ReactionPresets.register() 为自定义化学扩展入口。
    status: verified
    last_checked: 2026-06-10
  - id: ac-004
    summary: mp.builder/reacter/pack/compute 顶层属性可用
    type: runtime
    pass_when: |
      python -c "import molpy as mp;
      assert all(hasattr(mp, n) for n in ('builder','reacter','pack','compute'))" 退出码 0。
    status: verified
    last_checked: 2026-06-10
  - id: ac-005
    summary: anchor_selector_* 全库统一,site_selector_* 清零
    type: code
    pass_when: |
      rg -n 'site_selector_' src/ tests/ 无匹配;
      python -c "from molpy.builder.polymer import ReactionPresetSpec;
      import dataclasses; f=[x.name for x in dataclasses.fields(ReactionPresetSpec)];
      assert 'anchor_selector_left' in f and 'anchor_selector_right' in f" 退出码 0。
    status: verified
    last_checked: 2026-06-10
  - id: ac-006
    summary: find_port 为唯一导出名,find_port_atom 别名移除
    type: code
    pass_when: |
      python -c "import molpy.reacter as r; assert 'find_port' in r.__all__;
      assert 'find_port_atom' not in r.__all__ and not hasattr(r, 'find_port_atom')"
      退出码 0(find_port_atom_by_node 不受影响)。
    status: verified
    last_checked: 2026-06-10
  - id: ac-007
    summary: reacter/base.py 与 dsl.py 的 docstring 示例 doctest 通过
    type: runtime
    pass_when: |
      pytest --doctest-modules src/molpy/reacter/base.py src/molpy/builder/polymer/dsl.py -v
      通过(或 tests/test_docs/test_doc_examples.py 中等价的定向 doctest 用例通过),
      且 base.py 示例不再引用 select_port_atom / port_selector_left / run(..., port_L=...)。
    status: verified
    last_checked: 2026-06-10
  - id: ac-008
    summary: 两个 examples 脚本在本地 CI 测试中跑通
    type: runtime
    pass_when: |
      pytest tests/test_docs/test_doc_examples.py -k example -v 通过,执行
      examples/polymer_build.py 与 examples/reacter_bond_react_templates.py 的 main()
      (标记非 external;RDKit 缺失时 skip)。
    status: verified
    last_checked: 2026-06-10
  - id: ac-009
    summary: 用户面文档与 docstring 零 agent 基础设施泄漏
    type: code
    pass_when: |
      rg -n 'ToolRegistry|molpy\.builder\._tool' docs/ src/molpy/builder/polymer/dsl.py
      src/molpy/builder/__init__.py src/molpy/builder/polymer/__init__.py 无匹配。
    status: verified
    last_checked: 2026-06-10
  - id: ac-010
    summary: docs/api/reacter.md 与 04_crosslinking.ipynb 引用真实符号
    type: code
    pass_when: |
      rg -n 'TemplateReacter|select_nothing|product_info' docs/api/reacter.md 无匹配;
      rg -n 'def select_hydroxyl_group' docs/user-guide/04_crosslinking.ipynb 无匹配
      且该 notebook 含 'from molpy.reacter import' 引入 select_hydroxyl_group。
    status: verified
    last_checked: 2026-06-10
  - id: ac-011
    summary: 公共表面 docstring 齐备且 CHANGELOG 记录改名
    type: runtime
    pass_when: |
      python -c "from molpy.builder.polymer import polymer, polymer_system, PolymerBuilder,
      Connector, ReactionPresets; from molpy.reacter import Reacter, BondReactReacter,
      BondReactTemplate, find_port;
      objs=[polymer,polymer_system,PolymerBuilder,Connector,ReactionPresets,Reacter,
      BondReactReacter,BondReactTemplate,find_port];
      assert all(o.__doc__ and ('Args:' in o.__doc__ or 'Attributes:' in o.__doc__) for o in objs)"
      退出码 0;CHANGELOG.md 存在并提及 site_selector→anchor_selector 与
      find_port_atom→find_port 改名。
    status: verified
    last_checked: 2026-06-10
  - id: ac-012
    summary: 全量检查与测试套件通过
    type: runtime
    pass_when: |
      ruff format --check src tests && ruff check src tests && ty check src/molpy/ &&
      pytest tests/ -m "not external" -v 全部退出码 0。
    status: verified
    last_checked: 2026-06-10
---

# Acceptance criteria

ac-001/007/008 共同构成"文档示例永不腐烂"防线：API 文档代码块、类 docstring 示例、examples 脚本三类面向用户的代码都进入测试套件。ac-002/003/004 锁定入口收敛后的导出表面；ac-005/006 锁定术语统一（无弃用垫片，行为零变化由现有测试套件经 ac-012 兜底）；ac-009/010 是记忆规则（agent 基础设施不入用户文档）与已验证崩溃点的逐项消除；ac-011 覆盖 F 项 docstring 审计与 CHANGELOG 记录义务。
