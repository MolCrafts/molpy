---
slug: typifier-architecture
criteria:
  - id: ac-001
    summary: 六个死模块消失,且包外从未导入过它们
    type: code
    pass_when: |
      importlib.import_module("molpy.typifier.{base,bond,angle,dihedral,mmff,atomistic}")
      六个全部 ModuleNotFoundError。
      且 grep -rn "typifier\.(base|bond|angle|dihedral|mmff|atomistic)" src tests examples docs
      零命中(docs/zh/developer/attribution.md 的 foyer 版权归属除外 —— 它记的是历史来源,
      不是 import)。
      非空洞性:改动前这六个 import 全部成功,可 import 即证明它们此前确实存在。
    status: pending

  - id: ac-002
    summary: 包内不再有重名符号
    type: code
    pass_when: |
      对 molpy.typifier 的每个子模块做 AST 扫描,收集所有 top-level class/def 名字;
      断言没有任何名字在两个不同子模块里被**定义**两次。
      改动前该断言必失败并报出 TypifierBase / PairTypifier / atomtype_matches 三对。
    status: pending

  - id: ac-003
    summary: ForceFieldTypifier 是真 ABC,由 Python 在构造时拒绝
    type: code
    pass_when: |
      pytest.raises(TypeError, match="abstract") 覆盖两种情形:
      (a) ForceFieldTypifier(ff) 直接构造;
      (b) 一个只 pass 的子类 class _NoAtom(ForceFieldTypifier) 构造。
      而一个实现了 atom_typifier 属性的子类可以正常构造,且 typify() 走通。
      断言 "abstract" 出现在异常信息里 —— 即错误来自 ABCMeta,不是手写守卫。
    status: pending

  - id: ac-004
    summary: 两个 RegionTypifier 实现结构相符但无共同祖先
    type: code
    pass_when: |
      isinstance(ClpTypifier(), RegionTypifier) 且 isinstance(AmberToolsTypifier(...), RegionTypifier);
      not issubclass(AmberToolsTypifier, ForceFieldTypifier) 且反向亦然;
      一个只有 typify_region 没有 scope 的类 not isinstance(..., RegionTypifier)。
      最后一条锁住 GraphAssembler 拒绝无界 typifier 的那条路径。
    status: pending

  - id: ac-005
    summary: atom_typifier 的 strict 在契约里,_typify_relaxed 不再伸手改未声明属性
    type: code
    pass_when: |
      AtomTypifier 声明 strict;_ClpAtomTypifier 是 AtomTypifier 的子类。
      给 ClpTypifier.typify_region 一个区域,断言调用期间 atom_typifier.strict is False、
      返回后恢复为构造时的值(用一个记录 strict 变化的 spy AtomTypifier 观察)。
      异常路径也要恢复(typify 抛错时 finally 生效)。
    status: pending

  - id: ac-006
    summary: 三个 bonded typifier 共享一个算法,元数从 FF_TYPE 读出而非声明
    type: code
    pass_when: |
      issubclass(ForceFieldBondTypifier, ForceFieldBondedTypifier) 三个都成立;
      ForceField{Bond,Angle,Dihedral}Typifier.FF_TYPE 分别是 BondType/AngleType/DihedralType;
      _pattern() 对三种 FF 类型分别返回长度 2/3/4 的 tuple。
      源码层面:bonded.py 里 typify() 只出现一次。
    status: pending

  - id: ac-007
    summary: 打分规则逐字保留(exact > class > wildcard,双向,整条模式一票否决)
    type: code
    pass_when: |
      TypeClassIndex.score:
        exact(("opls_135","opls_140")) > class(("CT","HC")) > wildcard(("*","*")) == 0;
        某一位配不上 → None;
        正反两向同分;
        ("*","CT","CT","*") 必须严格小于 ("opls_135","CT","CT","opls_135") —— OPLS 二面角
        依赖这个序;
        元数不匹配 → ValueError(不再静默 zip 截断)。
      并且 TypeClassIndex 只在 ForceFieldBondedTypifier 构造时扫一次 ForceField
      (用 spy 计 get_types(AtomType) 调用次数;改动前每个 bonded typifier 各扫一次)。
    status: pending

  - id: ac-008
    summary: 全量回归绿,且下游行为逐字不变
    type: runtime
    pass_when: |
      ruff format --check + ruff check + ty check 全绿;
      pytest tests/ -m "not external" 通过数不低于改动前(1785),0 失败;
      tests/test_typifier/ + tests/test_builder/test_assembly.py + tests/test_typifier/test_clp.py 全绿;
      tests/test_io/test_forcefield/test_xml.py 里的 ForceFieldBondTypifier 断言全绿。
    status: pending

  - id: ac-009
    summary: 非空洞性 —— 拿掉 layer tiebreak,CL&P 必红
    type: code
    pass_when: |
      在 ForceFieldBondedTypifier._best_match 里把 key 从 (score, layer) 改成 (score,),
      tests/test_typifier/test_clp.py 至少一条断言变红(CL&P 是 OPLS 的 layer-1 overlay,
      tiebreak 是它覆盖基层参数的唯一机制)。改回后复绿。
      这条证明 ac-007 的 layer 部分不是空断言。
    status: pending

  - id: ac-010
    summary: AmberToolsTypifier 这条 RegionTypifier 实现未受影响(端到端)
    type: runtime
    pass_when: |
      跑 polymer_builders/peo_gel/build_gel.py:交联数、原子数、净电荷与改动前一致
      (130 crosslinks / 3574 atoms / 净电荷 +9.18e-4 e = 27 × 单链 +3.4e-5 e)。
      需要 AmberTools + LAMMPS,故为 runtime 而非 code。
    status: pending

  - id: ac-011
    summary: 破坏性删除只落在零调用者的符号上
    type: code
    pass_when: |
      改动前的 grep 存档证明:atomtype_matches / retype_region / skip_atom_typing /
      ForceFieldTypifier 直接构造 —— 在 src/ 与 examples/ 里各为 0 处。
      改动后 docs/changelog.md 的 Unreleased/BREAKING 记录这四项 + 六个模块删除。
      (docs/{,zh/}user-guide/15_mcp.md、getting-started/quickstart.md 里的
      skip_atom_typing 属于 OplsTypifier 那条既存腐坏,不在本 spec 范围 —— 见 Out of scope。)
    status: pending
---

# 验收账本 — typifier-architecture

ac-001/002 是"死代码与重名"这一半；ac-003..007 是"三个契约"这一半；
ac-008..011 是回归与诚实性。

ac-009 是本 spec 唯一的变异测试：`layer` tiebreak 是 CL&P overlay 覆盖 OPLS 基层参数的
**唯一**机制，如果把它拿掉而 `test_clp.py` 仍然全绿，那说明 CL&P 的测试根本没在测覆盖。
