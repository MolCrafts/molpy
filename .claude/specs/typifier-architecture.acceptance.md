---
slug: typifier-architecture
criteria:
  - id: ac-001
    summary: Typifier 对图类型泛型;基类不假定原子体系(判据不是文本禁令)
    type: code
    pass_when: |
      Typifier 是泛型类,类型参数上界是 MolGraph(= molrs.Graph);
      ClpTypifier / AmberToolsTypifier 特化为 Typifier[Atomistic],
      测试里的 fake CG 分型器特化为 Typifier[CoarseGrain] —— 两者共用同一个 typify。
      base.py 的 typify 只调用 graph.nodes / graph.links.classes() /
      graph.complete_valence() / graph.copy();AST 断言其中不出现
      Bond / Angle / Dihedral / Improper / ForceField 这些**具体力场与元素分解**的名字
      (`Atomistic` 作为类型标注出现在具体分型器里是正当的,不在禁止之列)。
      `Bond→BondType` 这张映射全库只在 typifier/forcefield.py 出现一次。
      不存在 src/molpy/typifier/atomistic.py。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      tests/test_typifier/test_contract.py::TestGenericOverTheGraph (3 条:base.py 无力场分解词汇 / _FF_TYPE_OF 仅在 forcefield.py / typify 不猜截断)。Typifier[G: molrs.Graph];ClpTypifier 与 AmberToolsTypifier 特化 Atomistic,_BeadTypifier 特化 CoarseGrain,共用同一 typify (2026-07-10)

  - id: ac-002
    summary: 一条流水线,唯一的抽象步骤是 match;没有 ParamTypifier 这种东西
    type: code
    pass_when: |
      Typifier.__abstractmethods__ == {"match"};typify 不是抽象的。
      ClpTypifier / AmberToolsTypifier 的 __dict__ 里都不含 "typify"
      (流程只有一份代码,子类只重写 match)。
      Typifier() 直接构造 TypeError 且信息含 "abstract"(来自 ABCMeta,非手写守卫)。
      molpy.typifier 导出的每一个 *Typifier 都以某个力场或某个工具命名
      (ClpTypifier / AmberToolsTypifier / OPLSAATypifier / MMFFTypifier);
      不存在 ParamTypifier / ElementTypifier / AtomTypifier / RegionTypifier /
      ForceFieldTypifier / TypeScope / PairTypifier 中的任何一个。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      test_contract.py::TestOnePipeline + ::TestNamingAndSurface。__abstractmethods__=={'match'};三个分型器 __dict__ 均无 typify;__all__ 中的 *Typifier 恰为 {Clp,AmberTools,OPLSAA,MMFF};7 个被否名字 hasattr 全 False (2026-07-10)

  - id: ac-003
    summary: ForceFieldParams 是零件,不是分型器
    type: code
    pass_when: |
      not issubclass(ForceFieldParams, Typifier);ForceFieldParams 没有 typify 方法。
      它有 match(graph, node_types=None) 与 assign(graph) 两个入口。
      ClpTypifier.match 与 AmberToolsTypifier.match 都把原子类型交给它
      (源码里各自不超过 ~6 行,不重复实现 bonded 匹配)。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      test_forcefield.py::TestItIsNotATypifier + test_contract.py。not issubclass(ForceFieldParams, Typifier);无 typify;ClpTypifier.match/AmberToolsTypifier.match 各 <=6 行,均转交 ForceFieldParams (2026-07-10)

  - id: ac-004
    summary: 既存 bug —— complete_valence 保留所有 link kind
    type: code
    pass_when: |
      4 原子链(3 bonds / 2 angles / 1 dihedral)经 complete_valence 后仍是
      2 angles / 1 dihedral(今天是 0 / 0 —— 只拷贝了原子与键)。
      帽子只新增键,不为帽子生成新的角/二面角(它们只是上下文,不进写回集合)。
      改动前该断言必失败。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      tests/test_core/test_capping.py::TestCompletionPreservesTopology(3 条)。4 原子链 2角/1二面角 补全后仍 2/1(改前 0/0);帽子只加键(10 个 H → +10 键,角数不变);link data 原样携带 (2026-07-10)

  - id: ac-005
    summary: 补全无分支,且对完整图恒等
    type: code
    pass_when: |
      AST 断言:Typifier.typify 源码里对 complete_valence() 的调用不在任何 if/try 分支内。
      对水/乙醇/苯/三甘醇/乙酸甲酯/吡啶/二甲砜/[Na+] 八个完整分子,
      complete_valence() 前后原子数不变(已实测,作为回归锁住)。
      CoarseGrain.complete_valence() 存在且返回等价拷贝(bead 无价键可补)。
      typifier/ 里 grep 不到 isinstance(graph, Atomistic) 之类的图类型分支。
      _GraphViews.nodes 存在,Atomistic 与 CoarseGrain 都能应答。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      test_contract.py::test_the_party_that_cut_the_graph_completes_it_unconditionally + test_capping.py::TestCompletionIsIdentityOnCompleteMolecules(7 分子)。**范围修正**:补全从 typify 移到 RegionTypes.of(见 spec\"实施期的一处修正\") —— 截断是身份不是数值(铁律 5),CL&P 纯连接图证伪了原方案。RegionTypes.of 里 complete_valence 恰 1 次且无 if/try;CoarseGrain.complete_valence 恒等;_GraphViews.nodes 两种图均应答 (2026-07-10)

  - id: ac-006
    summary: typify 是 graph -> 新 graph,入参逐元素不变
    type: code
    pass_when: |
      typed = ClpTypifier().typify(g);断言 typed is not g;g 的每个 node/link 的 data
      与调用前逐键相等(深比较快照);len(typed.nodes) == len(g.nodes)(帽子已剥掉);
      typed 的角/二面角数与 g 相同。
      AmberToolsTypifier 同断言(@pytest.mark.external)。两者互不为子类。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      test_contract.py::TestTypifyReturnsANewGraph + test_ambertools_typifier.py。typed is not g;入参逐原子未变;帽子剥落;两分型器互不为子类 (2026-07-10)

  - id: ac-007
    summary: 契约不排斥粗粒化 —— 20 行 fake CG 分型器只写 match 即可接入
    type: code
    pass_when: |
      测试内定义 _FakeCGTypifier(Typifier[CoarseGrain]),只实现 match(cg) -> Match,
      给 bead 打类型。断言:
        isinstance(_FakeCGTypifier(), Typifier);
        _FakeCGTypifier().typify(cg) 返回 CoarseGrain,bead 带上类型;
        RetypeCache(_FakeCGTypifier()) 可构造;
        GraphAssembler(rxn, typifier=_FakeCGTypifier(), reach=1) 不报错。
      **这条是本设计的证明**:若要让它通过必须修改 base.py,说明泛型层里仍焊着原子体系的
      假设(元素分解,或"补全"只对 Atomistic 有意义这件事)。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      test_contract.py::TestCoarseGrainNeedsNoContractChange(5 条)。20 行 _BeadTypifier(Typifier[CoarseGrain]) 只写 match:isinstance 通过、typify 返回带类型的 CoarseGrain、RetypeCache 可构造、GraphAssembler(reach=1) 接受。**base.py 一个字符未改** (2026-07-10)

  - id: ac-008
    summary: 科学 —— 补全后的区域分型与全图 oracle 逐原子相等;裸切的偏差被记录
    type: scientific
    pass_when: |
      取一个切口落在 sp3 碳上的体系(补全后切断处成 -CH3),用 reach=2 抽区域。
      oracle = typifier.typify(whole_graph) 的 interior 原子类型(类型的*定义*)。
      断言新流水线(第 ① 步无条件补全)的 interior 类型与 oracle **逐原子相等**。
      同时测量并记录旧的裸切路径的偏差原子数。
      理据:两半径定理里 extract_radius = interior_reach + reach,所以 interior 原子的
      感受野恰好伸到 boundary 原子;boundary 裸切时价键不满,在 SMARTS 眼里是自由基。
      若裸切偏差 > 0 ⟹ graph-assembler-01 遗留的科学正确性 bug,本 spec 修掉;
      若偏差 == 0 ⟹ 把体系与力场记进 evidence("此模式集不看切口度数"),仍保留补全。
      **两种结果都要写进 evidence,不得只报通过。**
    status: verified
    last_checked: 2026-07-10
    evidence: |
      tests/test_typifier/test_region_radii.py::test_a_raw_slice_of_an_aromatic_ring_cannot_be_typed_at_all + ::test_every_capped_slice_is_typable_and_agrees_with_the_oracle。实测 OPLS-AA reach=2:补全 3 体系全 0 错 0 拒;裸切对二甲苯 19 切片中 **12 个被分型器拒绝**(截断芳香碳 → 角无键型),PEO/丙烯酸甲酯裸切碰巧 0/0。⟹ 补全是前提不是便利;这是 graph-assembler-01 遗留的科学正确性 bug,本 spec 修掉 (2026-07-10)

  - id: ac-009
    summary: bonded 分型全库只剩一份实现;core 不再做力场判断
    type: code
    pass_when: |
      Atomistic.assign_bonded_types 不存在;grep -rn "assign_bonded_types" src/ → 零命中。
      ForceFieldParams(ff).assign(g) 在同一个已分型图上贴出的 bonded type 是
      旧 assign_bonded_types 的**超集**:构造一个只有 class 键控 BondType 的 ff,
      旧法贴不上、新法贴得上。
      且一个匹配不上的 term 现在 raise(strict)而不是静默跳过。
      polymer_builders/peo_gel/build_gel.py 改用 ForceFieldParams。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      core.Atomistic.assign_bonded_types 已删(32 行);grep src/ 零命中。实测超集:class 键控的 BondType('CT-HC') 上,旧法贴 None、新法贴 'CT-HC' 并写入 k/r0。test_forcefield.py::test_a_class_keyed_type_matches_an_atom_of_that_class;build_gel.py:168 改用 ForceFieldParams(net_ff, strict=False).assign (2026-07-10)

  - id: ac-010
    summary: 修既存 bug —— 未分型的 bonded term 不进快照,也不把母图类型抹成 None
    type: code
    pass_when: |
      母图那根键先带 type="X",其 ff 不含该键的 bonded 类型;
      跑 RegionTypes.of + apply_to 之后,母图那根键的 type 仍是 "X",不是 None;
      且 RegionTypes.bonds 里不含任何 type is None 的条目。
      改动前该断言必失败。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      region.py::_capture_links 跳过 type is None 的 term(并跳过触及帽子的 term);test_forcefield.py::test_an_untyped_endpoint_leaves_the_bond_undecided 锁住\"未决定 != None\" (2026-07-10)

  - id: ac-011
    summary: TypeScope 溶解;relaxed / typify_region 全消失;reach 归装配器
    type: code
    pass_when: |
      molpy.typifier.scope ModuleNotFoundError;TypeScope 在 src/ 零命中。
      AffectedRegion.around(g, touched, reach=r) 对 r ∈ {1,2,3}:
        region.interior == ball(touched, max(r, AffectedRegion.TERM_REACH));
        region.extract_radius == max(r, TERM_REACH) + r;球外原子逐原子未被写。
      GraphAssembler(rxn, typifier=t) 不给 reach → TypeError;
      AmberToolsTypifier 构造器不再有 reach=。
      relaxed / typify_region / retype_region / _typify_relaxed 在 src/ 零命中。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      test_region_radii.py(26 条)。molpy.typifier.scope ModuleNotFoundError;TypeScope 在 src/ 零命中;AffectedRegion.around 对 r∈{1,2,3,4} 半径算术正确;球外原子未被写;GraphAssembler 无 reach → TypeError('reach= is required');AmberToolsTypifier(amber, reach=2) → TypeError;relaxed/typify_region/retype_region/_typify_relaxed 在 src/ 零命中 (2026-07-10)

  - id: ac-012
    summary: 死模块消失,且包内不再有重名符号
    type: code
    pass_when: |
      importlib.import_module("molpy.typifier.{bond,angle,dihedral,pair,mmff,atomistic,scope}")
      七个全部 ModuleNotFoundError(base.py 仍在,但内容全新)。
      AST 扫描 molpy.typifier 所有子模块的 top-level class/def 名字:
      没有任何名字在两个不同子模块里被定义两次。
      改动前该断言必失败并报出 TypifierBase / PairTypifier / atomtype_matches 三对。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      test_contract.py::TestNamingAndSurface::test_the_dead_modules_are_gone(7 模块 ModuleNotFoundError)+ ::test_no_symbol_is_defined_twice_in_the_package(AST 扫描,零重名) (2026-07-10)

  - id: ac-013
    summary: 打分规则逐字保留,且力场只扫一次
    type: code
    pass_when: |
      TypeClassIndex.score:
        exact(("opls_135","opls_140")) > class(("CT","HC")) > wildcard(("*","*")) == 0;
        某一位配不上 → None;正反两向同分;
        ("*","CT","CT","*") 严格小于 ("opls_135","CT","CT","opls_135");
        元数不匹配 → ValueError(不再静默 zip 截断)。
      spy 计 ForceField.get_types(AtomType) 调用次数:一个 ForceFieldParams 构造期间
      恰好 1 次(改动前 3 次,三个 bonded typifier 各扫一遍)。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      tests/test_typifier/test_matching.py(9 条)+ test_forcefield.py::test_the_forcefield_is_scanned_once_not_once_per_bonded_kind(spy 计 get_types(AtomType) == 1,改前 3) (2026-07-10)

  - id: ac-014
    summary: 非空洞性 —— 拿掉 layer tiebreak 必须有测试变红(前提已被实测修正)
    type: code
    pass_when: |
      在 _TermMatcher._best 里把 key 从 (score, layer) 改成 (score,),
      至少一条断言变红;改回后复绿。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      **spec 的原前提是错的,实测推翻。** 原文写"layer tiebreak 是 CL&P overlay 覆盖 OPLS
      基层参数的唯一机制,拿掉它 test_clp.py 必红"。实跑变异:test_clp.py **21 条全绿**。
      进一步实测真实 CL&P 力场(329 BondType / 1003 AngleType / 1139 DihedralType,
      其中 128/556/958 个模式位于 layer>=1):在 120 个真实原子类型的 7140 对组合中,
      **没有任何一次 bond 匹配是靠 layer 打破平局的**(0 次)。CL&P 覆盖 OPLS 靠的是它自己的
      类型携带自己的 class,直接匹配上不同的 bonded type,而不是靠 tiebreak。
      ⟹ tiebreak 是真实但极少被触发的规则,test_clp.py 从未覆盖它。
      改为定点测试 test_forcefield.py::TestOverlayLayerTiebreak:构造真正的平局
      (("CX","B")=1+3 与 ("A","CY")=3+1 同分,layer 分别为 0/1),且把 layer-0 的模式**先**入表
      —— 否则去掉 layer 后靠 `>` 的严格比较,结果会碰巧不变(已实测两种声明顺序,见 evidence)。
      变异 → test_the_overlay_wins_the_tie 变红(1 failed, 18 passed);还原 → 19 passed。
      **后续**:layer tiebreak 在真实 CL&P 上疑似死代码,值得单独裁决(本 spec 声明"打分规则
      逐字保留",故不动它)。

  - id: ac-015
    summary: 全量回归绿
    type: code
    pass_when: |
      ruff format --check + ruff check + ty check 全绿;
      pytest tests/ -m "not external" 通过数不低于改动前(1785),0 失败;
      tests/test_typifier/ + tests/test_core/test_capping.py +
      tests/test_builder/test_assembly.py + test_clp.py +
      test_io/test_forcefield/test_xml.py 全绿。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      ruff format/check + ty check 全绿;pytest -m 'not external' → **1843 passed, 0 failed**(改前 1785;净增 58 条) (2026-07-10)

  - id: ac-016
    summary: AmberTools 那条路端到端未受影响
    type: runtime
    pass_when: |
      跑 polymer_builders/peo_gel/build_gel.py:交联数、原子数、净电荷与改动前一致
      (130 crosslinks / 3574 atoms / 净电荷 +9.18e-4 e = 27 × 单链 +3.4e-5 e);
      bonds/angles/dihedrals 计数不变(3677 / 7020 / 13921)。
      需要 AmberTools + LAMMPS,故为 runtime。
      同时验证:流水线第 ① 步只补全一次(AmberToolsTypifier 内不再自己 complete_valence);
      AmberTools 路径现在也经 ForceFieldParams 贴 bonded 参数 —— 这是**新行为**,
      须确认它没有把电荷经 _scalar_delta 捎带进 RegionTypes(净电荷不变即证)。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      polymer_builders/peo_gel/build_gel.py 端到端(AmberTools + LAMMPS):27 copies → 3834 atoms;**130 crosslinks → 3574 atoms(118 未反应)**;**3677 bonds / 7020 angles / 13921 dihedrals**;lmp_mpi push-off + minimize + NVT 通过。净电荷 **+9.180000e-04 e** == 27 × 单链 +3.4e-5 e,与基线逐字一致 ⟹ AmberTools 路径新增的 bonded 参数写回**未**把电荷经 _scalar_delta 捎带进 RegionTypes。AmberToolsTypifier 内已无 complete_valence(补全在 RegionTypes.of 一次) (2026-07-10)

  - id: ac-017
    summary: 破坏性删除只落在零调用者的符号上;reach 降级为设置项一事被记录
    type: code
    pass_when: |
      改动前的 grep 存档证明:atomtype_matches / retype_region / skip_atom_typing /
      ForceFieldTypifier 直接构造 —— 在 src/ 与 examples/ 里各为 0 处;
      PairTypifier 在 typifier/ 之外 0 处;.scope 的唯一生产消费者是 _assembler.py:137;
      assign_bonded_types 的唯一非 core 调用者是 build_gel.py:168。
      docs/changelog.md 的 Unreleased/BREAKING 记录:七个模块删除、TypeScope 移除、
      ForceFieldTypifier 移除、RegionTypifier/typify_region/relaxed 移除、
      PairTypifier 转私有、core.Atomistic.assign_bonded_types 移除(→ ForceFieldParams)、
      core.capping.complete_valence 行为修正(保留高阶 link)、
      GraphAssembler 新增必填 reach=、AmberToolsTypifier 丢掉 reach=。
      并明确写出:reach 由分型器声明改为装配器参数,配错 reach 不再由构造器拦截,
      而由 graph-assembler-01 ac-002 的 oracle 测试拦截。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      docs/changelog.md 的 Unreleased/BREAKING + Fixed 已记录:一个 Typifier 契约;8 个模块删除;ForceFieldTypifier/RegionTypifier/TypeScope/typify_region/retype_region/relaxed/atomtype_matches/skip_*_typing/公开 PairTypifier 全部移除;core.Atomistic.assign_bonded_types → ForceFieldParams(超集 + 不再静默);GraphAssembler 新增必填 reach=,AmberToolsTypifier 丢掉 reach=;complete_valence 保留高阶 link;裸切区域的科学 bug(对二甲苯 19 切片 12 个被拒);未分型 bonded term 不再写回 None。**并明确写出**:reach 由分型器声明改为装配器参数,这道防线从类型系统降级到测试套件(实测配错时 PEO junction 46 个写回原子错 22 个);且注明 AmberToolsTypifier 本就由用户传 reach。改动前 grep 存档:atomtype_matches/retype_region/skip_atom_typing/ForceFieldTypifier 直接构造 在 src+examples 各 0 处;PairTypifier 在 typifier/ 外 0 处;.scope 唯一生产消费者 _assembler.py:137;assign_bonded_types 唯一非 core 调用者 build_gel.py:168 (2026-07-10)
---

# 验收账本 — typifier-architecture

**ac-001 / ac-002 / ac-007 是本设计的三条证明。**

ac-001 的判据**不是**"`base.py` 里不许出现 `Atomistic` 这个词"（v5 那条是矫枉过正）。
`Typifier` 对图类型是泛型的：有的分型器吃 `Atomistic`，有的吃 `CoarseGrain`，
把 `G` 特化成 `Atomistic` 天经地义。真正要禁的是**基类假定原子体系**：
`base.py` 只许用 `graph.nodes` / `graph.links.classes()` / `graph.complete_valence()`，
而 `Bond→BondType` 这张映射全库只出现一次。

ac-002 除了"子类 `__dict__` 里没有 `typify`"，还锁住一条命名规则：
**导出的每个分型器都以某个力场或某个工具命名。**
`ParamTypifier` 这种东西不存在 —— "给我一个图和一个力场，把 bonded 项贴上标签"不是一种
分型器，是每个力场分型器的后半段（ac-003 的 `ForceFieldParams`）。

ac-007：一个 20 行的 fake 粗粒化分型器只写 `match` 就能接入，且**无需改动 `base.py`**。

**ac-004 是一个必须先修的既存 bug。** `complete_valence` 只拷贝原子与键：
一个 3 键的丁烷片段（2 角 1 二面角）补全后变成 0 角 0 二面角。
一个名为"返回补全后的分子"的函数静默丢掉拓扑（铁律 5）。

**ac-008 是唯一的科学判据，且可能翻出第二个既存 bug。**
两半径定理里 `extract_radius = interior_reach + reach`，interior 原子的感受野恰好伸到
boundary 原子。boundary 裸切时价键不满 —— 它是 interior 原子环境的一部分。
今天只有 AmberTools 那条路补全。两种结果都必须写进 evidence，不得只报"通过"。

**ac-009 收掉第三份 bonded 实现。** `core.Atomistic.assign_bonded_types` 是最粗的一份
（`name.split("-")` 精确匹配，无通配/class/layer），住在数据模型层，且静默跳过。

ac-014 是唯一的变异测试：`layer` tiebreak 是 CL&P overlay 覆盖 OPLS 基层参数的**唯一**机制。

ac-017 里那句关于 `reach` 的记录不是形式主义：实测 `reach` 取错时，一个 PEO junction 上
46 个写回原子里有 22 个分错。把 `reach` 从分型器挪到装配器参数，等于把这道防线从类型系统
降级到测试套件，必须在 changelog 里说出来。
