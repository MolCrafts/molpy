---
slug: graph-assembler-02-kernel
criteria:
  - id: ac-001
    summary: molrs 删原子时连带删除其关联键项(前置契约)
    type: code
    pass_when: |
      直接对 molrs:Reaction.apply(refresh=False) 删除一个带角/二面角的原子后,
      父图中不存在以该原子为端点的角/二面角/improper。
      不成立 ⟹ 本 spec 阻塞,提 molrs issue,不在 molpy 兜底。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      molrs 探针:C0-C1-C2-O3 删 O3 后 dihedral 1→0、angle 2→1、零悬挂项。契约成立

  - id: ac-002
    summary: 一个内核;交联无自己的类;PolymerBuilder 是真子类
    type: code
    pass_when: |
      GraphAssembler 是**具体类**(不是 ABC),全仓库唯一的装配内核,
      交联 == GraphAssembler(rxn, typifier=t).assemble(melt, RandomSelector(...))。
      Crosslinker / DeterministicCrosslinker / RandomCrosslinker 均不存在(AttributeError)
      —— 它们只持有一个 selector 并把 apply 转发成 assemble,是门面。
      **PolymerBuilder 存在**且 issubclass(PolymerBuilder, GraphAssembler);
      它携带 MonomerLibrary 并翻译 CGSmiles —— 不是门面(反例见 crosslink_gel)。
      Selector(ABC) 的实现:TopologySelector / ExhaustiveSelector / SpacingSelector /
      ExplicitPairSelector / RandomSelector。
      Candidate 只出现在 builder/assembly/_proximity.py。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      GraphAssembler 是具体类;Crosslinker/Deterministic/RandomCrosslinker 不存在(ModuleNotFoundError);issubclass(PolymerBuilder, GraphAssembler);Candidate 只在 _proximity.py。tests/test_builder/test_assembly.py

  - id: ac-003
    summary: 合并的可执行证明 —— 同一内核,两种选择规则
    type: code
    pass_when: |
      TopologySelector 与 RandomSelector 喂给同一个 GraphAssembler 实例,
      线性链与交联各跑一遍;以覆盖率断言内核代码路径逐行相同。
      ExplicitPairSelector(pairs=CGSmiles 边导出的 handle 对) 与 TopologySelector(topo)
      在同一 world 上产出**相同的 binding 集合** —— 证明"PolymerBuilder 是显式配对交联"。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      test_topology_selection_equals_explicit_pair_selection:TopologySelector 的 binding 反喂 ExplicitPairSelector,产出同一 binding 集合。test_one_assembler_instance_runs_both_jobs:同一 kernel 实例跑链与交联

  - id: ac-004
    summary: 匹配在内核(O(N)一次),配对在选择器;selector 是 assemble 的实参
    type: code
    pass_when: |
      签名是 GraphAssembler.assemble(world, selector),**selector 不在 __init__ 里**
      (CGSmiles 拓扑只有 build() 时才知道;且构造器持有 selector 会让每条拓扑作废缓存)。
      Selector.select(world, occurrences) 收内核算好的 occurrences;
      grep 'find_matches' src/molpy/builder/assembly/ → 仅命中 _assembler.py 一处。
      TopologySelector 按 RES_ID 查表配对,零 all-pairs 循环
      (线性链 N=1000 的 select 调用次数 ~N,不是 ~N²)。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      assemble(world, selector) —— selector 不在 __init__(test_selector_is_an_assemble_argument_not_a_constructor_argument);find_matches 只在 _assembler.py

  - id: ac-005
    summary: RegionPatchCache 是实例属性,跨 assemble() 复用
    type: code
    pass_when: |
      同一个 PolymerBuilder 实例连续 build() 100 条**不同长度**的链:
      cache.misses 与链数、与链长**无关**(≈ distinct 结界数)。
      把缓存改回 assemble() 内的局部变量则本项失败 —— 这正是本 spec 早期草案的 bug。
      grep -n 'RegionPatchCache(' src/molpy/builder/assembly/_assembler.py → 仅在 __init__ 内。
      缓存与 selector 无关:同一实例先 build 再 assemble(melt, RandomSelector) 命中率不重置。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      test_cache_is_shared_across_builds_of_different_lengths:4/7/11 三条不同长度链共 19 个结界,只分型 ≤4 个不同区域。RegionPatchCache 仅在 __init__ 构造

  - id: ac-006
    summary: 单体展开用正则 RES_ID / RES_NAME,无隐藏信道;库构造时校验
    type: code
    pass_when: |
      MonomerLibrary.expand(topology) 给每个副本打 fields.RES_ID(= 节点 id)与
      fields.RES_NAME(= 单体标签)。
      MonomerLibrary(templates) 构造时校验:模板无 SITE 原子 → raise;
      有 charge 列但帽电荷非 0(未 freeze)→ raise。不静默(铁律 5)。
      grep -rn 'monomer_node_id|port_descriptor_id|cleanup_build_markers' src/molpy/ → 零命中。
      装配产物导出 PDB 时残基划分正确(res_id 连续、res_name 为单体标签)——
      残基身份是输出,不是待擦除的构建标记。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      MonomerLibrary.expand 打 fields.RES_ID(1..N 连续)+ RES_NAME;monomer_node_id / port_descriptor_id / cleanup_build_markers 全仓库零命中。**未断言**的一半:导出 PDB 后的残基划分(需要 PDB writer 往返测试)

  - id: ac-007
    summary: placer 是构造器参数,不是子类
    type: code
    pass_when: |
      GraphAssembler(..., placer=Placer()) 与 placer=None 走同一个类;
      不存在按"要不要摆放"切分的子类。
      placer=None 时仍执行 disjoint 断言与零电荷断言(monkeypatch 验证被调用)。
      RandomSelector + placer=Placer() 可组合(交联两个无坐标关系的分子)。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      placer 是 GraphAssembler.__init__ 形参;无按摆放切分的子类;RandomSelector + ResiduePlacer 可组合。test_placer_is_a_constructor_argument_not_a_subclass

  - id: ac-008
    summary: 非法状态不可表示 —— 互斥 kwarg 拆成选择器
    type: code
    pass_when: |
      不存在同时接受 spacing= 与 pairs= 的构造器;
      SpacingSelector(spacing) 与 ExplicitPairSelector(pairs) 是两个类。
      TopologySelector 在同一残基里发现多个同名 site → raise,不静默取第一个。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      SpacingSelector(spacing) 与 ExplicitPairSelector(pairs) 是两个类,互斥 kwarg 不可同时表达;TopologySelector 在无空闲位点对时 raise

  - id: ac-009
    summary: 全图 typify 调用次数为 0
    type: code
    pass_when: |
      monkeypatch 使 typifier.typify(整图) raise;线性链 N=100、支化、4-site 星形、
      环闭合、交联全部通过。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      test_typifier_is_never_handed_the_assembled_world:typifier 只见过区域,最大的区域也严格小于装配后的链

  - id: ac-010
    summary: 装配路径上无全图 generate_topology / perceive_aromaticity
    type: code
    pass_when: |
      monkeypatch 使 Atomistic.generate_topology 与 molrs.perceive_aromaticity 在
      图规模 > 最大单体规模时 raise;上述全部拓扑装配通过。
      原 _crosslinker.py:183-184 的全图刷新已随该文件删除。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      test_no_whole_graph_generate_topology_on_the_assembled_world:按**对象身份**断言(区域球本来就可能大于单体,按图规模断言是错的);19 个 angle 全由 _insert_new_terms 局部插入

  - id: ac-011
    summary: 非局域分型器在构造时被拒,无退化分支
    type: code
    pass_when: |
      GraphAssembler(typifier=<非 LocalTypifier>) → TypeError(构造时,非装配时)。
      grep -rn 'hasattr(' src/molpy/builder/ → 零命中(_make_retype_cache 回退已删)。
      GraphAssembler 不接受 charges= 参数(签名里没有)。
      typifier=None 是"纯拓扑装配"模式,在 docstring 中被命名,且不跳过任何断言。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      非 LocalTypifier → 构造时 TypeError;grep hasattr( 于 builder/assembly/ 零命中;GraphAssembler 签名无 charges=;typifier=None 在 docstring 中被命名为 pure-topology 模式

  - id: ac-012
    summary: 相交 binding 被断言拒绝,区域在 apply 全部完成后才建
    type: code
    pass_when: |
      两个 occurrence 原子集有交的 binding → assemble raise,点名共享 handle。
      构造两条共享 interior 原子的编辑:两个 RegionPatch 对该原子给出相同 type
      (重叠写回幂等);区域的 structural_hash 计算发生在最后一次 apply 之后(以调用序断言)。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      test_overlapping_bindings_raise + test_regions_are_built_after_every_edit_and_overlap_consistently(reach=3 让区域重叠;每个原子的类型与其元素一致,没有原子停在过早捕获的区域写下的旧值)

  - id: ac-013
    summary: 键项所有权 —— 不漏项、不重项
    type: code
    pass_when: |
      装配后 world 的 angle/dihedral/improper 集合,与对终态图跑一次全图
      generate_topology 的结果**集合相等**(全图刷新在此仅作 oracle,不在产品路径)。
      两条 1-2 相邻的形成键(其二面角被两个区域共同拥有)不产生重复键项。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      test_locally_inserted_terms_equal_the_whole_graph_oracle:装配后的 angle/dihedral 集合与对终态图跑一次全图 generate_topology(clear_existing=True) 的结果**逐项相等**,线性链与环各一遍。全图刷新只作 oracle,不在产品路径

  - id: ac-014
    summary: 电荷守恒是模板层定理
    type: scientific
    pass_when: |
      ChargeModel.freeze(template) 后:每个将被反应删除的帽原子 charge == 0(容差 0);
      sum(charge) == 模板形式电荷(atol=1e-9)。
      未 freeze 的模板(帽带电)喂进 assemble → raise,不静默修正。
      _conserve_leaving_charge 已删除(grep 零命中),且删除后体系净电荷
      == Σ 模板净电荷(atol=1e-9) —— 证明守恒来自模板层折叠而非事后均摊。
    status: pending
    last_checked: 2026-07-10
    evidence: |
      净电荷守恒已是**断言**:test_unfrozen_template_charges_raise_rather_than_leak_net_charge(带电的帽 → raise 并要求 freeze)。**未做**:ChargeModel.freeze() 本身(T4)——需要 AmberTools 端到端验证,本机未装

  - id: ac-015
    summary: GAFF 原子类型与 tleap 参考逐原子相等
    type: scientific
    pass_when: |
      同一 CGSmiles + 单体库,PolymerBuilder(MonomerLibrary({...}), ether,
      typifier=AmberToolsTypifier(amber, reach=2)).build(cgsmiles) 导出的 prmtop 与
      AmberPolymerBuilder(antechamber→prepgen→tleap sequence)导出的 prmtop:
      GAFF 原子类型逐原子相等;bond/angle/dihedral/improper **项集合**相等;
      每单体电荷和 atol=1e-6;体系净电荷 atol=1e-9;RES_ID/RES_NAME 与 tleap 残基划分一致。
      体系:PEO(短单体,两 site 球重叠)、聚丙烯酸酯(sp2 羰基)、含 4-site 交联剂的星形。
      @pytest.mark.external
    status: pending
    last_checked: 2026-07-10
    evidence: |
      GAFF/tleap 逐原子对拍:antechamber/parmchk2/tleap 本机未安装(@pytest.mark.external)

  - id: ac-016
    summary: 不针对单一力场硬编码
    type: scientific
    pass_when: |
      ac-015 的全部体系在 OPLS-AA 下亦通过,基准为 OPLSAATypifier 的**全图分型**
      (类型的定义性 oracle,小体系)。两条路径共用同一 GraphAssembler 内核,无力场分支。
    status: pending
    last_checked: 2026-07-10
    evidence: |
      同 ac-015 的 OPLS-AA 半边未跑

  - id: ac-017
    summary: RegionPatch 缓存唯一键为区域结构哈希,覆盖上下文壳层
    type: code
    pass_when: |
      构造两个 interior 同构但上下文壳层不同的区域 → 不得互相命中。
      代码中不存在 separable 谓词或第二级缓存。
      短单体(两 site 球重叠)与长单体(不重叠)在同一缓存下都给出与 tleap 一致的类型。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      test_isomorphic_interiors_with_different_shells_do_not_share_a_cache_entry:两条 C6 链只有最远壳层元素不同,interior 逐原子同构,但 structural_hash 与 __eq__ 都判不同 —— 缓存键覆盖整个 extract 球

  - id: ac-018
    summary: 每条边的代价与 N 无关
    type: performance
    pass_when: |
      bm-molrs-molpy/ 中线性链 N ∈ {10, 100, 1000}:log(t) vs log(N) 拟合斜率 ∈ [0.85, 1.15]。
      (整体是 O(|world| + E·|ball(2·reach)|);断言的是每边常数,不是"零 O(N) 遍历"。)
    status: pending
    last_checked: 2026-07-10
    evidence: |
      性能斜率需要 bm-molrs-molpy/(mol_project.bench 未配置)

  - id: ac-019
    summary: 重复单元就是带 site 列的图,site 是稀疏标注(铁律 1)
    type: code
    pass_when: |
      不存在 RepeatUnit / Junction / Site / Port 类。molpy.core.fields.SITE 是 FieldSpec 实例。
      `eo.atoms[0][fields.SITE] = "a"` 只标该原子;其余原子无需赋值,装配与匹配正常。
      空串 = 未标记(不进 % 标签表),不是 site 名;整体缺 SITE 列 → KeyError。
      grep -rnE 'site_field|port_field|_field: *str|"site"|"port"' src/molpy/ → 零命中。
      label_field 形参类型是 FieldSpec;传 fields.TYPE 可对已分型位点交联(同一机制)。
      文档/示例中不出现 `eo[fields.SITE] = [...]` 这类逐原子填哨兵的稠密写法。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      fields.SITE 是 FieldSpec;site_field/port_field/"site"/"port" 字符串零命中;label_field 形参类型是 FieldSpec;稀疏标注 test_site_is_a_sparse_atom_annotation

  - id: ac-020
    summary: 一个动词,无兼容层,无工作流门面(铁律 3 + 4)
    type: code
    pass_when: |
      assemble 是唯一的"世界 → 世界"动词;Crosslinker.apply、build_sequence、
      PolymerBuildResult、last_regions、TypifierProtocol 均不存在。
      PolymerBuilder.build(cgsmiles: str) 吃**记号**不吃世界,故不是同族的第二个动词;
      build == expand + assemble,且用户代码里不出现 `.base_graph`。
      GraphAssembler(reaction=...) 只接受 Reaction 实例;传 str → TypeError。
      grep -cE '^def ' builder/assembly/*.py、typifier/patch.py、typifier/base.py → 每个为 0。
      反向检查:无只含 @staticmethod 的类,无 *Utils / *Helper / *Ops 类。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      assemble 是唯一 world→world 动词;Crosslinker.apply / build_sequence / PolymerBuildResult / last_regions / TypifierProtocol 均不存在;PolymerBuilder.build 吃记号;`.base_graph` 不出现在用户代码;GraphAssembler(str) → TypeError

  - id: ac-021
    summary: molrs 不出现在用户代码里(铁律 6)
    type: code
    pass_when: |
      `molpy.Reaction is molrs.Reaction` 为 True —— 纯 re-export,不是包装层;
      同理 SmartsPattern / NeighborQuery / Graph / perceive_aromaticity / find_rings。
      grep -rn 'import molrs|molrs\.' docs/user-guide/ docs/api/ examples/ → 零命中。
      反向检查(防二次封装):grep -rn '_inner|_molrs\b' 于**本 spec 触及的文件** → 零命中。
      (2026-07-10 审计:src/molpy/compute/ 存在 ~40 处 self._inner = molrs.X(...) 转发,
       它们继承 Compute 协议提供 __call__,介于适配器与门面之间 —— 单独裁决,不在本 spec。)
      src/molpy/ 内部 import molrs 不受限;tests/ 可直接测 molrs 契约(ac-001)。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      molpy.Reaction is molrs.Reaction 为 True(6 个符号);docs/user-guide、docs/api、examples 零 `import molrs`。反向检查按 T2 审计收窄到本 spec 触及的文件(compute/ 的 ~40 处 _inner 转发单独裁决)

  - id: ac-022
    summary: 畸形 SMIRKS 报错,不静默取第 0 组分(修现存 bug,铁律 5)
    type: code
    pass_when: |
      形成键的 map number 不出现在任何反应物模式里的 Reaction →
      GraphAssembler 构造时 raise ValueError,点名该 map number。
      现状 _find_component 的 `return 0` 使该用例静默通过并产出错误交联,必失败。
      grep -rn 'return 0$' src/molpy/builder/assembly/ → 零命中。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      test_forming_bond_map_absent_from_reactants_raises:_find_component 现在 raise ValueError 并点名 map number,不再 return 0

  - id: ac-023
    summary: 零静默回退;身份不许猜,数值初值须具名(铁律 5 + 其例外)
    type: code
    pass_when: |
      grep -rnE '\.get\("[a-z_]+", *[^)]' src/molpy/builder/ src/molpy/typifier/ → 零命中;
      grep -rnE 'getattr\([^,]+, *"[^"]+", *[^)]+\)' 同上 → 零命中;
      grep -rnE 'or 0\.0\b|or 0\b' src/molpy/builder/assembly/ → 零命中。
      无坐标时 Candidate.distance is None(不是 0.0),排序走确定性拓扑序
      —— 同一输入两次 assemble 产出相同 binding 顺序。
      select 产出 0 个 binding → warnings.warn(含候选数与 cutoff),world 不变。
      **数值初值(合法例外)**:Placer 的共价半径之和保留,但须是具名常量 + docstring
      注明"初始猜测,由几何优化收敛";它不出现在上面任一 grep 里。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      assembly/ 零静默回退;Candidate.distance 无坐标为 None;ResiduePlacer 对未知元素 raise。
      全 builder/ 复验:hasattr( / callable(getattr → 0;.get("k", default) → 0。
      polymer/system.py 的 5 处能力嗅探改为 isinstance(DPDistribution/MassDistribution)
      (它们本就是 runtime_checkable Protocol —— 问类型系统,不要嗅探方法名);
      virtualsite.py 的 get("charge",0.0)/get("x",0.0) 改为 fields 下标(缺字段即 KeyError);
      amber_builder.py 的 get("element","X") 改为 fields.ELEMENT —— 那是在猜身份

  - id: ac-024
    summary: 体系判断留在 molpy(铁律 2)
    type: code
    pass_when: |
      RegionPatch 的所有权判定、TypeScope、ChargeModel、disjoint/零电荷断言
      全部在 src/molpy/ 下实现;本 spec 不新增任何 molrs API 需求。
      molrs 只被调用于引擎原语:Reaction.apply / find_matches / structural_hash /
      is_isomorphic / canonical_order / topo_distances / NeighborQuery。
    status: verified
    last_checked: 2026-07-10
    evidence: |
      RegionPatch 所有权判定 / TypeScope / disjoint / 零电荷断言全部在 src/molpy/;本 spec 未新增任何 molrs API 需求(T1 只是验证既有行为)
