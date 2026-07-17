---
title: typifier 包的架构与继承关系整理
status: code-complete
created: 2026-07-10
revised: 2026-07-10  # v6：Typifier 对图类型泛型；ForceFieldParams 是零件不是分型器
depends_on: "graph-assembler-01/02/03 (code-complete)"
blocks: "opls-typifier-downsink (它落进来的位置就是一个 match 实现)"
---

> 2026-07-14 correction: the `capping` concept and both
> `Atomistic.complete_valence()` / `CoarseGrain.complete_valence()` facades were
> removed. A cut site that needs chemical completion invokes
> `Atomistic.adopt(molrs.Perceive().find_hydrogens(region))` directly.
> Coarse-grained graphs expose no no-op chemistry method. Older
> `complete_valence` wording below records the superseded implementation plan.

# `Typifier[G: MolGraph]` —— 一条流水线，只有 `match` 不一样

> 只整理 `src/molpy/typifier/` 的类层级与文件边界（外加 `core/` 里三处必须先修的东西）。
> 不碰打分常数与 `reach = 2` 这个数本身。
> 不做 OPLS 下沉后的重接（`opls-typifier-downsink`，阻塞在 molrs 发版）。

## Summary

**一个 Typifier 就是 `MolGraph -> MolGraph`，对图类型是泛型的。**
有的分型器吃 `Atomistic`，有的吃 `CoarseGrain` —— 两者都是 `MolGraph` 的子类。
**所有 Typifier 的流程完全一样，只有匹配那一步不同。**

```python
class Typifier[G: MolGraph](ABC):
    def typify(self, graph: G) -> G:              # 具体，子类不重写
        capped = graph.complete_valence()         # ① 截断处一律补全（图自己回答怎么补）
        typed  = graph.copy()                     # ② 返回新图，入参不变
        self.match(capped).write_onto(typed)      # ③ 逐 node / 逐 link kind 写回
        return typed

    @abstractmethod
    def match(self, graph: G) -> Match: ...       # 唯一的抽象步骤

@dataclass(frozen=True)
class Match:
    nodes: tuple[Mapping[str, ParamValue], ...]                        # 与 graph.nodes 位置对齐
    links: Mapping[type[Link], tuple[Mapping[str, ParamValue], ...]]   # 逐 link kind
```

基类不假定原子体系：它只用 `graph.nodes`、`graph.links.classes()`、`graph.complete_valence()`
—— 全是 `MolGraph` 层的词汇。具体分型器把 `G` 特化成 `Atomistic` 或 `CoarseGrain`，天经地义。

**分型器按力场 / 工具命名，从不按体系命名。**

| 分型器 | `G` | `match` 干什么 |
|--------|-----|---------------|
| `ClpTypifier` | `Atomistic` | molrs SMARTS（CL&P overlay）定原子类型 |
| `AmberToolsTypifier` | `Atomistic` | antechamber 定 GAFF 类型，并产出力场 |
| （将来）`OplsTypifier` | `Atomistic` | opls-typifier-downsink 落进来的就是这一个 `match` |
| （将来）`MMFFTypifier` | `Atomistic` | molrs MMFF94 |
| （将来）`CoarseGrainTypifier` | `CoarseGrain` | bead 类型 —— 流水线一个字不改 |

没有 `typifier/atomistic.py`：`atomistic` 不是一种分型器。
也没有 `ParamTypifier` 这种东西：**「给我一个图和一个力场，把 bonded 项贴上标签」不是一种
分型器**，它是每个力场分型器的**后半段**，是一个零件：

```python
class ForceFieldParams:          # 不是 Typifier，不假装是
    """从 node 的类型 + 一个 ForceField，标注 pair 与 bonded 参数。"""
    def __init__(self, forcefield, *, strict: bool = True) -> None: ...
    def match(self, graph, node_types=None) -> Match: ...   # None = 读图上已有的类型
    def assign(self, graph) -> MolGraph: ...                # 取代 core.assign_bonded_types
```

`ClpTypifier.match` 与 `AmberToolsTypifier.match` 各自四行：拿到原子类型，交给
`ForceFieldParams`。它是全库**唯一**知道 `Bond→BondType` 这类映射的地方。

## 现状：五层病

**① 噪音。** 13 文件 1756 行里 **6 个文件约 340 行是死代码**
（`base.py` `bond.py` `angle.py` `dihedral.py` `pair.py` `mmff.py`），包外零 import。
更糟的是它们与活代码**重名**：`TypifierBase`、`PairTypifier`、`atomtype_matches`
各在包内定义**两次**。`base.py` 的 docstring 写着 "Base class for all typifiers" —— 它不是。

**② 契约缺失。** `TypifierBase[T]` 同时是"分一根键"和"分一整个结构"的父类，
`typify` 一个动词两种互斥语义（就地改并返回自己 vs 返回新图）。

**③ 抽象性说谎。** `ForceFieldTypifier` 类型上具体、事实上抽象：从不设 `self.atom_typifier`。
测试自己已经承认 —— `test_scope.py:326` 写的是
`ForceFieldTypifier.scope.fget(object.__new__(ForceFieldTypifier))`。

**④ 同一件事三份实现。** 「从端点原子类型给 bonded term 贴力场类型」写了三遍：

| 实现 | 位置 | 算法 | 匹配不上时 |
|------|------|------|-----------|
| `BondTypifier` | `typifier/bond.py`（死） | 通配 + class 回退 | strict 则报错 |
| `ForceFieldBondTypifier` | `typifier/atomistic.py`（活） | 特异性打分 + layer tiebreak | strict 则报错 |
| `Atomistic.assign_bonded_types` | **`core/atomistic.py:523`**（活） | `name.split("-")` 精确匹配 | **静默不贴标签** |

第三份住在 `core/` —— 力场判断跑进数据模型层，是三份里最粗的一份，且违反铁律 5。
`polymer_builders/peo_gel/build_gel.py:168` 正在用它。

**⑤ 层次泄漏。** `typifier/atomistic.py` 这个名字本身就是病：它把"原子体系"当成了
分型器的一个种类。

## 实测：三个必须先修的前提

**(a) `complete_valence()` 静默丢掉高阶 link。** 它只拷贝原子与键：

```
原图        : 4 atoms  3 bonds  2 angles  1 dihedrals
complete_() : 14 atoms 13 bonds 0 angles  0 dihedrals    ← 角与二面角全没了
```

所以"无条件补全"这一步，今天会把本要分型的高阶项直接抹掉。
**必须先让 `capping.complete_valence` 保留所有 link kind**（帽子只新增键，
不为帽子生成新的角/二面角 —— 它们只是上下文，不进写回集合）。
一个名为"返回补全后的分子"的函数静默丢拓扑，这是 `core/` 的既存 bug。

**(b) 没有泛型的 node 视图。** `Atomistic.atoms` / `CoarseGrain.beads`，两者都没有 `.nodes`。
泛型 `typify` 要逐 node 写回注解，必须给 `_GraphViews` 补一个 `nodes` 视图。
`links.classes()` 已经是泛型的（实测返回 `[Bond, Angle, Dihedral]`）。

**(c) `CoarseGrain` 没有 `complete_valence`。** 补一个返回等价拷贝的实现
（bead 没有价键可补）。多态在**图**上，而不是在 typifier 里 `if isinstance(...)`。

另有两个已确认的事实：

- **`complete_valence()` 对完整分子是恒等的**（实测：水 / 乙醇 / 苯 / 三甘醇 / 乙酸甲酯 /
  吡啶 / 二甲砜 / `[Na+]` 原子数全不变；芳香键级、砜的高价、单原子离子都没被误加氢）。
  所以第 ① 步无条件调用是安全的，不需要分支。
- **`Dihedral` 与 `Improper` 元数都是 4**，所以"按元数配对 link kind 与 FF type"行不通，
  必须有一张显式映射表。这张表就是 `ForceFieldParams` 存在的理由。

## 被否掉的五版设计

- **v1**：三个公开契约 `ElementTypifier` / `AtomTypifier` / `RegionTypifier` ——
  把内部零件提拔成公开面。
- **v2**：`Typifier` 带 `scope` + `typify` + `relaxed()`，下挂抽象 `ForceFieldTypifier` ——
  后者只是"持有四个 matcher"，那是组合不是继承。
- **v3**：`Typifier` 只剩抽象 `typify` —— 那样"补全→写回→补参数"要抄三遍。
- **v4**：流水线归位，但把参数标注塞进泛型 `typify` —— `Bond`/`Angle`/`Dihedral` 是原子体系的概念。
- **v5**：把 v4 的错误矫枉过正成"`base.py` 里不许出现 `Atomistic` 这个词"（一条愚蠢的文本禁令），
  又发明了一个 `ParamTypifier`（**没有这种东西**：分型器必须绑在某个力场或某个工具上）。
  **本 spec 修正这两点。**

## Design

```
typifier/
├── base.py            Typifier[G: MolGraph](ABC) + Match
│                      typify = 补全 → copy → match → write_onto
│                      唯一抽象方法：match(graph: G) -> Match
│                      基类只用 graph.nodes / graph.links.classes() / graph.complete_valence()
│
├── _matching.py       TypeClassIndex（私有）    打分：exact>class>wildcard + layer tiebreak
├── forcefield.py      ForceFieldParams          **零件，不是分型器**
│                        ├── _TermMatcher[L]     元数从 FF_TYPE 读出
│                        ├── _PairMatcher
│                        └── {Bond: BondType, Angle: AngleType,
│                             Dihedral: DihedralType, Improper: ImproperType}
│                            ← 全库唯一把 link kind 绑到力场类型的地方
│
├── clp.py             ClpTypifier(Typifier[Atomistic])         match: molrs SMARTS(CL&P)
├── ambertools.py      AmberToolsTypifier(Typifier[Atomistic])  match: antechamber
├── affected_region.py AffectedRegion.around(g, touched, reach=)
├── region.py          RegionTypes.of(region, typifier)
└── cache.py           RetypeCache

删  typifier/{base(旧),bond,angle,dihedral,pair,mmff,atomistic,scope}.py
```

`CGBondType` 出现的那天，往 `forcefield.py` 的映射表里加一行，流水线一个字不改。

### `ForceFieldParams` 取代 `core.assign_bonded_types`

它不是分型器，所以它没有 `typify`，也不进 `Typifier` 的继承树。它有两个入口：

- `match(graph, node_types)` —— 力场分型器的后半段，`ClpTypifier` / `AmberToolsTypifier` 各调一次；
- `assign(graph)` —— 原子类型已经在图上了，只标参数。`build_gel.py:168` 用这个，
  取代 `core.Atomistic.assign_bonded_types`。匹配不上时**报错**，不再静默跳过。

### `TypeScope` 溶解，`reach` 归装配器

```python
class AffectedRegion(Atomistic):
    TERM_REACH: ClassVar[int] = 2          # 二面角元数减二，不是魔数
    @classmethod
    def around(cls, graph, touched, *, reach: int) -> AffectedRegion:
        interior_reach = max(reach, cls.TERM_REACH)
        extract_radius = interior_reach + reach
```

`scope.py` 整个删掉，`Typifier` 不知道半径的存在。
`GraphAssembler(reaction, typifier=t, reach=2)`；`AmberToolsTypifier` 丢掉 `reach=`。

> **代价，记录在案。** `graph-assembler-01` 的结论是"半径由分型器决定，永远不是设置项"，
> 依据是实测：`reach` 取错时 `AmberToolsTypifier` 在一个 PEO junction 上把 46 个写回原子里的
> **22 个**分错。`reach` 变成装配器参数后，这道防线从类型系统降级到测试套件
> （`graph-assembler-01` ac-002 的 oracle 须按 reach 参数化跑）。
> 注：`AmberToolsTypifier(amber, reach=2)` 今天本就是用户传 reach，只是换了个位置。

### 区域分型不是一种分型器

`AffectedRegion` 本就 IS-A `Atomistic`，所以 `typifier.typify(region)` 天然合法，
流水线第 ① 步已经替它补全了。剩下的只是把 interior 的类型快照成可缓存的 `RegionTypes`：

```python
RegionTypes.of(region, typifier)   # = typifier.typify(region) + capture
```

`RegionTypifier` Protocol、`typify_region`、`retype_region`、`relaxed()` 全部删除。
`relaxed()` 的实现是**临时改写调用方持有的对象的 `strict` 字段**（还不是线程安全的）——
把错误关掉，而不是把问题解决。正解是别把残缺分子递给匹配器。

### 三处顺手修的既存 bug

1. **`capping.complete_valence` 静默丢掉高阶 link**（见上，铁律 5）。修：保留所有 link kind。
2. **`_capture_links` 不跳过 `type is None` 的 bonded term**：把 `None` 记进快照，
   `apply_to` 再写回母图，抹掉原有类型。修：跳过。**没判定出来的东西不该写回去。**
3. **SMARTS 路径的区域是裸切的**：两半径定理里 `extract_radius = interior_reach + reach`，
   interior 原子的感受野**恰好伸到 boundary 原子**；裸切时那里价键不满，
   在 SMARTS 眼里是自由基/卡宾 —— 而它是 interior 原子环境的一部分。
   今天只有 AmberTools 那条路补全。统一到第 ① 步后自动修好；ac-008 负责测量修好前的偏差。

### 删除清单

| 删除物 | 理由 |
|--------|------|
| `bond.py` `angle.py` `dihedral.py` `pair.py` `mmff.py` `atomistic.py` | 死代码 / 层次泄漏 |
| `scope.py`（`TypeScope`） | 单字段类；半径算术归造区域的人 |
| `ForceFieldTypifier` | 只为持有四个 matcher 的抽象层 → 零件 `ForceFieldParams` |
| `core.Atomistic.assign_bonded_types` | 第三份 bonded 实现，在错误的层，静默失败 |
| `RegionTypifier` / `typify_region` / `retype_region` | 区域分型是自由算法 |
| `relaxed()` / `_typify_relaxed` | 由无条件补全取代 |
| `atomtype_matches`（两份）、`skip_atom_typing` | 0 调用者 |
| 公开的 `PairTypifier` | 是 `ForceFieldParams` 的零件 |

## 实施期的一处修正：补全归"切图的人"，不归 typifier

v6 写的是"流水线第 ① 步无条件补全"，即 `Typifier.typify` 里调 `graph.complete_valence()`。
**实施时被 CL&P 打脸，而且它是对的。**

`tests/test_typifier/test_clp.py` 的分子是从 z-matrix 建的**纯连接图**：没有坐标，没有键级，
但化学上完整。无条件补全会去给芳香咪唑鎓碳加氢（`KeyError: 'x'`，因为它连坐标都没有）。

根因不是 `complete_valence` 不够好，而是**"这个图是不是被截断的"是身份问题，不是数值问题**。
自由基是一个完全合法的分子；一个没有键级的连接图处处看起来价键不满。按铁律 5
（可以猜数值，不能猜身份），流水线**不许**从价键去猜 provenance。

**知道自己被切过的是区域，不是分型器。**所以：

```python
class Typifier[G](ABC):
    def typify(self, graph: G) -> G:      # 不补全，照单全收
        typed = graph.copy()
        self.match(graph).write_onto(typed)
        return typed

class RegionTypes:
    @classmethod
    def of(cls, region, typifier):
        capped = region.complete_valence()   # 区域总是一次切割 ⟹ 总是补全
        typed = typifier.typify(capped)
        return cls.capture(region, typed, before, after)
```

分支数仍然是零：`RegionTypes.of` 无条件补全（区域必然是切出来的），`typify` 从不补全。
"所有截断的部分都要补全"依然成立 —— 由知道存在截断的那一方来补。
`AmberToolsTypifier` 因此也不再自己 `complete_valence`。

**ac-008 的实测（本次落地时取得）：** OPLS-AA，`reach=2`

| 体系 | 补全：错/拒绝 | 裸切：错/拒绝 |
|------|--------------|--------------|
| PEO `COCCOC` | 0 / 0 | 0 / 0 |
| 对二甲苯 | 0 / 0 | 0 / **12**（共 19 个切片） |
| 丙烯酸甲酯 | 0 / 0 | 0 / 0 |

裸切芳香环时，分型器**根本分不出来**（截断的芳香碳配出 `opls_927`，角找不到键型），
而不是静默分错。脂肪链的裸切碰巧没事 —— 这正是为什么补全必须是无条件的一步，
而不能是 AmberTools 那条路的私有便利。

## Files

```
新增  src/molpy/typifier/base.py         Typifier[G: MolGraph](ABC) + Match        ~90
新增  src/molpy/typifier/_matching.py    TypeClassIndex（私有）                    ~110
新增  src/molpy/typifier/forcefield.py   ForceFieldParams + 映射表（零件，公开）    ~210
改    src/molpy/typifier/clp.py          ClpTypifier(Typifier[Atomistic])，只写 match
改    src/molpy/typifier/ambertools.py   AmberToolsTypifier(...)，只写 match；丢掉 reach=
改    src/molpy/typifier/affected_region.py  around(g, touched, reach=)；TERM_REACH
改    src/molpy/typifier/region.py       RegionTypes.of()；_capture_links 跳过未分型
改    src/molpy/typifier/cache.py        RetypeCache 经 RegionTypes.of
改    src/molpy/typifier/__init__.py     导出 Typifier / Match / ForceFieldParams /
                                         ClpTypifier / AmberToolsTypifier / molrs 两个 re-export
删    src/molpy/typifier/{base,bond,angle,dihedral,pair,mmff,atomistic,scope}.py

改    src/molpy/core/capping.py          complete_valence 保留所有 link kind（既存 bug）
改    src/molpy/core/entity.py           _GraphViews.nodes 泛型 node 视图
改    src/molpy/core/cg.py               CoarseGrain.complete_valence() -> 等价拷贝
改    src/molpy/core/atomistic.py        删 assign_bonded_types
改    src/molpy/builder/assembly/_assembler.py  收 reach=；isinstance(t, Typifier)
改    polymer_builders/peo_gel/build_gel.py     assign_bonded_types → ForceFieldParams(net_ff).assign(gel)

改名  tests/test_typifier/test_atomistic.py → test_forcefield.py
改名  tests/test_typifier/test_scope.py    → test_region_radii.py
新增  tests/test_typifier/test_matching.py  打分规则（承接 atomtype_matches 覆盖）
新增  tests/test_typifier/test_contract.py  一条流水线、泛型不假定原子体系、fake CG
新增  tests/test_core/test_capping.py       保留高阶 link + 对完整分子恒等
改    tests/test_io/test_forcefield/test_xml.py   import 改指 typifier.forcefield
改    docs/api/typifier.md                 指向已删的 molpy.typifier.mmff / .atomistic
```

## Tasks

- [ ] 写失败测试：`test_capping.py`（补全保留角/二面角；对完整分子恒等）
- [ ] `core/capping.py`：`complete_valence` 保留所有 link kind
- [ ] `core/entity.py`：`_GraphViews.nodes`；`core/cg.py`：`CoarseGrain.complete_valence`
- [ ] 写失败测试：`test_contract.py`（`Typifier.__abstractmethods__ == {"match"}`；子类 `__dict__` 无 `typify`；20 行 fake CG 分型器只写 `match` 即可接入且不改 `base.py`）+ `test_matching.py`
- [ ] `base.py`：`Typifier[G: MolGraph](ABC)` + `Match`；三步流水线
- [ ] `_matching.py` / `forcefield.py`：`TypeClassIndex`、`ForceFieldParams`、映射表；元数从 `FF_TYPE` 读出；力场只扫一次
- [ ] 删 `core.Atomistic.assign_bonded_types`；`build_gel.py` 改用 `ForceFieldParams.assign`
- [ ] `clp.py` / `ambertools.py`：各自只剩一个 `match`
- [ ] `affected_region.py`：`around(graph, touched, reach=)`；删 `scope.py`
- [ ] `region.py`：`RegionTypes.of`；`_capture_links` 跳过 `type is None`
- [ ] `_assembler.py`：`GraphAssembler(rxn, typifier=t, reach=2)`；`isinstance(t, Typifier)`
- [ ] **测量**（ac-008）：补全 vs 裸切，各自与全图 oracle 比对
- [ ] 删死模块；改 import / 导出 / 测试 / `docs/api/typifier.md`
- [ ] 全量 `ruff` + `ty` + `pytest -m "not external"`；跑 `build_gel.py` 对齐数字

## Testing

- **泛型不假定原子体系**（判据不是文本禁令）：`Typifier` 的类型参数绑定到 `MolGraph`；
  `base.py` 只调用 `graph.nodes` / `graph.links.classes()` / `graph.complete_valence()` / `graph.copy()`；
  `Bond→BondType` 映射全库只在 `forcefield.py` 出现一次；不存在 `typifier/atomistic.py`。
- **一条流水线**：`Typifier.__abstractmethods__ == {"match"}`；`ClpTypifier` / `AmberToolsTypifier`
  的 `__dict__` 里都**没有** `typify`。`Typifier()` 直接构造 `TypeError` 且信息含 `abstract`。
- **补全**：保留角/二面角/improper（今天丢）；对八个完整分子恒等；`typify` 里对
  `complete_valence()` 的调用不在任何 `if` 分支内；`CoarseGrain.complete_valence()` 返回等价拷贝。
- **CG 可扩展性（本设计的证明）**：20 行 `_FakeCGTypifier(Typifier[CoarseGrain])` 只写 `match`。
  必须**无需改动 `base.py`** 即可被 `RetypeCache` 消费、被 `GraphAssembler` 接受。
- **`ForceFieldParams` 不是分型器**：`not issubclass(ForceFieldParams, Typifier)`；它没有 `typify`。
  `ForceFieldParams(ff).assign(g)` 贴出的 bonded `type` 是旧 `assign_bonded_types` 的**超集**
  （后者按 `name.split("-")` 精确匹配，前者带通配/class/layer）；匹配不上的 term 现在报错。
- **补全 vs 裸切（科学，ac-008）**：切口落在 sp3 碳上的体系，比较补全与裸切各自对全图 oracle
  的逐原子偏差。两种结果都写进 evidence。
- **半径算术**：`AffectedRegion.around(g, touched, reach=r)`，`r ∈ {1,2,3}`：
  `interior == ball(touched, max(r,2))`、`extract_radius == max(r,2)+r`；球外原子逐原子未被写。
- **打分**：exact > class > wildcard；某一位配不上 → `None`；正反两向同分；
  `X-CT-CT-X` 严格小于全解析模式；元数不匹配 → `ValueError`。`TypeClassIndex` 只扫一次力场。
- **非空洞性**：拿掉 layer tiebreak，`test_clp.py` 必须变红。
- **回归**：`tests/test_typifier/` + `tests/test_core/test_capping.py` +
  `tests/test_builder/test_assembly.py` + `test_io/test_forcefield/test_xml.py` 全绿；
  `build_gel.py` 数字对齐。

## Out of scope

- **OPLS 重接。** `OplsTypifier` 自 OPLS 下沉 molrs 后就不存在，`docs/user-guide/05/06` 因此跑不起来。
  那是 `opls-typifier-downsink`。本 spec 只把它将来要落进来的位置留出来 —— 它就是一个 `match`。
- 打分常数与 `reach = 2` 这个数本身。
- 真正的 `CoarseGrainTypifier` 与粗粒化力场类型（`CGBondType` 今天不存在）；
  只用一个 fake 证明流水线不排斥它。
- `compute/` 的 ~40 个 `self._inner = molrs.X(...)` 转发壳。
