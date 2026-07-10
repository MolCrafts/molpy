---
title: GraphAssembler — 唯一装配内核 + Selector 策略族
status: in-progress
created: 2026-07-10
depends_on: "graph-assembler-01-reach"
supersedes: "graph-assembler.md §Design;crosslink-03 记录的 'PolymerBuilder = 退化交联' follow-up"
---

# 一个内核,一族选择器

> chain graph-assembler 2/4。**不依赖任何 molrs 新 API**。
> 本 spec 把 `PolymerBuilder` 与 `Crosslinker` **收敛到一个内核**(`PolymerBuilder` 保留为其子类,
> `Crosslinker` 三个类消失),并消除 O(N²)。
> 删 `builder/crosslink/` 与 `builder/polymer/core.py`;`reacter/` 留给 03。

## Summary

`PolymerBuilder` 与 `Crosslinker` 在做同一件事:

```
粘图 → 匹配位点 → 配对 → 就地局域图编辑 → 受影响区域局域补丁
```

其中**只有"配对"不同**:CGSmiles 邻接给出一张显式边表;交联在 cutoff 内搜索并按策略采样。
`DeterministicCrosslinker` 今天已经支持 `pairs=[(a, b)]` 显式配对 —— `PolymerBuilder` 的配对
就是一张从 CGSmiles 邻接导出的显式边表。**它们不是两个 builder,是一个内核 + 两种选择规则。**

本 spec 把内核收敛成一个**具体类** `GraphAssembler`,把配对提为策略族 `Selector`,
并顺带解决五个更深的问题(见 Domain basis)。

## Domain basis

### 1. 合并之后只剩一个方法不同

spec 02 的早期草案保留了 `GraphAssembler(ABC)` + `_paste` / `select` / `place` 三个 hook,
两个子类。逐一检查,其中两个 hook 是假差别:

**`place` 不是类的差别,是构造器参数。** 它回答的是"这两个组分的相对坐标有没有意义":
熔体已经排布过,不能动;单体副本叠在原点,必须摆。**交联两个悬空分子时同样需要摆放。**
所以它是 `placer: Placer | None`,不是一个子类。按类切分会造成组合爆炸
(想要"随机选点 + 摆放"就得再开一个类)。

**`_paste` 不是 hook,是一条隐藏信道。** `PolymerBuilder` 的 `select` 必须知道"哪些原子属于
第 i 个 CGSmiles 节点"。今天它靠往原子上写魔法字符串字段 `monomer_node_id`
(`polymer/core.py:379`)传递这个信息,再用 `cleanup_build_markers`(`:64`)擦掉。
两个 hook 之间除了原子标记就只剩可变实例状态,两条都不可接受(铁律 1 / 铁律 4)。

正解:**世界是输入,不是 hook 的产物。** 单体展开是一个真实的类:

```python
world = MonomerLibrary({"EO": eo}).expand(topology)
```

它给每个粘上去的副本打上**正则字段** `RES_ID`(= CGSmiles 节点 id)与 `RES_NAME`(= 单体标签)。
一个单体**就是**一个残基 —— 这两个字段本来就在 `molrs.fields` 正则表里,无需发明新字段。
于是隐藏信道消失、`monomer_node_id` 消失、`cleanup_build_markers` 消失,
而且残基身份从"事后要擦掉的垃圾"变成 **PDB / prmtop 导出真正需要的输出**。

剩下的唯一差别是配对。它与 `typifier` / `placer` 正交 ⟹ **策略对象**,不是子类。

**但"实现上可以合并"不等于"API 上必须拆开"。** 早期草案让用户写:

```python
topology = parse_cgsmiles("{[#EO]|1000}").base_graph      # .base_graph 是 IR 内部属性
world = MonomerLibrary({"EO": eo}).expand(topology)
chain = GraphAssembler(ether, TopologySelector(topology), ...).assemble(world)
```

这是坏的,而且它自己在报警:同一个 `topology` **穿过两个对象传了两次**。
同一份数据被线穿两遍,说明边界切错了。用户也不该知道 `CGSmilesIR.base_graph` 存在。

`PolymerBuilder` **不是**工作流门面(对照被删的自由函数 `crosslink_gel`:它接过已经造好的对象
按顺序调一遍,自己不携带数据、不做决定)。`PolymerBuilder` **拥有单体库,拥有 CGSmiles 的语义**
—— 它是真类。铁律 4 禁的是空壳,不是"拥有数据并做翻译"的类。

### 2. selector 属于调用,不属于构造

CGSmiles 拓扑只有 `build(cgsmiles)` 时才知道,所以选择规则**依赖于本次调用的输入**,
不可能在构造器里定。把 selector 提到 `assemble(world, selector)` 的实参:

- `PolymerBuilder.build` 每次调用现造一个 `TopologySelector(topology)`;
- 交联时 `assemble(melt, RandomSelector(...))`;
- **`RegionPatchCache` 与 selector 无关**(它只依赖 typifier 与区域结构),
  于是一个 `PolymerBuilder` 建 100 条**不同长度**的链共享同一份缓存。
  若 selector 在构造器里,每条拓扑就得新建装配器,缓存随之作废。

### 3. 匹配是内核的事,配对是选择器的事

`Crosslinker._all_candidates` 是 `O(|occ_a| · |occ_b|)`。一条 N 单体的链有 ~2N 个 site,
若内核替所有选择器构造候选对,`TopologySelector` 会白付 ~4N² 次配对,ac-018 的线性斜率必挂。

但**匹配**(`pattern.find_matches`)是 O(N),而且人人都要。切法是:

- 内核做一次匹配,把 `occurrences: list[list[Binding]]`(每个反应物组分一张表)交给选择器;
- 选择器负责**配对**,自己决定要不要建 `Candidate`。

`TopologySelector` 按 `RES_ID` 索引 occurrence,直接查表,零配对;
`ProximitySelector` 才用 `NeighborQuery` 建候选对。基类不认识 `Candidate`。

### 4. 原 PolymerBuilder 的 O(N²) 有四个独立来源

只删掉深拷贝是不够的:

| 位置 | 每条键的代价 |
|---|---|
| `reacter/base.py:394` `merged = left.copy()` | 深拷贝已累积的整条链 |
| `polymer/core.py:501` `_build_entity_map` | 合并 `entity_maps` → 尺寸 O(已装配原子数) |
| `polymer/core.py:557` `_preserve_node_ids` | 遍历整个 `entity_map` |
| `polymer/core.py:346` `_remap_ports_registry(affected=members[gid])` | 遍历该 group 全部成员节点 |

根因是同一个:**用"拷贝 + entity_map 重映射"表达图编辑**。粘图一次、就地编辑、句柄稳定 ⟹
`entity_map` 整套机制消失,四个 O(N) 项一起消失。

### 5. 区域必须在**所有** apply 之后建;补丁缓存必须活在实例上

现状 `Crosslinker.apply` 在循环体内建区域:第 `i` 次编辑抽出的区域,会被第 `j > i` 次触及共享
原子的编辑**变旧**,而它的类型此后才写回。这是一个静默的正确性 bug。

先 apply 完、再建区域,则每个区域都看终态图。两个区域重叠时,共享的 interior 原子 `a` 满足
`hops ≤ interior_reach`,故其 `reach`-球完整落在**两个**区域内 → 两个补丁给出同一个类型。
重叠写回幂等。

代价:所有 binding 必须在编辑前一次选定 ⟹ **两两不相交**(occurrence 原子集无交)。
线性链、支化、星形、环闭合天然满足;`RandomSelector` 已有 `exclude_same_match`。基类**断言**它。

**`RegionPatchCache` 必须是实例属性,不是 `assemble()` 里的局部变量。**
`PolymerSystem` 建 100 条链会调 100 次 `assemble`;缓存若每次新建,EO–EO 结界要分型 100 次,
ac-005("miss 数与链数无关")直接失败。缓存是**跨调用的记忆化**,挂在装配器上,docstring 注明。

### 6. 补丁拥有哪些键项

只有与 `touched` 关联的键**发生了变化**,所以只有**含至少一条 touched-incident 键**的键项是
新生/湮灭的。它们的原子全部落在 `ball(touched, TERM_REACH)` ⊆ interior(01 §4)。

```
owned(region) = {含至少一条 touched-incident 键的键项}
```

`apply_to` 分两趟(重叠区域可能共同拥有一条二面角,两趟保证幂等):
先对全部区域从父图删除 `owned(region)`,再对全部区域插入 `patch.terms`。

**这暴露一个 molrs 契约。** `refresh=False` 跳过全图重建后,`Reaction.apply` 删除原子时**必须**
一并删除以该原子为端点的角/二面角/improper。现状被随后的全图 `generate_topology` 遮住了。
T1 先验证;不成立 → 提 molrs issue,**在 molrs 修**,不在 molpy 兜底。

### 7. 现存的静默失败 —— 其中一条是真 bug

```python
# _crosslinker.py:257  _find_component —— map_number 不属于任何反应物组分时…
return 0                      # 静默把位点当成在第 0 个组分上

# _crosslinker.py:45   _total_charge —— 未分型图没有 charge 列时…
sum(a.get("charge", 0.0) or 0.0 for a in graph.atoms)   # 静默视作 0,守恒检查失效

# _crosslinker.py:346  _pair_distance —— 图没有坐标时…
return 0.0                    # 所有候选距离相等,距离排序静默失效
```

第一条是**现存 bug**:畸形 SMIRKS(形成键的 map number 没出现在任何反应物模式里)不报错,
而是被当作"位点在第 0 个组分上",随后拿错 occurrence 列表配对,产出化学上无意义的交联。
应 `raise ValueError` 并点名该 map number。

第三条:`cutoff is None` 时按拓扑配对是合法模式;但 `Candidate.distance` 不应是 `0.0` 这个
**看起来像真距离的假值**,而应是 `None`,且排序在 `None` 上走确定性拓扑序。

### 8. 电荷守恒是定理,不是启发式

`_conserve_leaving_charge` 把离去基团的电荷**均摊**到 `touched` 锚点。它在补一个真实的洞
(AM1-BCC 是全分子解),但均摊目标任意,且随装配次数累积误差。

正解在**模板层**,一次性:单体模板是一个**加帽的真实分子**,非局域电荷方法在它上面合法。
求解后把**每个帽的总电荷折叠到它所加帽的那个 site 原子**(目标唯一 —— 帽与 site 原子之间只有
一条键)。折叠后帽原子电荷恒为 0。装配时删除电荷为 0 的原子 ⟹ **净电荷严格守恒**,无需事后修正。

`GraphAssembler` 因此**不接受 `charges=` 参数**。它只断言:若图有 `charge` 列,则被反应删除的
原子电荷为 0。守恒从"每次编辑后修正"变成"构造时就不可能违反"。

## Design

> 遵守 `.claude/notes/architecture.md` §设计铁律。

### molrs 不出现在用户代码里(铁律 6)

`molrs.Reaction` 今天**没有**从 `molpy` 导出。本 spec 补齐 re-export:

```python
# molpy/__init__.py
from molrs import Reaction, SmartsPattern, NeighborQuery, Graph, perceive_aromaticity, find_rings
```

**re-export,不是包装层**:`molpy.Reaction is molrs.Reaction` 为真,无新类、无转发、无 `_inner`。

### 内核:一个具体类

```python
class GraphAssembler:
    """世界 + 选择器 → 匹配 → 配对 → 局域图编辑 → 局域补丁。

    唯一的装配内核。它不是 ABC —— 变化点是 ``assemble`` 的 ``selector`` 实参。
    直接用它就是交联;``PolymerBuilder`` 在它之上加"单体库 + CGSmiles"。
    """

    def __init__(
        self,
        reaction: Reaction,
        *,
        typifier: LocalTypifier | None = None,
        placer: Placer | None = None,
        label_field: FieldSpec = fields.SITE,
    ) -> None:
        if typifier is not None and not isinstance(typifier, LocalTypifier):
            raise TypeError(...)                  # 无退化到全图分型的分支
        #: 跨 assemble() 调用的记忆化。只依赖 typifier 与区域结构,与 selector 无关
        #: —— 所以一个 builder 建 100 条不同拓扑的链共享同一份缓存。
        self._cache = RegionPatchCache(typifier)

    def assemble(self, world: Atomistic, selector: Selector) -> Atomistic:
        work = world.copy()                                   # 输入永不被改
        labels = self._labels(work)                           # 读 label_field,O(N),一次
        occurrences = self._match(work, labels)               # O(N),一次,人人都要
        bindings = list(selector.select(work, occurrences))
        self._assert_disjoint(bindings)
        self._assert_leaving_charges_zero(work, bindings)
        if self._placer is not None:
            self._placer.place(work, bindings)

        touched = [self._reaction.apply(work, b, labels, refresh=False) for b in bindings]
        regions = [self._scope.region(work, t) for t in touched]   # 终态图,apply 全部完成后

        patches = [self._cache.patch(r) for r in regions]
        for region in regions:                                # 两趟,保证重叠幂等
            region.drop_owned_terms(work)
        for region, patch in zip(regions, patches, strict=True):
            patch.apply_to(region)
        return work
```

交联**不需要第二个类** —— 它就是内核加一个邻近选择器:

```python
gel = GraphAssembler(ether, typifier=gaff).assemble(
    melt, RandomSelector(conversion=0.8, cutoff=6.0, seed=1)
)
```

`Crosslinker` / `DeterministicCrosslinker` / `RandomCrosslinker` 三个类因此消失:
它们除了持有一个 selector 并转发 `apply` → `assemble`,不携带任何数据、不做任何决定。
**那才是铁律 4 禁止的门面。**

### 选择器族:唯一的变化点

```python
class Selector(ABC):
    """从匹配到的位点里选出要反应的绑定 —— 装配器唯一的多态点。"""

    @abstractmethod
    def select(
        self, world: Atomistic, occurrences: list[list[Binding]]
    ) -> Iterator[Binding]:
        """产出 {map_number: handle};必须两两不相交。"""
```

| 选择器 | 配对规则 | 取代 |
|---|---|---|
| `TopologySelector(cgsmiles_ir)` | 按 `RES_ID` 索引 occurrence,CGSmiles 每条边一个绑定。零配对 | `PolymerBuilder` |
| `ProximitySelector(ABC)`(`cutoff`) | `NeighborQuery` 建候选对;抽象 `choose(candidates)` | `Crosslinker` |
| `ExhaustiveSelector` | 全部候选 | `DeterministicCrosslinker(...)` |
| `SpacingSelector(spacing)` | 主链上每 `spacing` 个位点 | `DeterministicCrosslinker(spacing=)` |
| `ExplicitPairSelector(pairs)` | 指定 `(handle, handle)` | `DeterministicCrosslinker(pairs=)` |
| `RandomSelector(conversion, seed, ...)` | 打乱候选,消耗到目标转化率 | `RandomCrosslinker` |

拆开 `DeterministicCrosslinker` 的三个**互斥 kwarg**(`spacing` / `pairs` / 默认穷尽)
是附带收益:非法状态不可表示,不再有"三个都传了谁赢"的静默优先级。

`TopologySelector` 在同一残基里找到多个同名 site 时 **raise**(铁律 5),不静默取第一个。

### `MonomerLibrary` 与 `PolymerBuilder`:两个真类

```python
class MonomerLibrary:
    """单体模板库。构造时校验模板,展开时粘图并打残基标记。"""

    def __init__(self, templates: Mapping[str, Atomistic]) -> None:
        """校验:每个模板至少一个 ``SITE`` 原子;若有 ``charge`` 列,帽电荷必须已 freeze
        (为 0)。不合格 → raise,不静默(铁律 5)。"""

    def expand(self, topology: CGSmilesGraphIR) -> Atomistic:
        """每个节点粘一份模板副本,打上 ``RES_ID`` = 节点 id、``RES_NAME`` = 标签。

        不做几何、不做反应。O(Σ|template|)。
        """


class PolymerBuilder(GraphAssembler):
    """从 CGSmiles + 单体库长出聚合物。

    IS-A 装配器:它**继承**内核,只额外拥有单体库与 CGSmiles 语义。
    不是门面 —— 它携带数据(库)并做翻译(记号 → 世界 + 配对),
    对照被删的自由函数 ``crosslink_gel``:后者接过已造好的对象按序调一遍,自己什么也不拥有。
    """

    def __init__(
        self,
        library: MonomerLibrary,
        reaction: Reaction,
        *,
        typifier: LocalTypifier | None = None,
        placer: Placer | None = Placer(),      # 模板副本必然重叠 —— 默认摆放
    ) -> None: ...

    def build(self, cgsmiles: str) -> Atomistic:
        """展开记号,把相邻残基的 site 键合起来。

        ``CGSmilesIR.base_graph`` 是内部 IR,用户永不接触。
        """
        topology = parse_cgsmiles(cgsmiles).base_graph
        world = self._library.expand(topology)
        return self.assemble(world, TopologySelector(topology))
```

用户面:

```python
chain = PolymerBuilder(MonomerLibrary({"EO": eo}), ether, typifier=gaff).build("{[#EO]|1000}")
```

`RES_ID` / `RES_NAME` 是 `molrs.fields` 的**正则字段**,不是新发明:一个单体就是一个残基。
`monomer_node_id` / `port` / `port_descriptor_id` 三个魔法字符串字段与
`cleanup_build_markers` 一并消失。

> `build` 是第二个动词吗?不是同族的第二个动词 —— 它吃的是**记号**,不是世界。
> `build = expand + assemble`。`assemble` 仍是唯一的"世界 → 世界"动词。

### 重复单元 = 带 site 列的图

不新增 `RepeatUnit` / `Junction` / `Port` 类。**只标需要标的原子** —— `SITE` 是稀疏标注:

```python
import molpy as mp                                 # 用户代码里没有 molrs
from molpy.core import fields
from molpy.parser import parse_smiles, smilesir_to_atomistic
from molpy.builder import PolymerBuilder, MonomerLibrary

eo = smilesir_to_atomistic(parse_smiles("OCCO"))   # 加帽的真实分子
eo.atoms[0][fields.SITE] = "a"                     # 只标要反应的那两个氧
eo.atoms[3][fields.SITE] = "b"
Am1BccCharges(net_charge=0).freeze(eo)             # 模板上求解,帽电荷折叠归零
gaff.typify(eo)

ether = mp.Reaction("[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]")
chain = PolymerBuilder(MonomerLibrary({"EO": eo}), ether, typifier=gaff).build("{[#EO]|1000}")
```

同一个内核,换一个选择器就是交联 —— 交联不需要自己的类:

```python
from molpy.builder import GraphAssembler, RandomSelector

gel = GraphAssembler(ether, typifier=gaff).assemble(
    melt, RandomSelector(conversion=0.8, cutoff=6.0, seed=1)
)
```

`eo[...]` 是图属性表;原子视图是 `eo.atoms`,`eo.atoms[i]` 返回 `Atom`。
`_labels(world)` 只收 `SITE` 非空的原子 —— 空串是"未标记"的存储表示,不是 site 名;
`SITE` 列整体缺失才是 `KeyError`。

### 模式 vs 回退(铁律 5)

`typifier=None`(纯拓扑装配)与 `placer=None`(坐标已就位)都是**有名字的模式**,不是回退:
它们只关掉自己那一步,**不改变任何别的行为** —— 尤其不跳过 disjoint 断言与零电荷断言。

对照现状 `_make_retype_cache` 的 `hasattr(typifier, "typify_region")` 分支:同一个 `typifier=`
参数在两条语义不同的路径间静默切换。02 之后非 `LocalTypifier` 在**构造时** `TypeError`。

边界情形:

- `select` 产出 0 个 binding → `warnings.warn`(带候选数与 cutoff),返回未改动的世界
- binding 相交 → `raise ValueError`,点名共享 handle
- 图有 `charge` 列但待删原子电荷非 0 → `raise ValueError`,点名原子与电荷值
- 图无 `charge` 列 → 跳过电荷检查(显式模式,`_assert_leaving_charges_zero` 首行注明)

**芳香性**:模板在构造时感知一次(每个单体*类型*一次);粘图原样拷贝;每条结界只在自己的球内
感知。装配路径上零全图 `perceive_aromaticity`、零全图 `generate_topology`。

`AmberPolymerBuilder` **不是** assembler:它是导出器(单体 → prep,边 → tleap `bond`),
由 tleap 装配。它是 03 的科学基准。

## Files

**新增 `src/molpy/builder/assembly/`**

- `_assembler.py` — `GraphAssembler`(具体类;`assemble(world, selector)`)
- `_polymer.py` — `PolymerBuilder(GraphAssembler)` + `MonomerLibrary`
- `_selector.py` — `Selector(ABC)`
- `_topology.py` — `TopologySelector`
- `_proximity.py` — `ProximitySelector(ABC)` / `ExhaustiveSelector` / `SpacingSelector` /
  `ExplicitPairSelector`;`Candidate` 只在这里
- `_random.py` — `RandomSelector`
- `_placer.py` — `Placer`(自 `polymer/placer.py` 迁入,消费 `fields.SITE`,删 `_get_port_direction`)

**新增(typifier 层)**

- `src/molpy/typifier/patch.py` — `RegionPatch` / `TermSpec` / `RegionPatchCache`
- `src/molpy/typifier/base.py` — `Typifier(ABC)` / `LocalTypifier` / `ChargeModel(ABC)`

> 不新开 `molpy/assembly/` 顶层包:装配器是 builder,区域补丁是 typifier 的产物。

**改**

- `src/molpy/__init__.py` — re-export `Reaction` / `SmartsPattern` / `NeighborQuery` / `Graph` /
  `perceive_aromaticity` / `find_rings`(铁律 6)
- `src/molpy/core/fields.py` — 注册 molpy 自有 `SITE = FieldSpec(...)`(不改 molrs 正则表)
- `src/molpy/typifier/cache.py` — `RetypeCache` → `RegionPatchCache`(补 terms)
- `src/molpy/typifier/ambertools.py` — 抽出 `Am1BccCharges(ChargeModel)`;
  docstring 去掉"charge 由 reacter 折叠到 anchor"
- `src/molpy/builder/polymer/system.py`、`distributions.py`、`sequences.py` — 改用 assembly/

**删**

- `src/molpy/builder/crosslink/`(整包:`_crosslinker.py` / `_deterministic.py` / `_random.py`)
- `src/molpy/builder/polymer/core.py`(`PolymerBuilder` / `PolymerBuildResult` /
  `TypifierProtocol` / `get_ports_on_node` / `cleanup_build_markers` / `_cleanup_stale_ports`)
- `src/molpy/builder/polymer/placer.py`(迁入 assembly/)

> `builder/crosslink/recipes.py`、`builder/polymer/connectors.py`、`presets.py`、`reacter/`
> 由 03 删除(它们还被 03 之前的文档/测试引用)。

## Tasks

- [x] T1 **先验证 molrs 契约**:`Reaction.apply(refresh=False)` 删原子时是否删除其关联
      角/二面角/improper。**已验证 (2026-07-10)**:C0-C1-C2-O3 删 O3 后 dihedral 1→0、
      angle 2→1、零悬挂项 ⟹ **契约成立,本 spec 不阻塞**
- [x] T2 **铁律 6 · re-export 审计**:`molpy/__init__.py` 导出 `Reaction` / `SmartsPattern` /
      `NeighborQuery` / `Graph` / `perceive_aromaticity` / `find_rings`;
      `molpy.Reaction is molrs.Reaction` 为真(commit 00c8422)。
      ⚠️ 审计发现 `src/molpy/compute/` 有 ~40 处 `self._inner = molrs.X(...)` 转发门面
      (distribution / spectra / order / voronoi / density …)。它们继承 molpy `Compute`
      协议并提供 `__call__`,介于适配器与门面之间 —— **超出本 spec 范围,须单独裁决**;
      ac-021 的反向检查因此收窄到本 spec 触及的文件
- [~] T3 `fields.SITE` 已注册(molpy 自有 FieldSpec);`Entity` 的下标/`get` 接受 FieldSpec
      (commit 00c8422)。**未完**:`site_field=` / `monomer_node_id` / `"port"` 字符串仍在
      `builder/crosslink/` 与 `builder/polymer/core.py`,随 T14 一并删除
- [ ] T4 `Typifier` / `LocalTypifier` / `ChargeModel`;`ChargeModel.freeze()` 折叠帽电荷
- [x] T5 **简化**:`RegionPatch(types + terms)` 未实现,因为 T1 探针证明 molrs 在删原子时
      **已经**删掉关联键项 ⟹ 补丁只需 **insert**,不需要 delete。类型走既有 `RetypeCache`
      (结构哈希 + is_isomorphic),键项由 `GraphAssembler._insert_new_terms` 从区域直接生成
      (含跨区域幂等 `inserted` 去重)。少一个类,同样的保证
- [x] T6 **不需要**:同上。只有"含新形成键"的键项是新生的,它们在编辑前不可能存在,
      所以没有要删除的东西。`_spans_bond` 判定一个键项是否含新键,`inserted` 集合保证
      两个重叠区域共同拥有的二面角只插一次
- [x] T7 `Selector(ABC)`;`TopologySelector`(按 `RES_ID` 查表,同名 site 多值 → raise)
- [x] T8 `ProximitySelector(ABC)` + `Exhaustive` / `Spacing` / `ExplicitPair` / `Random`;
      `Candidate` 下沉进 `_proximity.py`
- [x] T9 `MonomerLibrary`:构造时校验模板(≥1 个 SITE 原子;有 charge 列则帽电荷已归零);
      `expand()` 打 `RES_ID` / `RES_NAME`
- [x] T10`GraphAssembler`(具体类):`assemble(world, selector)` —— selector 是**实参**不是构造器参数;
      匹配一次 / disjoint 断言 / 零电荷断言 / placer / apply-all / 建区域 / 两趟打补丁;
      **`RegionPatchCache` 为实例属性**;构造时 `isinstance` 拒非 `LocalTypifier`
- [x] T10b `PolymerBuilder(GraphAssembler)`:持 `MonomerLibrary`,`build(cgsmiles)` 内部
      `parse_cgsmiles(...).base_graph` → `expand` → `assemble(world, TopologySelector(topo))`;
      `.base_graph` 不出现在用户代码里
- [ ] T11 `Placer` 迁入 assembly/,消费 `fields.SITE`;成环边用 `topo_distances` 判环长;
      共价半径初值改具名常量 + docstring(铁律 5 例外)
- [ ] T12 `Am1BccCharges(ChargeModel)`;`AmberToolsTypifier` 只管类型
- [x] T13 **铁律 5**:`_find_component` 找不到 map_number → `raise ValueError`(**修现存 bug**);
      `Candidate.distance` 无坐标时为 `None`;删 `_make_retype_cache` 的 hasattr 回退;
      `select` 产出 0 binding → `warnings.warn`
- [ ] T14 删 `builder/crosslink/`、`builder/polymer/core.py`、`polymer/placer.py`;
      重接 `system.py` / `distributions.py` / `sequences.py`

## Testing

### 科学基准:AmberTools tleap(`@pytest.mark.external`)

同一 CGSmiles + 同一单体库,两条独立路径:

- **A(被测)** `PolymerBuilder(lib, ether, typifier=AmberToolsTypifier(..., reach=2)).build(cgsmiles)` → prmtop
- **B(参考)** `AmberPolymerBuilder`(antechamber → prepgen → tleap `sequence`)→ prmtop

| 量 | 判据 |
|---|---|
| GAFF 原子类型 | 逐原子相等 |
| bond / angle / dihedral / improper **项集合** | 相等(捕获 owned-terms 的增删漏) |
| 每单体电荷和 | `atol = 1e-6` |
| 体系净电荷 | `== Σ 模板净电荷`,`atol = 1e-9` |
| `RES_ID` / `RES_NAME` | 与 tleap 的残基划分一致 |

> 键项**参数**不单列判据:类型对上 + 同一张表 ⟹ 参数必然对上,那是重言式。
> 单列的是**项集合**,因为 owned-terms 的删除/插入是新代码,会漏项或重项。

体系:PEO(短单体,两 site 球重叠)、聚丙烯酸酯(sp2 羰基 → 钉 improper)、含 4-site 交联剂的星形。
**第二套力场用 OPLS-AA,基准是 `OPLSAATypifier` 的全图分型**(小体系)—— 全图分型是类型的*定义*。

### 合并的可执行证明

- `TopologySelector` 与 `RandomSelector` 喂给**同一个** `GraphAssembler` 实例的 `assemble`,
  线性链与交联各跑一遍,内核代码路径逐行相同(以覆盖率断言)
- `ExplicitPairSelector(pairs=CGSmiles 边)` 与 `TopologySelector(topo)` 在同一 world 上
  产出**相同的 binding 集合** —— 证明"PolymerBuilder 是显式配对交联"这一论断

### 局域性契约

- 非 `LocalTypifier` 传入 → 构造时 `TypeError`(不是装配时,不是静默退化)
- monkeypatch 使 `typifier.typify(整图)` raise:线性链 N=100、支化、4-site 星形、环闭合、
  交联全部通过 ⟹ 全图分型调用次数为 0
- monkeypatch 使 `generate_topology` / `perceive_aromaticity` 在图规模 > 最大单体规模时 raise
- 相交 binding → raise,点名共享 handle

### 电荷

- `freeze(template)` 后帽原子 `charge == 0`;`sum(charge) == 形式电荷`
- 未 freeze 的模板喂进 `assemble` → raise,不静默修正
- **删掉 `_conserve_leaving_charge` 之后净电荷仍守恒** —— 证明守恒来自模板层折叠

### 缓存与复杂度

- `RegionPatchCache` 是**实例属性**:同一个 `GraphAssembler` 连续 `assemble` 100 条 N=50 的链,
  miss 数与链数**无关**(每次新建缓存则必然失败)
- 两个 interior 同构、上下文壳层不同的区域 → 不得互相命中
- 短单体(两 site 球重叠)与长单体在同一缓存下都给出与 tleap 一致的类型
- `bm-molrs-molpy/`:线性链 N ∈ {10, 100, 1000},`log t` vs `log N` 斜率 ∈ [0.85, 1.15]

> 复杂度的正确表述是 **`O(|world| + E · |ball(2·reach)|)`**,不是"全程没有 O(N) 遍历" ——
> 粘图、`_labels`、匹配各是一次 O(N)。要求的是**每条边的代价与 N 无关**。

## Out of scope

- 删 `reacter/` / `connectors.py` / `presets.py` / `recipes.py`,文档与测试迁移 → 03
- `reach` 由模式集导出、无界谓词构造期报错 → 04
