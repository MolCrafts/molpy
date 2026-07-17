---
title: 局域分型的感受野 —— 单一 reach 取代 context_radius 魔数
status: code-complete
created: 2026-07-10
depends_on: "incremental-typify-01/02/03 (done)"
supersedes: "graph-assembler.md §Domain basis 1-2、§Design(TypeScope 部分)"
---

# reach —— 一个分型器只有一个感受野半径

> chain graph-assembler 1/4。**不依赖任何 molrs 新 API**,可以立刻落地。
> 本 spec 只修科学正确性:区域重分型抽多远、写回哪些原子。
> 不删代码、不改类层级 —— 那是 02/03。

## Summary

区域重分型有两个半径:**抽取球半径**和**写回集合半径**。现状把它们揉成一个
`context_radius`(`ForceFieldTypifier` 声明 4,`AmberToolsTypifier` 声明 2,
`_FLOOR` 兜底 4),结果两个都不对。

但**它们不是两个独立的自由度**。它们是同一个数 `reach` 和它的两倍。把 `reach` 拆成
`(retype_reach, context_reach)` 两个可独立设置的字段,会引入一个不存在的自由度,并且掩盖
环谓词把 `reach` 抬高这件事。

本 spec:

1. 定义 `TypeScope(reach)` —— 一个数,一个类,是**唯一**知道半径算术的地方;
2. 把写回集合从 "non-boundary"(= `ball(touched, R-1)`)改成 `ball(touched, interior_reach)`,
   由每个原子到 `touched` 的跳数判定,不再由"有没有球外邻居"判定;
3. `RegionTypes` 补 `impropers`;
4. 删 `_FLOOR` / `region_radius()` / 三处 `context_radius`。

## Domain basis

### 1. 感受野是一个数

设分型器判定原子 `a` 的类型只看 `a` 的 `reach`-球(即 `ball(a, reach)` 这张带标签的诱导子图)。

- **写回集合。** 编辑发生在 `touched`。`a` 的类型可能改变 ⟺ 编辑落进 `a` 的判定邻域
  ⟺ `touched ∩ ball(a, reach) ≠ ∅` ⟺ `dist(a, touched) ≤ reach`。
  距离对称,所以需要重分型的集合恰是 `ball(touched, reach)`。
- **抽取球半径。** `ball(touched, reach)` 里每个原子还各自要看自己的 `reach`-球,
  最远的原子在 `reach` 跳外,它的球再伸 `reach` 跳 → 抽取半径 `2 · reach`。

`retype_reach` 与 `context_reach` **恒等**,因为二者都是"`a` 的类型由多大的球决定"这同一个量,
只是一次从 `touched` 看出去、一次从 `a` 看出去。写成两个可独立赋值的字段是引入伪自由度。

### 2. 现状为什么错(可证)

现状只有一个 `radius = R`:抽取 `ball(touched, R)`,`boundary` = 最外壳(有球外邻居的原子),
写回**所有非 boundary 原子** = `ball(touched, R-1)`。于是最外那圈被写回的原子,距 `touched`
`R-1` 跳,它自己的球只剩 `R - (R-1) = 1` 层上下文。

正确性要求同时满足:

```
写回集合覆盖真实变更集:   R - 1 ≥ reach
最外写回原子上下文够用:   R - (R-1) ≥ reach   ⟹   reach ≤ 1
```

两式同时成立仅当 `reach = 1, R = 2`。**任何 `reach ≥ 2` 的分型器在现状下必然写回错类型。**

实测:

| 分型器 | 声明 | 抽取 | 写回集合 | 最外写回原子的上下文 | 真实 reach |
|---|---|---|---|---|---|
| `ForceFieldTypifier` (OPLS/GAFF SMARTS) | `context_radius = 4` | `ball(4)` | `ball(3)` | 1 层 | ≥ 2(芳环需 3) |
| `AmberToolsTypifier` | `context_radius = 2` | `ball(2)` | `ball(1)` | 1 层 | 2(其自身 docstring 写 "1–2 bond") |
| 无声明 | `_FLOOR = 4` | `ball(4)` | `ball(3)` | 1 层 | — |

`region.py:180` 那条"interior 未分型就 raise"的守卫抓不住:上下文残缺的 SMARTS 更可能**匹配到一条
错的规则**,而不是匹配不到。

`AffectedRegion.interior` 被 `_from` 设成 `centers`(只有 touched),而真正的写回集合由
`typify_region` 用 `boundary` 补集算出。**`interior` 字段今天是死的,且与实际行为不符。**

### 3. 环谓词把 reach 抬高,不是另开一个半径

`[r6]`(在六元环上)这类**带尺寸**的环谓词是局域的:含 `a` 的 `N` 元环,其所有原子沿环距 `a`
最多 `⌊N/2⌋` 跳,整个环落在 `ball(a, ⌊N/2⌋)` 里。芳香性感知还要看环上原子的**环外**取代基
(Hückel 计数里 exocyclic C=O 贡献 0 电子),再加一层。

```
reach = max(max_bond_depth, ⌊max_ring_system_size / 2⌋ + 1)
```

注意是**环系(ring system)**尺寸,不是 SSSR 单环尺寸:稠环体系的芳香性是整个稠环系的性质。

这条把环谓词并进 `reach`,而不是像旧 spec 那样只并进 `context_reach` —— 后者会漏掉**成环编辑**:
合上一根键新生成一个 `N` 元环,环上距 `touched` `⌊N/2⌋` 跳的原子其 `[r6]` / 芳香标志翻转。
若 `retype_reach = max_bond_depth < ⌊N/2⌋`,这些原子不会被重分型,类型停在旧值。

**无界谓词**(使 `reach` 不存在)不止 `[R]` / `[!R]`:任何不带尺寸的环性质都需要全图 SSSR ——
`[R]`、`[!R]`、`[R0]`、`[R2]`(环数)、`[x2]`(环键数)。带尺寸的 `[r3]`…`[r8]` 才是局域的。
本 spec 只**记录**这条契约;把它变成编译期检查是 04(需要 molrs)。

### 4. 新键项把 interior 抬到至少 2

新键 `(u,v)` 引出的键项里,原子离 `{u,v}` 最远的是**以 `(u,v)` 为端边的二面角** `u-v-b-c`:
`c` 在 2 跳外。一般地,arity `k` 的路径型键项含边 `(u,v)` 时,其原子距 `{u,v}` ≤ `k - 2`。
molpy 的 `Atomistic` 最高 arity 是 4(dihedral / improper):

```
TERM_REACH = 4 - 2 = 2          # 不是魔数:是二面角的元数减二
interior_reach = max(reach, TERM_REACH)
extract_radius = interior_reach + reach
```

这些键项的端点必须**已被分型**才能查表,所以它们必须落在写回集合里 —— 这就是 `TERM_REACH`
抬高 `interior_reach`(而不是抬高 `extract_radius`)的原因。

### 5. `touched` 的契约

`interior = ball(touched, interior_reach)` 只在 `touched` 覆盖**每一条形成/断裂键的存活端点**
时才是完备的。

证:设存活原子 `a` 的球里有一个被删除的原子 `f`,取最短路 `a → f`,沿路回退到第一个非删除原子
`q`;`q` 与某个删除原子相邻,故 `q` 是 anchor ∈ `touched`,且 `dist(a,q) < dist(a,f) ≤ reach`。
所以 `a ∈ ball(touched, reach)`。形成键同理。∎

该证明依赖 `touched ⊇ anchors`。今天这只是对 `molrs.Reaction.apply` 的**假设**。本 spec 把它
变成 `AffectedRegion._from` 里的一条断言。

## Design

> 遵守 `.claude/notes/architecture.md` §设计铁律:零硬编码字段、体系判断留 molpy、
> 可 breaking、真 OOP(无模块级 `def`,但也不造命名空间类)。

```python
# molpy/typifier/scope.py
@dataclass(frozen=True)
class TypeScope:
    """一个分型器的感受野。整个 molpy 里唯一做半径算术的地方。"""

    #: 二面角 / improper 的元数减二 —— 新键引出的键项,其原子距新键最多这么远。
    #: 不是魔数:是 ``Atomistic`` 最高键项元数(4)减二的推论,证明见 spec §4。
    TERM_REACH: ClassVar[int] = 2

    reach: int
    """判定一个原子的类型所需的邻域半径(键跳数)。

    ``max(SMARTS 键深, ⌊最大环系尺寸/2⌋ + 1)``。见 spec §3。
    """

    @property
    def interior_reach(self) -> int:
        """写回集合的半径:类型可能改变、或新键项需要其类型的原子。"""
        return max(self.reach, self.TERM_REACH)

    @property
    def extract_radius(self) -> int:
        """抽取球半径:写回集合里最远的原子还要看满自己的 reach-球。"""
        return self.interior_reach + self.reach

    def region(self, graph: Atomistic, touched: Iterable[Atom | int]) -> AffectedRegion:
        """抽出该分型器所需的区域。``interior`` = ball(touched, interior_reach)。"""
```

### 自由函数 → 方法(铁律 4)

本 spec 触及的模块必须做到模块级 `def` 计数为 0。逐条去向:

| 现状自由函数 | 去向 |
|---|---|
| `core/affected_region.py:region_radius()` | 删(被 `TypeScope` 取代) |
| `core/affected_region.py:_resolve_centers()` | `AffectedRegion._resolve_centers()` classmethod |
| `typifier/region.py:typify_region()` | `LocalTypifier.typify_region()` 方法 |
| `typifier/region.py:apply_region_types()` | `RegionTypes.apply_to(region)` 方法 |
| `typifier/region.py:_typify_nonstrict()` | `LocalTypifier._typify_nonstrict()` 方法 |
| `typifier/region.py:_as_param()` / `_atom_info()` | `TypeInfo.from_data()` classmethod |
| `typifier/region.py:_link_info()` | `BondedTypeInfo.from_link()` classmethod |
| `typifier/region.py:_scalar_delta()` / `_capture_links()` | `RegionTypes` 内部 classmethod |

**不要**为了消灭自由函数而新建 `class RegionUtils:` 之类的命名空间壳 —— 上表每一项都落在
一个**已经拥有该数据**的类上。

`AffectedRegion` 随之改两处:

- `_from` 记录每个区域原子到 `touched` 的**跳数**(BFS 已经在算,顺手带出来),
  `interior = tuple(a for a in atoms if hops[a] <= interior_reach)`;
- `boundary` 保留,但降级为**文档概念**(上下文壳层 = `hops > interior_reach`)。
  写回不再用 `not in boundary` 判定 —— 那正是 bug 的来源。

`typify_region` / `apply_region_types` 的写回过滤器从 `handle in boundary` 换成
`hops[handle] <= interior_reach`。`RegionTypes` 增 `impropers: tuple[BondedTypeInfo, ...]`。

分型器侧:

| 类 | `scope` |
|---|---|
| `ForceFieldTypifier` | `TypeScope(reach=3)` —— 实测最小通过值(ac-003),docstring 记录扫描结果 |
| `AmberToolsTypifier` | 构造器 **`reach: int` 必填**,内部包成 `TypeScope`。antechamber 是黑盒,感受野由用户按 ac-003 实测声明 |

`RegionTypifier` 协议的 `context_radius: int` 换成 `scope: TypeScope`。

**`TypeScope` 不进用户代码。** 它是内部类型:装配器与区域路径之间的契约。用户面上只有一个
整数 `reach`,且只在**无法被 molpy 读取的黑盒分型器**上出现:

```python
gaff = AmberToolsTypifier(AmberTools(), reach=2)   # 不是 scope=TypeScope(reach=2)
opls = OPLSAATypifier()                            # 自己导出,不收 reach
```

理由:用户不需要理解"抽取球是 `2·reach`、写回球是 `reach`"这套算术 —— 那是实现。
他们只需要回答一个可实测的化学问题:"这个力场判定一个原子的类型要看几根键?"
把 `TypeScope` 塞进构造器签名等于把内部算法漏进公开面。

### 字段(铁律 1)

`typify_region` 现在用 `data.get("type")` 这样的字符串字面量读写。全部换成
`molpy.core.fields` 的 `FieldSpec`:`fields.TYPE.key`。本 spec 不引入新字段
(`SITE` 由 02 引入)。缺字段 fail-fast,不 `getattr(..., default)`。

### 静默失败(铁律 5)

本 spec 触及的三处静默回退必须一起消除 —— 它们正是让"写回错类型"这个 bug 活了这么久的原因:

| 位置 | 现状 | 改成 |
|---|---|---|
| `affected_region.py:49` | `int(getattr(typifier, "context_radius", 0) or 0)`,为 0 就用 `_FLOOR = 4` | 分型器**必须**有 `scope`;没有就不是 `LocalTypifier`,`TypeError` |
| `region.py:205` | `strict = bool(getattr(getattr(typifier, "atom_typifier", None), "strict", False))` | **删掉这个开关**。interior 原子未分型**永远** raise,与 typifier 的 strict 无关 |
| `region.py:253` | `atom_typifier is None` → 直接 `typifier.typify(region)`,strict 语义静默丢失 | 同上 |

第二行是重点。`region.py:180` 那条"interior 未分型就 raise"的守卫**只在 `strict` 为真时生效**,
而 `strict` 来自一个带默认值 `False` 的双层 `getattr` —— 任何没有 `atom_typifier` 属性的分型器,
守卫永不触发。**守卫依赖一个可选属性,等于没有守卫。**

正确语义:boundary(上下文壳层)原子未分型是**预期**(它们只提供环境,本就不写回);
interior 原子未分型是**契约违反**,无条件 raise,错误信息给出 canonical position 与当前
`extract_radius`。两者由 hops 区分,不由 `strict` 区分。

## Files

- 新增 `src/molpy/typifier/scope.py` — `TypeScope`(含 `TERM_REACH` ClassVar)
- 改 `src/molpy/core/affected_region.py` — 删 `_FLOOR` / `region_radius()`;`_resolve_centers`
  → classmethod;`_from` 带出 hops;`interior` 语义修正
- 改 `src/molpy/core/__init__.py` — 不再导出 `region_radius`
- 改 `src/molpy/typifier/region.py` — 8 个模块级 `def` 全部下沉为方法(见上表);
  写回过滤器换成 hops;`RegionTypes` 补 `impropers`;`RegionTypifier.context_radius` → `scope`;
  字段读写走 `fields.TYPE`
- 改 `src/molpy/typifier/atomistic.py` — `context_radius` → `scope`,删 `return 4`
- 改 `src/molpy/typifier/ambertools.py` — `scope` 必填,删 `_DEFAULT_CONTEXT_RADIUS`;
  修 docstring 里"charge 由 reacter 折叠到 anchor"的说法(那是 02 要删的启发式)
- 改 `src/molpy/builder/crosslink/_crosslinker.py` — `region_radius(...)` → `self._scope.region(...)`
- 改 `src/molpy/reacter/base.py:534` — 同上(reacter 本身在 03 才删)

## Tasks

- [x] T1 `TypeScope`(`TERM_REACH` 为 ClassVar,非模块常量),含 `region()`;单测覆盖 ac-004 的算术
- [x] T2 `AffectedRegion._from` 带出 per-atom hops;`interior` = `ball(touched, interior_reach)`;
      `touched ⊇ anchors` 断言(ac-006);`_resolve_centers` → classmethod
- [x] T3 `RegionTypes.impropers`;`typify_region` / `apply_region_types` 等 8 个自由函数下沉为方法;
      写回过滤器改用 hops
- [x] T4 `ForceFieldTypifier.scope` / `AmberToolsTypifier.scope`;删三处 `context_radius`、
      `_FLOOR`、`region_radius`
- [x] T5 `typifier/region.py` 内的字段字面量换成 `fields.TYPE`
- [x] T6 **铁律 5**:删 `strict` 开关与两处 `getattr(..., default)`;interior 未分型无条件 raise;
      boundary 未分型是预期,由 hops 区分
- [x] T7 reach 扫描测试(ac-003),把实测值写进 docstring
- [x] T8 更新 `tests/test_core/test_affected_region.py`、`tests/test_typifier/test_ambertools_typifier.py`
      的断言;每条改动在 commit body 里注明旧断言错在哪
- [x] T9 Hygiene:`/mol:simplify` 跑通 —— 删掉无外部调用者的 `RetypeCache.apply`(内联进
      `retype_and_apply`);无其他死代码

## Testing

**Oracle 是全图分型,不是旧代码。** 一个原子的类型的**定义**就是全图分型器赋给它的类型;
区域分型是一个声称等价的优化。所以 ac-002 用 `typifier.typify(g.copy())` 做逐原子基准 —— 这不
违反"不拿旧实现当参考",因为被测的是*区域路径*,基准是*定义路径*。

GAFF 与 tleap 的端到端对拍留给 02(那里才有 `GraphAssembler`)。

- ac-002 的芳环体系(对二甲苯)是关键:`reach=3` 时通过,`reach=2` 时苯环 `[cA]` 退化 →
  证明 §3 的 `⌊N/2⌋ + 1` 不是装饰。
- reach 扫描(ac-003)同时给出**性能上界**:`extract_radius = 2·reach`,球大小随 reach
  近指数增长,所以"实测最小值"既是正确性也是性能约束。

## Out of scope

- `GraphAssembler` / `RegionPatch` / 删 `reacter/` → 02、03
- 从 molrs 编译好的模式集**导出** `reach`、无界谓词编译期报错 → 04(需 molrs)
- `ChargeModel` / `_conserve_leaving_charge` → 02
