---
title: typifier 包的架构与继承关系整理
status: draft
created: 2026-07-10
depends_on: "graph-assembler-01/02/03 (code-complete)"
blocks: "opls-typifier-downsink (它要落的那个 AtomTypifier 缝，本 spec 先凿出来)"
---

# 一个 typifier 有三种尺度，不是一个 `TypifierBase`

> 本 spec 只整理 `src/molpy/typifier/` 的**类层级与文件边界**。
> 不碰分型算法的数值行为（打分规则、reach=2、两半径定理都原样保留）。
> 不做 OPLS 下沉后的重接（那是 `opls-typifier-downsink`，阻塞在 molrs 发版）。

## Summary

`molpy/typifier/` 现在是 13 个文件、1756 行，其中 **6 个文件、约 340 行是死代码**，
而且死代码和活代码在同一个包里**用同样的名字定义了三对不同的东西**：

| 名字 | 定义 1（死） | 定义 2（活） |
|------|-------------|-------------|
| `TypifierBase` | `base.py:9`（非泛型，带 `strict`） | `atomistic.py:126`（泛型 `[T]`，无 `strict`） |
| `PairTypifier` | `pair.py:9` | `atomistic.py:322` |
| `atomtype_matches` | `base.py:21` | `atomistic.py:96` |

包外**没有任何模块**导入 `base/bond/angle/dihedral/pair/mmff` —— 它们只被彼此导入。
这是一次没做完的重构留下的旧布局：早期是"每种元素一个文件"，后来被
`atomistic.py` 合并接管，旧文件从没删。

在这层噪音底下还有三个真实的结构问题：

1. **`ForceFieldTypifier` 类型上是具体类，事实上是抽象类。** 它从不设置
   `self.atom_typifier`（只有子类 `ClpTypifier` 设），所以用默认参数构造它，再调
   `typify()`，会在深处炸 `AttributeError`。测试自己已经承认了这一点 ——
   `tests/test_typifier/test_scope.py:326` 写的是
   `ForceFieldTypifier.scope.fget(object.__new__(ForceFieldTypifier))`，
   一个绕开构造器的 workaround。
2. **三个 bonded typifier 是同一个算法抄了三遍**，只有元数不同；每个都各自把同一份
   `ForceField` 扫一遍建 `type→class` / `class→layer` 两张表。
3. **`typify` 这个动词在同一个基类下有两种互斥的语义**：
   `ForceFieldBondTypifier.typify(bond)` 原地改并返回同一个对象；
   `ForceFieldTypifier.typify(struct)` 返回一个**新** `Atomistic`。
   `TypifierBase[T]` 同时是这两者的父类，于是它什么也没约束住。

## Domain basis

分型发生在**三个尺度**上，这不是实现细节，是问题本身的结构：

**元素尺度。** 一根键、一个角、一个二面角、一个原子的非键参数。输入是一个已经知道
两端原子类型的元素，输出是给它贴上力场类型和参数。它就地改（和 core 数据模型的其余部分
一致），返回自己。

**结构尺度的原子分型。** 给整个结构的每个原子定类型。这是 SMARTS 匹配，而
**SMARTS 匹配器属于 molrs**（铁律 2：molrs 回答语法/结构事实，molpy 做力场/化学判断）。
所以 molpy 里不存在一个通用的原子分型器实现，只存在**适配器**：把 molrs 的匹配结果写回
molpy 结构。今天唯一的实例是 `clp.py::_ClpAtomTypifier`；`opls-typifier-downsink`
落地时会出现第二个。

这个尺度还需要一个 `strict` 开关，而且这个开关必须在**契约里**，因为
`_typify_relaxed` 现在直接伸手去改 `self.atom_typifier.strict` —— 一个没有任何类型
声明过的属性。区域的最外壳原子上下文被截断，**本来就该分不出类型**，所以给区域分型时
必须临时把它关掉。

**区域尺度。** 一次图编辑扰动的那个球。它要声明自己看多远（`TypeScope`），
而这一个数同时定死抽取半径和写回半径（graph-assembler-01 的两半径定理）。

一个 `TypifierBase` 描述不了这三件事。**三个契约。**

## 现状的证据

```
$ wc -l src/molpy/typifier/*.py                      # 13 文件 1756 行
$ grep -rn "typifier.(base|bond|angle|dihedral|pair|mmff)" src tests examples
    → 包外零命中；六个模块只被彼此导入
$ grep -rn "atomtype_matches(" src/                  → 0 处生产调用（只有测试）
$ grep -rn "retype_region" src tests                 → 0 处调用（死方法）
$ grep -rn "ForceFieldTypifier(" src tests examples  → 0 处直接构造
$ grep -rn "skip_atom_typing" src/                   → 只有 clp.py 读它
```

`mmff.py` 整个文件是 `from molrs.typifier import MMFFTypifier` —— 和 `__init__.py`
第一行重复。`atomistic.py` 还 import 了一个从不使用的 `ImproperType`。

## Design

### 目标：三个契约 + 一条真实的继承链

```
╔═════════════════════════ protocol.py：三个契约 ═════════════════════════╗
║                                                                          ║
║  ElementTypifier[T](ABC)     AtomTypifier(ABC)        RegionTypifier      ║
║  ─────────────────────      ─────────────────        (Protocol,结构型)   ║
║  typify(elem: T) -> T        typify(struct)->struct   scope -> TypeScope  ║
║  就地改，返回自己             返回新结构                typify_region(r)     ║
║  .ff  .strict                .strict                    -> RegionTypes    ║
╚══════╤══════════════════════════════╤═══════════════════════╤═══════════╝
       │                              │                       │
       │ 4 个实现                      │ molrs SMARTS 的缝      │ 2 个实现
       ▼                              ▼                       │ (无共同祖先)
 ForceFieldBondedTypifier[L]    _ClpAtomTypifier ──┐          │
   ├─ ForceFieldBondTypifier      (clp.py)         │          ├──► ForceFieldTypifier
   ├─ ForceFieldAngleTypifier                      │          │      (ABC, forcefield.py)
   └─ ForceFieldDihedralTypifier                   │          │           △
 PairTypifier                                      │          │           │
       ▲                                           │          │      ClpTypifier
       │ 拥有 4 个                                  │          │
       └──── ForceFieldTypifier ◄──────────────────┘          └──► AmberToolsTypifier
                 abstract property                                   (antechamber, reach=)
                 atom_typifier -> AtomTypifier
```

三个要点：

**`RegionTypifier` 保持结构型 Protocol，不改成基类。** 它的两个实现共享零行代码、零祖先：
一个读力场自己的模式，一个 shell out 给 antechamber。让它们继承 Protocol 会白送一个
`scope` 属性 —— 子类忘了覆盖时它静默返回 `None`。那正是铁律 5 禁止的静默兜底。
`@runtime_checkable` 的 `isinstance` 仍然照常工作（`GraphAssembler` 就是这么拒绝无界
typifier 的），测试替身也仍然只要长得像就行。

**`ForceFieldTypifier` 变成真 ABC，`atom_typifier` 变成抽象属性。**
于是 Python 在构造时就拒绝它，不需要我手写的那个 `TypeError` 守卫
（那个守卫是上一轮改动里贴的创可贴，本 spec 把它连同病因一起拿掉）。
抽象的理由是**领域的**，不是偶然的：力场的**参数**是 molpy 的判断，力场的**原子类型**
是 molrs 的计算。编排器提供前者，要求子类指名后者。

**三个 bonded typifier 收敛成一个泛型基类。** 元数从 `FF_TYPE` 上读出来
（`BondType` 有 `itom/jtom`，`AngleType` 多一个 `ktom`，`DihedralType` 多一个 `ltom`，
`hasattr` 在 pyo3 类上正确返回 False），所以元数**不可能**和它匹配的力场类型不一致。
三个具体类各自只剩两行：`TERM` 和 `FF_TYPE`。

### `TypeClassIndex`：一次扫描，三处共享

`_build_type_class_layer` / `_end_score` / `_sequence_score` 三个模块级函数合并成
`matching.py::TypeClassIndex`：`class_of(type)` / `layer_of(pattern)` / `score(pattern, atoms)`。
它替 `ForceFieldBondedTypifier` 的构造器扫一次 `ForceField`，而不是每个 bonded typifier
各扫一次同一份力场。打分规则（exact type 3 > class 1 > wildcard 0，正反两向取优）
**逐字保留**，只是搬了家。

### 删除清单，以及为什么可以删

| 删除物 | 理由 | 铁律 |
|--------|------|------|
| `base.py` `bond.py` `angle.py` `dihedral.py` `mmff.py` | 死代码，且遮蔽活名字 | 无死代码 |
| 旧 `pair.py` 内容 | 被 `atomistic.PairTypifier` 取代（新 `pair.py` 复用文件名） | 同上 |
| `atomtype_matches`（两份） | 0 处生产调用；语义并入 `TypeClassIndex._component_score`，测试同步迁移到 `test_matching.py` | 同上 |
| `ForceFieldTypifier.retype_region` | 0 处调用；`RetypeCache.retype_and_apply` 是唯一入口 | 同上 |
| `skip_atom_typing` | 0 处使用；有了抽象 `atom_typifier` 之后它自相矛盾（"必须提供一个你永不使用的原子分型器"） | 3（实验期不背兼容） |
| 手写的 `TypeError` 守卫 | 由 ABC 在构造时接管 | 5（快失败，但让语言去做） |

破坏性变更全部落在**没有任何调用者**的符号上，`ClpTypifier`、`AmberToolsTypifier`、
`GraphAssembler` 的公开行为逐字不变。

## Files

```
新增  src/molpy/typifier/protocol.py      三个契约            ~100
新增  src/molpy/typifier/matching.py      TypeClassIndex      ~110
新增  src/molpy/typifier/bonded.py        泛型 bonded 基类+3   ~130
新增  src/molpy/typifier/forcefield.py    ForceFieldTypifier   ~165
重写  src/molpy/typifier/pair.py          PairTypifier          ~50
改    src/molpy/typifier/clp.py           接上 AtomTypifier
改    src/molpy/typifier/region.py        移出 RegionTypifier，只剩数据
改    src/molpy/typifier/cache.py         import 改指 protocol
改    src/molpy/typifier/ambertools.py    import + 陈旧 docstring（Crosslinker/reacter）
改    src/molpy/typifier/__init__.py      导出契约 + 编排器 + 元素分型器
改    src/molpy/builder/assembly/_assembler.py, _polymer.py   import 改指 protocol
删    src/molpy/typifier/{base,bond,angle,dihedral,mmff,atomistic}.py

改名  tests/test_typifier/test_atomistic.py → test_bonded.py（去掉私有内部断言）
新增  tests/test_typifier/test_matching.py   打分规则（承接 atomtype_matches 的覆盖）
新增  tests/test_typifier/test_protocol.py   契约、抽象性、无重名
改    tests/test_typifier/test_scope.py      去掉 object.__new__ workaround
改    tests/test_io/test_forcefield/test_xml.py  import 改指 bonded
改    docs/api/typifier.md                   指向已删的 molpy.typifier.mmff
```

## Tasks

- [ ] 写失败测试：`test_protocol.py`（ABC 不可构造 / 两个 RegionTypifier 无共同祖先 / 死模块已消失）+ `test_matching.py`
- [ ] `protocol.py`：`ElementTypifier[T]`、`AtomTypifier`、`RegionTypifier`（从 `region.py` 迁出）
- [ ] `matching.py`：`TypeClassIndex` 吃掉三个模块级打分函数
- [ ] `bonded.py`：`ForceFieldBondedTypifier[L]` + 三个两行子类
- [ ] `pair.py` 重写为 `ElementTypifier[Atom]`
- [ ] `forcefield.py`：`ForceFieldTypifier(ABC)` + 抽象 `atom_typifier`；删 `skip_atom_typing`、`retype_region`、手写守卫
- [ ] `clp.py`：`_ClpAtomTypifier(AtomTypifier)`；`ClpTypifier` 实现 `atom_typifier` 属性
- [ ] 删六个死模块；改所有 import；改 `__init__.py` 导出
- [ ] 迁移测试；`docs/api/typifier.md`
- [ ] 全量 `ruff` + `ty` + `pytest -m "not external"`；跑一遍 `polymer_builders/peo_gel/build_gel.py`（`AmberToolsTypifier` 是 `RegionTypifier` 的另一个实现，必须不受影响）

## Testing

- 契约层：`ForceFieldTypifier` / 忘记指名 `atom_typifier` 的子类都必须 `TypeError`；
  指名了的可以构造。`ClpTypifier` 与 `AmberToolsTypifier` 都 `isinstance(..., RegionTypifier)`
  且互相**不是**子类。缺 `scope` 的东西不是 `RegionTypifier`。
- 打分层：exact > class > wildcard；某一位配不上则整条模式作废；正反两向等价；
  `X-CT-CT-X` 必须输给全解析模式（这是 OPLS 二面角能工作的原因）；元数不匹配直接报错。
- 行为回归：`tests/test_typifier/` 全绿；`tests/test_builder/test_assembly.py` 全绿
  （`RetypeCache` + `TypeScope` 是本次改动的下游）；`test_io/test_forcefield/test_xml.py`
  里那条 `ForceFieldBondTypifier` 断言全绿。
- 非空洞性：删掉 `bonded.py` 里的 layer tiebreak，CL&P 的 `test_clp.py` 必须变红。

## Out of scope

- **OPLS 重接。** `OplsTypifier` 这个名字自 OPLS 下沉 molrs 后就不存在了，
  `docs/user-guide/05/06` 因此跑不起来。那是 `opls-typifier-downsink`，阻塞在 molrs 发版。
  本 spec 只把它将来要落进来的那个 `AtomTypifier` 缝凿出来，不去填。
- 分型算法的任何数值行为：打分常数、`REACH = 2`、两半径定理。
- `compute/` 的 ~40 个 `self._inner = molrs.X(...)` 转发壳（用户已明确单独讨论）。
- `affected_region.py` / `scope.py` / `cache.py` / `region.py` 的数据结构本身。
