# 粗粒化结构

`CoarseGrain` 是把 `Atomistic` 中的原子换成珠子（bead）的结构——工厂方法、变换操作、组合方式都跟 `Atomistic` 一致，不施加任何映射约定。

## 粗粒化结构与全原子结构是同一类对象

在 MolPy 中，`CoarseGrain` 之于珠子系统，正如同 `Atomistic` 之于全原子系统。一个珠子可以对应细粒度模型中的一组原子（Martini、MARTINI 3、VOTCA 风格的映射），也可以完全没有全原子前身（DPD、平衡后的 Martini 快照、等待反向映射的聚合物支架）。

`CoarseGrain` 与 `Atomistic` 接口完全一致——同样的工厂方法、空间操作、组合运算符、基于字典的属性访问。会构建全原子系统，就会构建粗粒化系统。

```python
import molpy as mp

cg = mp.CoarseGrain(name="lipid")
b1 = cg.def_bead(type="P4", x=0.0, y=0.0, z=0.0)
b2 = cg.def_bead(type="C1", x=4.7, y=0.0, z=0.0)
cg.def_cgbond(b1, b2, k=120.0)

cg.move([1, 0, 0])
print(b1["x"])   # 1.0
```

## 珠子可以携带任意自定义字段

`Bead` 是类字典对象，没有强制字段。位置、质量、电荷、类型标签、原子来源——每个字段跟 `Atom` 一样，可以设置也可以省略。没有力场特定的模式，也不对珠子位置与其代表原子之间的关系做任何内置假设。

这种不加约束本身就是设计意图。Martini 3 珠子取组成重原子（含氢）的几何中心，Martini 2 珠子取质量加权中心，VOTCA 风格的珠子用任意逐原子权重，DPD 珠子直接存位置、背后没有原子。MolPy 不替用户选约定——每种约定在自己的场景下都是对的。

```python
# Martini 风格的珠子：类型标签 + 位置
cg.def_bead(type="P4", x=1.0, y=2.0, z=3.0)

# 记住其所代表原子的珠子
ato = mp.Atomistic()
c1 = ato.def_atom(element="C", x=0.0, y=0.0, z=0.0)
c2 = ato.def_atom(element="C", x=1.5, y=0.0, z=0.0)
cg.def_bead(atoms=(c1, c2), type="CG_C2")

# 使用 vermouth 风格残基模板键的珠子
cg.def_bead(template="ALA_BB", residue_id=12)
```

这些布局没有一种"正确"或"推荐"的。数据结构只管记录放进去的内容。

## 只对一个约定键提供内置支持

核心数据结构只认一个约定键：`bead["atoms"]`。有这个键，就表示珠子代表的 `Atom` 引用元组。这个约定跟 `entity["x/y/z"]` 存在的理由一样——让空间混入（spatial mixin）有东西可操作。`move(delta)` 依赖 `x/y/z`，反向查找方法 `beads_of(atom)` 依赖 `atoms`。

```python
ato = mp.Atomistic()
a = ato.def_atom(element="C")
b = ato.def_atom(element="C")
c = ato.def_atom(element="O")

bead_ab = cg.def_bead(atoms=(a, b), type="CC")
bead_c  = cg.def_bead(atoms=(c,),     type="O")

cg.beads_of(a)   # (bead_ab,)
cg.beads_of(c)   # (bead_c,)
cg.beads_of(b)   # (bead_ab,)
```

这是线性扫描；涉及大量原子且频繁调用的场景，建议自己构建 `id(atom) → list[Bead]` 索引。数据结构有意不做缓存，否则缓存失效会跟 `CoarseGrain` 上的每个工厂方法耦合在一起。

如果映射存在重叠，`beads_of` 返回多个珠子；原子未被任何珠子引用时返回空元组。

```python
shared = cg.def_bead(atoms=(a,), type="virtual")
cg.beads_of(a)   # (bead_ab, shared)
```

共享原子在实际的力场中确实存在。Martini 在稠合芳香环里就用到了；AdResS 风格的混合分辨率方法在 AA/CG 边界也会用到。数据结构不需要理解这些——它只需要让用户能表达出来。

## 从全原子投影是用户代码的事，不是框架的事

`CoarseGrain` 没有 `from_atomistic` 工厂方法，也没有 `to_atomistic`。这有意为之。从全原子投影到粗粒化涉及多个独立决策：原子怎么分组、每个珠子位置怎么算、CG 键是从跨珠子的全原子键推断还是显式声明、哪些附加属性要复制过来。每个决策都有不止一个合理答案。

框架的作用是让用户能方便地表达投影方案，而不是替用户选策略。

```python
import numpy as np

def my_coarsegrain(ato, mask):
    """一个简单的不相交分区投影，使用质心位置和
    跨键推断。可自由改编。"""
    cg = mp.CoarseGrain()
    bead_of = {}
    for idx in np.unique(mask):
        atoms = tuple(a for a, m in zip(ato.atoms, mask) if m == idx)
        pos = np.mean([[a["x"], a["y"], a["z"]] for a in atoms], axis=0)
        bead_of[int(idx)] = cg.def_bead(
            atoms=atoms,
            x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
        )

    atom_to_idx = {id(a): i for i, a in enumerate(ato.atoms)}
    seen = set()
    for bond in ato.bonds:
        bi = int(mask[atom_to_idx[id(bond.itom)]])
        bj = int(mask[atom_to_idx[id(bond.jtom)]])
        if bi == bj:
            continue
        key = (bi, bj) if bi < bj else (bj, bi)
        if key in seen:
            continue
        seen.add(key)
        cg.def_cgbond(bead_of[bi], bead_of[bj])
    return cg
```

这二十行代码就涵盖了基本 Martini 风格映射的完整 AA→CG 流程。要换成质量加权位置、基于残基的分区、用 ITP 显式声明键、或者给珠子加虚拟位点，改这个函数就行——不用跟框架较劲。

## 往返转换是构建器的职责，不是数据结构的职责

反过来，从粗粒化快照回到全原子，同样有意不在 `CoarseGrain` 里提供。反向映射（Backmapping）是构造性操作：需要按珠子类型索引的片段库、尊重键几何的放置过程，通常还要加一个弛豫步骤。Backward、initram、vermouth 这些工具都把它实现成流水线，而不是一个方法。在 MolPy 里，这个角色属于聚合物构建器层——它消费 `CoarseGrain` 快照，用用户提供的片段模板生成 `Atomistic` 输出。

归结起来要点很简单：`CoarseGrain` 就是放珠子和珠子间键的地方。投影、反向映射、能量计算、力场分配——都在它外面做。

## 空间和组合操作与 Atomistic 完全相同

`CoarseGrain` 的公共接口与 `Atomistic` 一致，空间混入和系统组合运算符的行为也就完全相同。

```python
cg2 = cg.copy()
cg2.move([10, 0, 0])
combined = cg + cg2

cg.replicate(4, transform=lambda copy, i: copy.move([i * 5, 0, 0]))
```

可以用谓词选珠子子集、批量重命名珠子类型、给结构附加任意元数据。

```python
cg.select(lambda b: b.get("type") == "P4")
cg.rename_type("P4", "Q4")
# 使用 .get 使得没有 "x" 坐标的珠子（例如仅用于映射的珠子）
# 不会被标记，而不是引发 KeyError。
cg.set_property(lambda b: b.get("x", 0.0) > 0, "region", "right")
```

这些方法不对 `"type"` 的含义、`"region"` 的语义、位置与物理空间的关系做任何隐含假设。它们不过是在图上做图操作——只不过节点恰好是珠子。

## 何时使用 CoarseGrain 而非 Atomistic

当类型系统需要明确区分节点是珠子还是原子时，用 `CoarseGrain`。当结构要序列化成 CG 拓扑（Martini ITP、用 CG 模型写 LAMMPS DATA 等）而不是全原子拓扑时，用它。当下游构建器要消费它并产生全原子输出时，用它。

反过来，如果珠子和原子在工作流中物理上可互换——比如跑联合原子（united-atom）模型，每个原子就是一个珠子——直接用 `Atomistic` 把原子当珠子使就行。两种数据结构的接口特意做到一致；选哪个是标签问题，不是能力问题。
