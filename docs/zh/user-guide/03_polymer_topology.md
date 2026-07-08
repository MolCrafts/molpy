[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/molcrafts/molpy/blob/master/docs/user-guide/03_polymer_topology.ipynb)

# 拓扑驱动的 CGSmiles 组装

只需换一条 CGSmiles 字符串，其他全部不变：同一套构建器配置就能生成线性链、环和支化星形聚合物。

!!! note "前置条件"
    本指南需要 RDKit（用于 `RDKitAdapter.generate_3d`）、`oplsaa.xml` 力场，并熟悉[逐步聚合物构建](02_polymer_stepwise.md)。

## 一个构建器，多种拓扑结构

逐步构建指南里的反应内核始终不变，变的只是循环结构。CGSmiles 把这个思路又推进一步：构建器配置完全一样，换一条拓扑表达式就能得到不同的结构。

```text
linear:  {[#EO2]|4[#PS]}
ring:    {[#EO2]1[#PS][#EO2][#PS][#EO2]1}
branch:  {[#PS][#EO3]([#PS])[#PS]}
```

这三种产物的构建器完全相同。变化的只是字符串本身。

## 构建前先解析拓扑字符串

CGSmiles 表达式把单体标签和连接关系编成一个图。在运行化学过程之前验证这个图，几乎不花什么成本，却能提前发现标签拼写错误和结构错误——比如环闭合数字没写对、分支括号位置不对——要是等构建器跑到深处才暴露，这些错误信息往往晦涩难懂。

```python
from molpy.parser import parse_cgsmiles

expressions = {
    "linear": "{[#EO2]|4[#PS]}",
    "ring": "{[#EO2]1[#PS][#EO2][#PS][#EO2]1}",
    "branch": "{[#PS][#EO3]([#PS])[#PS]}",
}

for name, expr in expressions.items():
    ir = parse_cgsmiles(expr)
    labels = [node.label for node in ir.base_graph.nodes]
    print(f"{name}: nodes={len(ir.base_graph.nodes)}, labels={labels}")
```

```text
linear: nodes=5, labels=['EO2', 'EO2', 'EO2', 'EO2', 'PS']
ring: nodes=5, labels=['EO2', 'PS', 'EO2', 'PS', 'EO2']
branch: nodes=4, labels=['PS', 'EO3', 'PS', 'PS']
```

图验证通过后，下一步给每个节点分配实际的分子结构。

## 每个标签对应一个分子模板

解析器生成的图上，节点带着 `EO2`、`EO3`、`PS` 这类标签。构建器维护一个库，里面是类型化后的 `Atomistic` 对象，它在库里查找每个标签的对应项。CGSmiles 里写了什么标签，库里就必须有——找不到就直接报错，不往下走。所有模板都用 `$` 标记反应端口。

```python
import molpy as mp
from molpy.typifier import OplsTypifier

ff = mp.io.read_xml_forcefield("oplsaa.xml")
typifier = OplsTypifier(ff, strict_typing=True)

BIGSMILES = {
    "EO2": "{[][$]OCCO[$][]}",
    "EO3": "{[][$]OCC(CO[$])(CO[$])[]}",
    "PS": "{[][$]OCC(c1ccccc1)CO[$][]}",
}


def build_monomer(bigsmiles, typifier):
    monomer = mp.parser.parse_monomer(bigsmiles)
    monomer = mp.adapter.RDKitAdapter(monomer).generate_3d(add_hydrogens=True, optimize=False)
    monomer = monomer.get_topo(gen_angle=True, gen_dihe=True)
    monomer = typifier.typify(monomer)
    return monomer


library = {label: build_monomer(bs, typifier) for label, bs in BIGSMILES.items()}

for label, mon in library.items():
    ports = [a.get("port") for a in mon.atoms if a.get("port")]
    print(f"{label}: atoms={len(mon.atoms)}, ports={ports}")
```

```text
2026-06-30 21:08:23,619 - molpy.io.forcefield.xml - INFO - Using built-in force field: /Users/roykid/work/molcrafts/molpy/src/molpy/data/forcefield/oplsaa.xml
```

```text
2026-06-30 21:08:23,623 - molpy.io.forcefield.xml - INFO - Parsing force field: OPLS-AA v0.1.0


2026-06-30 21:08:23,623 - molpy.io.forcefield.xml - INFO - Combining rule: geometric


2026-06-30 21:08:23,630 - molpy.io.forcefield.xml - INFO - Parsed 825 atom types


2026-06-30 21:08:23,631 - molpy.io.forcefield.xml - INFO - Parsed 307 bond types (OPLS-AA with unit conversion)


2026-06-30 21:08:23,634 - molpy.io.forcefield.xml - INFO - Parsed 964 angle types (OPLS-AA with unit conversion)


2026-06-30 21:08:23,635 - molpy.io.forcefield._rb_opls - WARNING - RB coefficients do not lie on the ideal 4-term OPLS manifold (C0+C1+C2+C3+C4 = 10.041600, expected ≈ 0). Conversion will preserve forces and relative energies exactly, but will introduce a constant energy offset of ΔE = 10.041600 kJ/mol. This does not affect MD simulations.


2026-06-30 21:08:23,638 - molpy.io.forcefield.xml - INFO - Parsed 1089 dihedral types (OPLS-AA with unit conversion)


2026-06-30 21:08:23,640 - molpy.io.forcefield.xml - INFO - Parsed 825 nonbonded parameters (OPLS-AA with unit conversion)


2026-06-30 21:08:23,640 - molpy.io.forcefield.xml - INFO - Parsed 825 atom types (by type)


EO2: atoms=10, ports=['$', '$']
EO3: atoms=17, ports=['$', '$', '$']
PS: atoms=29, ports=['$', '$']
```

库准备好了，剩下的就是定义反应——把一种模板连到另一种模板上去。

## 反应化学与拓扑结构无关

这里的反应内核和逐步聚合物构建指南里定义的脱水缩合完全一样。`select_hydroxyl_group` 从左侧单体的反应位点上找到 -OH 离去基团；`select_one_hydrogen` 从右侧单体的端口原子上取一个 H。两者配合，移除 -OH 和 -H，形成新的 C-O 键，同时释放出水分子。这些逻辑跟最终产物是线性链、环还是支化结构无关——拓扑的事全交给 CGSmiles 字符串。

??? note "反应设置（与逐步构建指南相同）"
    `select_hydroxyl_group` 函数从左侧单体的反应位点上找到 -OH 离去基团。`select_one_hydrogen` 从右侧单体的端口原子上取一个 H。两者配合完成脱水缩合：移除 -OH + -H，形成新的 C-O 键，释放出水。

```python
from molpy.core.atomistic import Atom, Atomistic
from molpy.builder.polymer import (
    Connector,
    CovalentSeparator,
    LinearOrienter,
    Placer,
    PolymerBuilder,
)
from molpy.reacter import (
    Reacter,
    find_neighbors,
    form_single_bond,
    select_neighbor,
    select_self,
)


def select_hydroxyl_group(struct: Atomistic, site: Atom) -> list[Atom]:
    for nb in find_neighbors(struct, site):
        if nb.get("element") != "O":
            continue
        hs = [a for a in find_neighbors(struct, nb, element="H")]
        if hs:
            return [nb, hs[0]]
    raise ValueError("No hydroxyl group found")


def select_one_hydrogen(struct: Atomistic, site: Atom) -> list[Atom]:
    hs = [a for a in find_neighbors(struct, site, element="H")]
    if not hs:
        raise ValueError("No hydrogen found")
    return [hs[0]]


rxn = Reacter(
    name="dehydration",
    anchor_selector_left=select_neighbor("C"),
    anchor_selector_right=select_self,
    leaving_selector_left=select_hydroxyl_group,
    leaving_selector_right=select_one_hydrogen,
    bond_former=form_single_bond,
)

rules = {(l, r): ("$", "$") for l in library for r in library}
connector = Connector(port_map=rules, reacter=rxn)
placer = Placer(
    separator=CovalentSeparator(buffer=-0.1),
    orienter=LinearOrienter(),
)

builder = PolymerBuilder(
    library=library,
    connector=connector,
    placer=placer,
    typifier=typifier,
)
```

构建器配置好了。三种产物的输入区别，只有开头的三个表达式。

## 拓扑结构只由 CGSmiles 字符串决定

把三个表达式传给同一个构建器，得到三种结构各不相同的聚合物。线性表达式编码一条链；环表达式用环闭合数字编成一个环；分支表达式用括号编出一个三官能团连接点。构建器把每条图边翻译成一次反应调用和一次放置调用——没有专门的环模式或分支模式。

```python
for name, expr in expressions.items():
    result = builder.build(expr)
    polymer = result.polymer
    print(f"{name}: atoms={len(polymer.atoms)}, steps={result.total_steps}")
```

```text
linear: atoms=57, steps=4
```

```text
ring: atoms=73, steps=5


branch: atoms=95, steps=3
```

同一个构建器，同一个反应，同一个库——变来变去只有 CGSmiles 字符串。这是拓扑驱动组装的核心好处：新增一种拓扑结构不需要写新代码。产物在内存中生成后，无论什么拓扑结构，写入磁盘的步骤都一样。

## 每种产物导出 LAMMPS 的步骤相同

所有产物共用同一套导出流程。

```python
import numpy as np
from pathlib import Path

output_dir = Path("03_output")
output_dir.mkdir(exist_ok=True)

for name, expr in expressions.items():
    result = builder.build(expr)
    typed_polymer = typifier.typify(result.polymer)
    frame = typed_polymer.to_frame()
    atoms = frame["atoms"]
    if "mol_id" not in atoms:
        atoms["mol_id"] = np.ones(atoms.nrows, dtype=int)

    coords = np.column_stack([atoms["x"], atoms["y"], atoms["z"]])
    lo = coords.min(axis=0) - 5.0
    hi = coords.max(axis=0) + 5.0
    frame.box = mp.Box(matrix=hi - lo, origin=lo)

    mp.io.write_lammps_system(output_dir / name, frame, ff)
```

## 故障排除

调试按以下顺序排查：

1. 先解析 CGSmiles 并验证节点数和键数
2. 确认每个标签在库中存在
3. 确认每个反应标签对的连接器规则存在
4. 如果反应失败，打印选定的位点和离去基团原子
5. 如果几何结构出现重叠，调整 `CovalentSeparator(buffer=...)`

参考：[逐步聚合物构建](02_polymer_stepwise.md)、[交联网络](04_crosslinking.md)。
