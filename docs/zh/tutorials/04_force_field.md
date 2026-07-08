# 力场

MolPy 把力场当作可查询、可验证的数据——在它被写入引擎文件之前，始终是透明的。

## 为什么把结构和参数分开？

经典分子动力学里，体系的物理行为完全由力场决定——也就是定义原子间相互作用的方程和参数集。参数设错是模拟流程中最隐蔽的错误来源。原子类型不对或二面角项缺失，程序不会报错，只会给出看似合理但实际错误的结果。

多数模拟工具把结构和参数捆在同一个文件里（LAMMPS data + coefficients、GROMACS topology）。这样一来，投入昂贵计算之前很难单独检查参数分配。MolPy 把它们分开存放，参数分配正确与否，在运行模拟之前就能验证。

## 先定义参数，再计算数值

很多库把力场定义和数值计算揉成一团。设一个参数，立刻得到一个数值对象，中间没有任何可以检查或验证的状态。MolPy 刻意把这两步拆开。

**力场在被编译成可执行的数值对象之前，应该始终是可检查的数据结构。**

参数化正是静默错误的高发区。原子类型搞错、键参数缺失或不一致——这些问题最好在模型还是透明数据结构的时候发现，而不是等它被打包成数组或特定引擎的文件之后。

## 三层结构：Style、Type、Potential

MolPy 的力场数据分为三层：

```text
ForceField
├── AtomStyle "full"
│   ├── AtomType "CT"  (mass=12.011, charge=-0.18)
│   └── AtomType "HC"  (mass=1.008, charge=0.06)
├── BondStyle "harmonic"
│   ├── BondType "CT-HC"  (k=340.0, r0=1.09)
│   └── BondType "CT-CT"  (k=268.0, r0=1.529)
├── AngleStyle "harmonic"
│   └── AngleType "HC-CT-HC"  (k=33.0, theta0=107.8)
├── DihedralStyle "opls"
│   └── DihedralType "HC-CT-CT-HC"  (K1=0.0, K2=0.0, K3=0.3, K4=0.0)
└── PairStyle "lj126/cut"
    ├── PairType "CT"  (epsilon=0.066, sigma=3.50)
    └── PairType "HC"  (epsilon=0.030, sigma=2.50)
```

`Style` 定义一类相互作用及其参数契约——谐振键、OPLS 二面角、Lennard-Jones 对都是不同的 Style。`Type` 是某个 Style 下的一条具体参数记录。`Potentials` 求值器是数值实体，由完整的力场模型生成，在已赋类型的 `Frame` 上运行。数值内核位于 molrs Rust 扩展中。

整个流程始终是：定义 styles → 填入 types → 求值为 potentials。

## 构建最小力场

先创建 `ForceField`，定义原子类型。原子类型是基础——所有成键和非键相互作用都引用它们。

```python
import molpy as mp

ff = mp.AtomisticForcefield(name="tutorial", units="real")

# "full" 对应 LAMMPS atom_style full（每原子电荷 + 分子 ID）
atom_style = ff.def_atomstyle("full")
ct = atom_style.def_type("CT", mass=12.011, charge=-0.18, element="C")
hc = atom_style.def_type("HC", mass=1.008,  charge=0.06,  element="H")
oh = atom_style.def_type("OH", mass=15.999, charge=-0.68, element="O")
```

键、角、二面角和对 styles 遵循同样的模式：创建 style，然后用显式参数名添加 types。

```python
bond_style = ff.def_bondstyle("harmonic")
bond_style.def_type(ct, hc, k=340.0, r0=1.09)
bond_style.def_type(ct, ct, k=268.0, r0=1.529)
bond_style.def_type(ct, oh, k=320.0, r0=1.41)

angle_style = ff.def_anglestyle("harmonic")
angle_style.def_type(hc, ct, hc, k=33.0, theta0=107.8)

dihedral_style = ff.def_dihedralstyle("opls")
dihedral_style.def_type(hc, ct, ct, hc, K1=0.0, K2=0.0, K3=0.3, K4=0.0)

# "lj126/cut" = 带截断的 12-6 Lennard-Jones（LAMMPS：lj/cut）
pair_style = ff.def_pairstyle("lj126/cut")
pair_style.def_type(ct, epsilon=0.066, sigma=3.50)
pair_style.def_type(hc, epsilon=0.030, sigma=2.50)
pair_style.def_type(oh, epsilon=0.170, sigma=3.12)
```

到这里，力场还是一份完整的数据结构。没有创建任何数值内核。所有内容仍然可读可改。

## 检查模型

导出之前，先把力场当数据检查一遍。文件可能语法正确，但参数仍然是错的。

单个 type 通过字典接口暴露参数。

```python
print(f"CT mass={ct['mass']}, charge={ct['charge']}")
print(f"CT element={ct.get('element')}")

bt = bond_style.get_type_by_name("CT-OH")
print(f"CT-OH: k={bt['k']}, r0={bt['r0']}")
```

遍历所有 styles 和 types，可以快速掌握模型的全局状态。

```python
from molpy.core.forcefield import Style, Type

for style in ff.get_styles(Style):
    types = style.get_types(Type)
    print(f"style={style.name!r}  [{len(types)} types]")
    for t in types:
        params = {k: v for k, v in t.params.kwargs.items()}
        print(f"  {t.name}: {params}")
```

按名称查找可以直接定位到特定的 style 或 type。

```python
bs = ff.get_style("bond", "harmonic")
ct_ct = bs.get_type_by_name("CT-CT")
print(f"CT-CT k={ct_ct['k']}")
```

## 求值为 Potentials

求值是模型的第一个严格完整性测试。`ff.to_potentials()` 返回一个**惰性的** `Potentials`——它不关联任何 frame（`len() == 0`，不可迭代）。要计算数值，需要传入一个已赋类型的 `Frame`：一个包含坐标的 `atoms` block，加上携带 `type` 列的成键 block（`bonds`、`angles` 等）。数值内核在 molrs Rust 扩展中运行。

```python
import numpy as np

# 最小帧：两个原子相距 1.2 Å，由一个 CT-HC 键连接。
frame = mp.Frame()
atoms = mp.Block()
atoms.insert("x", np.array([0.0, 1.2]))
atoms.insert("y", np.array([0.0, 0.0]))
atoms.insert("z", np.array([0.0, 0.0]))
frame["atoms"] = atoms

bonds = mp.Block()
bonds.insert("atomi", np.array([0], dtype=np.uint32))
bonds.insert("atomj", np.array([1], dtype=np.uint32))
bonds.insert("type", np.array(["CT-HC"], dtype=str))
frame["bonds"] = bonds

pots = ff.to_potentials()
energy = pots.calc_energy(frame)
forces = pots.calc_forces(frame)
print(f"energy = {energy}")
print(f"forces =\n{forces}")
```

如果引用的 type 缺失或必需参数不存在，求值会在此处抛出异常，而不是默默给出错误的结果。

## 导出到模拟引擎

模型内部一致之后，序列化就只是接口问题，和建模无关。同一个力场可以渲染成不同引擎的格式，无需重新定义物理参数。

### LAMMPS

```python
import io
from molpy.io.forcefield import LAMMPSForceFieldWriter

buf = io.StringIO()
writer = LAMMPSForceFieldWriter(buf, precision=4)
writer.write(ff)
print(buf.getvalue())
```

### GROMACS

```python
from molpy.io.forcefield.top import GromacsForceFieldWriter

GromacsForceFieldWriter("system.itp", precision=4).write(ff)
```

### XML

```python
from molpy.io.forcefield import XMLForceFieldWriter

XMLForceFieldWriter("system.xml", precision=6).write(ff)
```

## 何时需要扩展内置 Styles

实际项目总归会遇到内置 styles 不够用的情况——Morse 键、Buckingham 对、自定义扭转势。新形式的数值内核在 molrs Rust 扩展中添加；Python 端只需暴露一个轻量的 `Style`，并为每个导出后端注册参数格式化器。

完整扩展方案见[扩展力场](../developer/extending-forcefield.md)。

## 力场不在分子内部

还有一个值得明确指出的区别：结构和参数化是相关但独立的。分子可以在没有类型之前就存在。赋类型的系统也可以在力场导出之前就存在。MolPy 保留这些边界，目的是让模型验证和格式转换的逻辑更容易推理。

另请参阅：[原子结构与拓扑](01_atomistic_and_topology.md)、[Block 与 Frame](02_block_and_frame.md)。
