# 快速入门

两种方式入门。**快速路径** 用六行代码跑通完整流程。**完整教程** 随后一步步构建一个 TIP3P 水盒子——模板、类型分配、盒子、导出——每步边界都清晰可见，后续自动化也是同样的流程。

## 快速路径：从 SMILES 到已分配类型的系统

```python
import molpy as mp

mol   = mp.parser.parse_molecule("CCO")                  # 从 SMILES 解析乙醇（重原子）
mol   = mp.adapter.RDKitAdapter(mol).generate_3d(add_hydrogens=True)  # 添加氢原子 + 3D 坐标
ff    = mp.io.read_xml_forcefield("oplsaa.xml")          # 内置 OPLS-AA
typed = mp.typifier.OplsTypifier(ff).typify(mol)         # 分配力场类型

frame = typed.to_frame()                                 # 列式数组
print(frame["atoms"].nrows, "typed atoms")               # 9 个已分配类型的原子
```

这就是 MolPy 的核心流程——解析、嵌入、类型分配、转换。本手册的每篇指南都是围绕这些边界展开的变体。下面分步重做一遍，每一步都完全可控、显式操作。

## 完整教程：一个 TIP3P 水盒子

本节从头到尾构建一个小型 TIP3P 水盒子并导出 LAMMPS 输入文件：

- 一个描述已分配类型系统的 LAMMPS 数据文件 (`.data`)
- 一个包含 TIP3P 系数的 LAMMPS 力场文件 (`.ff`)

```python
from pathlib import Path

import numpy as np
import molpy as mp
from molpy.io.forcefield import read_xml_forcefield
from molpy.typifier import OplsTypifier
```

### 1. 定义一个 TIP3P 水分子

`Atomistic` 是 MolPy 的化学容器：一个分子图，原子是节点，键是边。内置 `tip3p.xml` 中 TIP3P 的键长以 **nm** 为单位，所以我们构建坐标时也用 **nm**，保持单位一致。

```python
water_template = mp.Atomistic(name='water_tip3p')

theta = 1.82421813418  # 弧度（TIP3P H-O-H 角）
r_oh = 0.09572  # nm（TIP3P O-H 键长）

o = water_template.def_atom(element='O', name='O', x=0.0, y=0.0, z=0.0, charge=-0.834)
h1 = water_template.def_atom(element='H', name='H1', x=r_oh, y=0.0, z=0.0, charge=0.417)
h2 = water_template.def_atom(
    element='H',
    name='H2',
    x=r_oh * float(np.cos(theta)),
    y=r_oh * float(np.sin(theta)),
    z=0.0,
    charge=0.417,
 )

water_template.def_bond(o, h1, order=1)
water_template.def_bond(o, h2, order=1)

# get_topo 对 *副本* 感知角/二面角（非变异操作）——保存结果
water_template = water_template.get_topo(gen_angle=True, gen_dihe=False)

print('atoms:', len(water_template.atoms), 'bonds:', len(water_template.bonds))
print('angles:', len(list(water_template.links.bucket(mp.Angle))))
print('atom names:', [a.get('name') for a in water_template.atoms])
```

### 2. 分配 TIP3P 类型

加载内置 `tip3p.xml`，让 `OplsTypifier` 分配原子类型、键合类型和非键参数——完全基于内置数据，结果确定。

> **注意：** `OplsTypifier` 是 MolPy 通用的基于 SMARTS 的类型分配引擎，名称指向匹配引擎而非力场，不要误以为它只适用于 OPLS。它根据所加载力场中的模式匹配原子——这里是 `tip3p.xml`——然后分配该文件定义的类型。

```python
ff = read_xml_forcefield('tip3p.xml')

typifier = OplsTypifier(
    ff,
    skip_atom_typing=False,
    skip_dihedral_typing=True,
    strict_typing=True,
 )
water_template = typifier.typify(water_template)

print('atom types:', [a.get('type') for a in water_template.atoms])
print('bond types:', [b.get('type') for b in water_template.bonds])
print('angle types:', [a.get('type') for a in water_template.links.bucket(mp.Angle)])
print('example LJ params on O:', {k: water_template.atoms[0].get(k) for k in ['sigma', 'epsilon']})
```

### 3. 实例化并变换分子

模板是可复用的 `Atomistic`；实例是放入更大系统的副本。变换是确定性的刚体操作：

```python
water_instance = water_template.copy()

water_instance.rotate(axis=[0.0, 0.0, 1.0], angle=float(np.pi / 2.0), about=[0.0, 0.0, 0.0])
water_instance.move(delta=[0.5, 0.0, 0.0])

coords = np.array([[a['x'], a['y'], a['z']] for a in water_instance.atoms], dtype=float)
print('instance center (nm):', coords.mean(axis=0).tolist())
```

### 4. 构建水盒子

在正交周期盒子内的简单 3D 网格上放置副本。（这是确定性网格，不是填充算法——如需在目标密度下进行无碰撞填充，请参见 [Packing Systems](../user-guide/09_packing.md)。）

```python
nx, ny, nz = 4, 4, 4
spacing = 0.32  # nm
n_total = nx * ny * nz

water_box_atomistic = mp.Atomistic(name='water_box_tip3p')
mol_id = 1
idx = 0

for iz in range(nz):
    for iy in range(ny):
        for ix in range(nx):
            mol = water_template.copy()
            mol.rotate(axis=[0.0, 0.0, 1.0], angle=float(0.1 * idx), about=[0.0, 0.0, 0.0])
            mol.move(delta=[ix * spacing, iy * spacing, iz * spacing])
            for atom in mol.atoms:
                atom['mol_id'] = mol_id
            water_box_atomistic.merge(mol)
            mol_id += 1
            idx += 1

water_box_atomistic = typifier.typify(water_box_atomistic)

box = mp.Box.orth([nx * spacing, ny * spacing, nz * spacing])
print('box lengths (nm):', box.lengths.tolist())
print('box atoms:', len(water_box_atomistic.atoms), 'box bonds:', len(water_box_atomistic.bonds))
```

### 5. 将 `Atomistic` 转换为 `Frame`

`Frame` 是列式容器——命名表加上模拟盒子和元数据。写入器操作的是 `Frame`，所以编辑好的图结构在这里变成可导出的表格。

```python
frame = water_box_atomistic.to_frame()
frame.box = box  # box 是 Frame 的一等属性；写入器读取 frame.box

atoms = frame['atoms']
n_atoms = atoms.nrows

atoms['id'] = np.arange(1, n_atoms + 1, dtype=int)
atoms['mol_id'] = np.asarray(atoms['mol_id'], dtype=int)
atoms['charge'] = np.asarray(atoms['charge'], dtype=float)

print('atoms rows:', frame['atoms'].nrows)
print('bonds rows:', frame['bonds'].nrows)
# `to_frame()` 仅为存在的链接类型生成块。此 TIP3P
# 模板带有键但不含显式的角链接，因此不存在
# 'angles' 块——访问前需要守卫。
if 'angles' in frame:
    print('angles rows:', frame['angles'].nrows)
```

### 6. 导出为 LAMMPS 文件

```python
out_dir = Path('quickstart-output')
out_dir.mkdir(parents=True, exist_ok=True)

mp.io.write_lammps_data(out_dir / 'water_box_tip3p.data', frame, atom_style='full')
mp.io.write_lammps_forcefield(out_dir / 'water_box_tip3p.ff', ff)

print('wrote:', out_dir / 'water_box_tip3p.data')
print('wrote:', out_dir / 'water_box_tip3p.ff')
```

## 你构建了什么

- 一个 TIP3P 水分子作为可编辑的 `Atomistic` 图——从内置 `tip3p.xml` 中分配了类型、参数和推导出的角。
- 在周期盒子中确定性放置了 64 个分子。
- 一个带有附接盒子的 `Frame`，导出为 LAMMPS 数据文件 + 力场文件。

**下一步：** 参见 [Example Gallery](examples.md) 获取更多可套用工作流，或参见 [data-model tutorials](../tutorials/index.md) 理解刚刚使用的每个对象。
