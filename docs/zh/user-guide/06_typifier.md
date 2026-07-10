[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/molcrafts/molpy/blob/master/docs/user-guide/06_typifier.ipynb)

# 力场类型化

类型化负责将化学结构映射到力场参数。力场中定义的 SMARTS 模式决定了每个原子、键、键角和二面角的类型归属。

!!! note "前置条件"
    本指南需要 RDKit 来生成三维坐标。类型化过程本身不依赖 RDKit。

## 类型化解决的问题

分子结构只描述原子和键，分子模拟真正需要的是*类型*——一种能把每个原子、键、键角和二面角对应到力场参数的标识符。同一个碳原子，化学环境一变，类型就不同：脂肪族 CH₃ 里的碳是 `opls_135`，芳环上的碳则是 `opls_145`。类型分错了不会报错，但模拟结果会偏离物理真实。

**Typifier 通过 SMARTS 模式匹配检查每个原子的化学环境，然后分配对应的力场类型。**

MolPy 的 `OplsTypifier` 一次调用就能完成所有分配：先确定原子类型，再确定成对参数，最后根据原子类型推导键、键角、二面角的类型。

## 端到端的类型化流程

流程就四步：构建结构、加载力场、创建 Typifier、调用 `typify`。

```python
import molpy as mp
from molpy.typifier import OplsTypifier

# 1. 构建结构
mol = mp.parser.parse_molecule("CCO")
mol = mp.adapter.RDKitAdapter(mol).generate_3d(add_hydrogens=True, optimize=True)
mol = mol.get_topo(gen_angle=True, gen_dihe=True)

print(f"atoms: {len(mol.atoms)}, bonds: {len(mol.bonds)}")
print(f"angles: {len(mol.angles)}, dihedrals: {len(mol.dihedrals)}")
```

```text
atoms: 9, bonds: 8
angles: 13, dihedrals: 12
```

加载力场和构建结构是两个独立步骤。力场是独立对象，可以在多个分子间共享，随时换版本，类型化之前也能随时检查。拿到力场后，用它构造一个 Typifier，对分子调用 `typify` 即可。

```python
# 2. 加载力场并进行类型化
# "oplsaa.xml" 随 MolPy 捆绑提供——无需单独下载
ff = mp.io.read_xml_forcefield("oplsaa.xml")
typifier = OplsTypifier(ff, strict_typing=True)

typed_mol = typifier.typify(mol)
```

```text
2026-06-30 21:11:37,176 - molpy.io.forcefield.xml - INFO - Using built-in force field: /Users/roykid/work/molcrafts/molpy/src/molpy/data/forcefield/oplsaa.xml
```

```text
2026-06-30 21:11:37,181 - molpy.io.forcefield.xml - INFO - Parsing force field: OPLS-AA v0.1.0


2026-06-30 21:11:37,181 - molpy.io.forcefield.xml - INFO - Combining rule: geometric


2026-06-30 21:11:37,187 - molpy.io.forcefield.xml - INFO - Parsed 825 atom types


2026-06-30 21:11:37,188 - molpy.io.forcefield.xml - INFO - Parsed 307 bond types (OPLS-AA with unit conversion)


2026-06-30 21:11:37,190 - molpy.io.forcefield.xml - INFO - Parsed 964 angle types (OPLS-AA with unit conversion)


2026-06-30 21:11:37,192 - molpy.io.forcefield.xml - WARNING - RB coefficients do not lie on the ideal 4-term OPLS manifold (C0+C1+C2+C3+C4 = 10.041600, expected ≈ 0). Conversion will preserve forces and relative energies exactly, but will introduce a constant energy offset of ΔE = 10.041600 kJ/mol. This does not affect MD simulations.


2026-06-30 21:11:37,195 - molpy.io.forcefield.xml - INFO - Parsed 1089 dihedral types (OPLS-AA with unit conversion)


2026-06-30 21:11:37,197 - molpy.io.forcefield.xml - INFO - Parsed 825 nonbonded parameters (OPLS-AA with unit conversion)


2026-06-30 21:11:37,197 - molpy.io.forcefield.xml - INFO - Parsed 825 atom types (by type)
```

`typify` 对结构做原地修改然后返回——返回结果的每个原子都带上了 `type` 字段和对应的参数。

```python
# 3. 查看结果
for atom in typed_mol.atoms:
    element = atom.get("element", "?")
    atype = atom.get("type", "untyped")
    charge = atom.get("charge") or 0.0
    print(f"  {element:2s} -> {atype:15s}  q={charge:+.4f}")
```

```text
  C  -> opls_135         q=-0.1800
  C  -> opls_157         q=+0.1450
  O  -> opls_154         q=-0.6830
  H  -> opls_140         q=+0.0600
  H  -> opls_140         q=+0.0600
  H  -> opls_140         q=+0.0600
  H  -> opls_140         q=+0.0600
  H  -> opls_140         q=+0.0600
  H  -> opls_155         q=+0.4180
```

## 原子类型化原理

Typifier 从力场 XML 里读取 SMARTS 模式。每条模式定义一种原子类型——比如 `[CX4;H3]` 匹配带三个氢的 sp3 碳（甲基碳）。Typifier 遍历所有原子，逐一比对模式库，分配最匹配的类型。

多条模式同时命中时，力场里的优先级和覆盖规则负责解决冲突。芳香氮和脂肪族氮这类容易混淆的情况，全靠这套分级匹配机制自动处理，不需要人工干预。

没有模式匹配时，MolPy 不会偷偷估算参数。未匹配的情况会明确报告出来，方便你检查并扩充规则集。参数分配和参数开发分开了，力场定义始终透明、可复现。

原子类型定下来后，Typifier 手里就有了机械推导键合类型所需的一切——下一节细说。

## 键合类型化原理

原子类型一确定，键合类型的推导就成了机械性工作：原子类型 `CT` 和 `OH` 之间的键，映射到键类型 `CT-OH`。键角按三个原子类型的序列确定，二面角按四个原子类型的序列确定。力场里的通配符类型（`*`）在没有精确匹配时充当回退。

MolPy 不内置静电模型。部分电荷要么来自力场自带的参数（比如 OPLS 风格），要么通过外部流程分配。MolPy 只管存储和传递已经定义好的电荷，不做电荷推导。

## 严格模式与非严格模式

严格模式（`strict_typing=True`）下，一遇到未类型化的原子就报错。这对开发来说是合理的默认行为——缺失的力场参数当场暴露，不会悄悄混进生产计算。

非严格模式（`strict_typing=False`）下，未类型化的原子直接跳过。通用力场覆盖不了特殊官能团时——比如带特殊官能团的分子——就用这个模式。

## 原子、键、键角和二面角都带上了类型标签

类型化完成后，遍历键、键角和二面角就能看到分配到的类型。

```python
# 键类型
for bond in typed_mol.bonds[:3]:
    i_sym = bond.itom.get("element")
    j_sym = bond.jtom.get("element")
    btype = bond.get("type", "untyped")
    print(f"  {i_sym}-{j_sym} -> {btype}")

# 键角类型
for angle in typed_mol.angles[:3]:
    names = [a.get("type", "?") for a in angle.endpoints]
    atype = angle.get("type", "untyped")
    print(f"  {'-'.join(names)} -> {atype}")
```

```text
  C-C -> CT-CT
  C-O -> CT-OH
  C-H -> CT-HC
  opls_157-opls_135-opls_140 -> CT-CT-HC
  opls_157-opls_135-opls_140 -> CT-CT-HC
  opls_157-opls_135-opls_140 -> CT-CT-HC
```

## 导出类型化结果到模拟

类型化完的结构就可以导出做模拟了。转成 `Frame`，加个盒子，写到 LAMMPS 或 GROMACS 格式就行。

```python
import numpy as np
from pathlib import Path

frame = typed_mol.to_frame()
frame.box = mp.Box.cubic(30.0)

# mol_id 不由类型化工具设置——为 LAMMPS full atom style 添加它
atoms = frame["atoms"]
if "mol_id" not in atoms:
    atoms["mol_id"] = np.ones(atoms.nrows, dtype=int)

outdir = Path("06_output")
outdir.mkdir(exist_ok=True)

mp.io.write_lammps_system(outdir / "ethanol", frame, ff)

print(f"exported to {outdir}")
```

```text
exported to 06_output
```

`write_lammps_system` 这个便捷函数会自动过滤力场，只保留帧里用到的类型，并通过格式化器系统把规范字段名（`charge`、`mol_id`）转成 LAMMPS 认得的名字（`q`、`mol`）。

## 聚合物连接处的增量重新类型化

`molpy.Reaction` 在两个单体之间形成新键时，连接处的化学环境变了，原来的原子类型、键类型、键角类型和二面角类型也随之失效。

MolPy 不会每个耦合步骤都重新类型化整条链，而是只重新类型化这次编辑真正扰动到的那一小片邻域。

这片邻域有多大，不是一个可调的旋钮。每个 typifier 都以 `TypeScope` 的形式声明自己的感受野——它要看到一个原子周围几根键，才能叫出这个原子的类型——而这一个数字同时定下了操作的两个半径。`GraphAssembler` 会围绕每根新键取出一个半径为 `2 x reach` 的球，对整个球做类型化，但只把内层 `reach` 那一壳写回去：真正改变了环境的正是这些原子，而外壳存在的唯一理由，是给它们提供一个正确的环境去匹配。内壳之外的原子本来就是对的，因此不去动它们。

相同的连接处会哈希到同一个键，只被类型化一次。于是类型化的遍数取决于体系中**不同**化学环境的数量，而不是形成了多少根键：构建一个千聚体，所花的类型化遍数和构建一个十聚体差不多。

要启用它，在构造 builder 时传入 typifier：

```text
from molpy.typifier.ambertools import AmberToolsTypifier

builder = PolymerBuilder(
    MonomerLibrary({"EO": eo}),
    mp.Reaction(ETHER),
    typifier=AmberToolsTypifier(amber, reach=2),  # 连接处在成键时就地重新类型化
    placer=ResiduePlacer(),
)
chain = builder.build("{[#EO]|20}")
```

不传 typifier，组装过程就完全不分配类型；那就等链构建完再对它做一次类型化。结果是一样的，只是代价正比于整条链，而不是正比于它的连接处。

## 标准力场不够用时

标准 OPLS-AA 覆盖了常见的有机官能团。碰上特殊分子——比如离子液体（TFSI）、金属配合物、反应中间体——往往需要自定义力场参数。这时有两个选择：

1. 用专用力场 XML，里面包含你需要的 SMARTS 模式和类型定义
2. 回到[力场](../tutorials/04_force_field.md)层手动定义类型

Typifier 不关心力场里有什么内容。只要 XML 里有 SMARTS 模式和类型定义，它就能匹配。

参见：[力场](../tutorials/04_force_field.md)、[组装](02_assembly.md)。
