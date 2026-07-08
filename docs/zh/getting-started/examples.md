# 示例集锦

每个示例用少量代码演示一个完整的工作流：从分子描述到可计算的模拟对象。覆盖范围从单个小分子、溶剂盒子、虚拟位点模型到聚合物体系——MolPy 的编辑机制在这些场景下依次接受检验。每个示例末尾都链接着对应的深度指南。

需要带逐步讲解的完整教程（含 LAMMPS 导出），请从[快速入门](quickstart.md)开始。

## 小分子 — 解析、类型化、导出

从 SMILES 出发，加氢、生成坐标，最后分配 OPLS-AA 力场类型。

```python
import molpy as mp

mol   = mp.parser.parse_molecule("CCO")                  # 从 SMILES 生成乙醇（重原子）
mol   = mp.adapter.RDKitAdapter(mol).generate_3d(add_hydrogens=True)  # 添加氢原子 + 3D 坐标
ff    = mp.io.read_xml_forcefield("oplsaa.xml")          # 内置 OPLS-AA
typed = mp.typifier.OplsTypifier(ff).typify(mol)         # 分配力场类型

frame = typed.to_frame()   # 可供模拟的列式数组
# mp.io.write_lammps_system("output/", frame, ff) 写入 system.data + system.ff
# （请先设置 frame.box 和每个原子的 mol_id —— 参见快速入门）。
```

另见：[解析化学结构](../user-guide/01_parsing_chemistry.md) ·
[力场类型分配](../user-guide/06_typifier.md)。

## 溶剂盒子 — 填充 500 个水分子

构建一个水分子，再通过 Packmol 将 500 份无冲突副本填充到周期性立方体盒子里。

!!! note "需要 `packmol` 可执行文件"
    填充功能会调用外部 Packmol 程序。请先安装它，并确保 `packmol` 在您的 `PATH` 环境变量中。

```python
import molpy as mp
from molpy.pack import Packmol, InsideBoxConstraint

water = mp.Atomistic(name="water")
o  = water.def_atom(element="O", x=0.000, y=0.000, z=0.000)
h1 = water.def_atom(element="H", x=0.957, y=0.000, z=0.000)
h2 = water.def_atom(element="H", x=-0.239, y=0.927, z=0.000)
water.def_bond(o, h1)
water.def_bond(o, h2)

p = Packmol(workdir="pack_out")
p.def_target(
    water.to_frame(),
    number=500,
    constraint=InsideBoxConstraint(length=30.0),  # 30 Å 立方体
)
packed = p(max_steps=1000, seed=42)       # → 一个已填充的 Frame（1500 个原子）
```

另见：[体系填充](../user-guide/09_packing.md)。

## 虚拟位点 — TIP4P 水模型

在水分子 HOH 角平分线上添加一个非原子的 M 位点。构建器会复制输入、放置位点并重新分配电荷。

```python
import molpy as mp
from molpy.builder.virtualsite import Tip4pBuilder

water = mp.Atomistic(name="water")
o  = water.def_atom(element="O", x=0.000, y=0.000, z=0.000)
h1 = water.def_atom(element="H", x=0.957, y=0.000, z=0.000)
h2 = water.def_atom(element="H", x=-0.239, y=0.927, z=0.000)
water.def_bond(o, h1)
water.def_bond(o, h2)

water4p = Tip4pBuilder(d_om=0.1546).apply(water)   # d_om: O–M 距离（nm）；输入保持不变
```

另见：[极化与虚拟位点模型](../user-guide/10_polarizable.md)。

## 聚合物链 — 从 G-BigSMILES 到类型化的 Frame

从 G-BigSMILES 字符串直接构建带 3D 坐标的聚合物链，然后走一遍类型分配。

```python
import molpy as mp
from molpy.builder import polymer

peo   = polymer("{[<]CCOCC[>]}|10|")             # PEO，聚合度 = 10
ff    = mp.io.read_xml_forcefield("oplsaa.xml")
typed = mp.typifier.OplsTypifier(ff).typify(peo)

frame = typed.to_frame()   # 使用 mp.io.write_lammps_system(dir, frame, ff) 写入
```

另见：[拓扑驱动组装](../user-guide/03_polymer_topology.md)。

## 多分散熔体 — Schulz-Zimm 分布

从分子量分布中采样，生成可重现的链群体。

```python
import molpy as mp
from molpy.builder import polymer_system

# Mn = 1500 Da, Mw = 3000 Da, 总质量 ≈ 500 kDa
chains = polymer_system(
    "{[<]CCOCC[>]}|schulz_zimm(1500,3000)||5e5|",
    random_seed=42,
)
print(f"Built {len(chains)} chains")   # 可重现的链群体
frames = [c.to_frame() for c in chains]
```

另见：[多分散体系](../user-guide/05_polydisperse_systems.md) ·
[体系填充](../user-guide/09_packing.md)。

## AmberTools 管道 — GAFF2 参数

用 antechamber、parmchk2 和 tleap 处理单体，生成带 GAFF2 参数和部分原子电荷的 AMBER 拓扑文件。

!!! note "需要 AmberTools"
    该工作流会调用外部程序 `antechamber`、`parmchk2` 和 `tleap`。请先安装 AmberTools 并激活其环境。

```python
import molpy as mp
from molpy.builder import polymer, prepare_monomer

eo = prepare_monomer("{[<]CCOCC[>]}")  # BigSMILES → 3D + 连接端口

result = polymer(
    "{[#EO]|20}",
    library={"EO": eo},
    backend="amber",   # 运行 antechamber + parmchk2 + tleap
)
# result.prmtop_path, result.inpcrd_path, result.pdb_path
```

另见：[AmberTools 集成](../user-guide/13_ambertools_integration.md)。
