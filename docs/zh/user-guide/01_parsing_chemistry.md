[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/molcrafts/molpy/blob/master/docs/user-guide/01_parsing_chemistry.ipynb)

# 解析化学结构

从一行字符串到可编辑的结构：`mp.parser` 能读取 SMILES、SMARTS、BigSMILES 和 CGSmiles —— 本指南说明何时该用哪种。

## 四种记号，四种用途

化学记号本质上都是压缩方案。每种格式只回答一个特定问题，只编码该问题所需的信息。SMILES 回答"这个分子长什么样？"，编码原子、键和立体化学。SMARTS 回答"要匹配什么结构模式？"，编码逻辑约束而不是物理原子。BigSMILES 回答"重复单元是什么，从哪里连接？"，编码聚合物的连接意图。CGSmiles 回答"构建块怎么排列？"，编码拓扑结构，完全不涉及化学细节。

MolPy 的解析器模块（`mp.parser`）为每种问题都提供了对应的函数。便捷函数直接返回 `Atomistic` 对象；IR 级别的函数返回中间表示，供需要精细控制的场景使用——本章末尾会详细介绍。

## SMILES：描述一个具体分子

`parse_molecule` 用于处理完整指定的小分子，解析 SMILES 字符串后直接返回带原子和键的 `Atomistic` 对象。SMILES 中的氢原子是隐式的，坐标生成时再补上。

```python
import molpy as mp

mol = mp.parser.parse_molecule("CC(=O)OCC")  # 乙酸乙酯
print(f"atoms: {len(mol.atoms)}, bonds: {len(mol.bonds)}")

elements = [atom.get("element") for atom in mol.atoms]
print(elements)  # ['C', 'C', 'O', 'O', 'C', 'C']
```

```text
atoms: 6, bonds: 5
['C', 'C', 'O', 'O', 'C', 'C']
```

SMILES 中的点分隔组分表示离子对或溶剂混合物。碰到这类输入用 `parse_mixture`，它总是返回列表。

```python
ions = mp.parser.parse_mixture("[Li+].[F-]")
print(len(ions))  # 2

# parse_mixture 也适用于单分子
mols = mp.parser.parse_mixture("CCO")
print(len(mols))  # 1
```

```text
2
1
```

SMILES 用小写字母表示芳香原子。闭环数字必须成对出现：第一次出现打开环，第二次闭合。这样解析出来的苯环，每个原子都带芳香性标记——下游原子类型器依赖这些标记做判断。

```python
benzene = mp.parser.parse_molecule("c1ccccc1")
for atom in benzene.atoms:
    print(f"{atom.get('element')}, aromatic={atom.get('aromatic')}")
```

```text
C, aromatic=True
C, aromatic=True
C, aromatic=True
C, aromatic=True
C, aromatic=True
C, aromatic=True
```

SMILES/SMARTS 中显式指定的立体化学信息（四面体手性、双键构型）在解析和拓扑构建过程中会保留。

SMILES 适合已明确连接关系的小分子。如果要描述一个反复出现的结构模式，而非具体分子，SMARTS 才是正确的选择。

## SMARTS：模式匹配，不是结构构建

SMARTS 表面语法跟 SMILES 很像，但语义完全不同。SMILES 编码一个具体的分子，SMARTS 编码一个查询：一组约束条件，可以匹配多种不同分子。每个原子规格可以携带逻辑运算符、属性测试和通配符。

解析器返回 `SmartsIR` 对象，不是 `Atomistic`。这个区别很重要：SMARTS 表达式是原子类型器用的匹配规则，不是可供模拟的物理结构。

```python
query = mp.parser.parse_smarts("[C;X4][O;H1]")

print(f"query atoms: {len(query.atoms)}")
print(f"query bonds: {len(query.bonds)}")

for i, atom in enumerate(query.atoms):
    print(f"  atom {i}: {atom.expression}")
```

```text
query atoms: 2
query bonds: 1
  atom 0: AtomExpressionIR(op='weak_and', children=[AtomPrimitiveIR(type='symbol', value='C'), AtomPrimitiveIR(type='neighbor_count', value=4)])
  atom 1: AtomExpressionIR(op='weak_and', children=[AtomPrimitiveIR(type='symbol', value='O'), AtomPrimitiveIR(type='hydrogen_count', value=1)])
```

SMARTS 是力场原子类型化的语言：用 SMARTS 模式把原子环境映射成力场类型。原子类型和参数都确定之后，如果想模拟这些原子如何重复连接成聚合物链，就轮到 BigSMILES 出场了。

## BigSMILES：带连接意图的单体

标准 SMILES 没有重复单元或连接点的概念。BigSMILES 把端口标记（`<`、`>`、`$`）直接塞进字符串，让聚合物连接关系一目了然。每个标记把某个原子标为末端，标记它可以跟另一个重复单元成键。

`parse_monomer` 生成携带端口元数据的 `Atomistic` 对象，可以直接交给聚合物构建器链式拼接。

```python
monomer = mp.parser.parse_monomer("{[][<]CC(c1ccccc1)[>][]}")

print(f"atoms: {len(monomer.atoms)}")
ports = [a for a in monomer.atoms if a.get("port")]
print(f"ports: {len(ports)}")
for p in ports:
    print(f"  port '{p.get('port')}' on {p.get('element')}")
```

```text
atoms: 8
ports: 2
  port '<' on C
  port '>' on C
```

如果字符串里写了多种不同单体（共聚物），`parse_polymer` 会保留片段级别的组织结构，每种单体类型各自独立可访问。

```python
spec = mp.parser.parse_polymer("{[<]CC[>],[<]CC(C)[>]}")

print(f"topology:  {spec.topology}")
print(f"monomers:  {len(spec.all_monomers())}")
```

```text
topology:  random_copolymer
monomers:  2
```

BigSMILES 管的是每个嵌段的化学细节：原子、键和连接端口。如果要把视角拉高，描述这些嵌段在架构层面怎么排列——不关心内部化学——那就用 CGSmiles。

## CGSmiles：嵌块怎么连，不关心它们是什么

CGSmiles 在更高的抽象层运作。字符串里的节点是带标签的构建块，边是连接关系。字符串本身不涉及嵌块内部的原子信息——片段定义（句点后面那部分）才把每个标签绑定到对应的化学结构。

```python
from molpy.parser import parse_cgsmiles

# 线性链：5 个 PEO 副本，附带片段定义
cg = parse_cgsmiles("{[#PEO]|5}.{#PEO=[$]COC[$]}")

print(f"nodes: {len(cg.base_graph.nodes)}")
print(f"bonds: {len(cg.base_graph.bonds)}")
print(f"fragments: {len(cg.fragments)}")
```

```text
nodes: 5
bonds: 4
fragments: 1
```

语法紧凑且精确。`[#LABEL]` 引用命名嵌块；`|n` 重复嵌块；括号表示分支；匹配的数字做环闭合——跟 SMILES 完全一样。CGSmiles 是 `PolymerBuilder` 的输入：BigSMILES 定义每个嵌块是什么，CGSmiles 定义这些嵌块怎么排列。

## 把解析和转换分开：暴露中间表示

便捷函数（`parse_molecule`、`parse_monomer`）一次调用完成解析加转换，这对大多数工作流来说够用了。但如果需要诊断——检查芳香性判断对不对、验证端口分配、排查意外的原子数量——可以把两步拆开，看得更清楚。

```python
from molpy.parser import parse_smiles, smilesir_to_atomistic

# parse_smiles 返回 SmilesGraphIR（单分子）或 list[SmilesGraphIR]
# （点分隔的混合物）。如果确定输入是单个分子，不需要用 [0] 取——
# 裸 SMILES 字符串始终返回单个 IR 对象。
ir = parse_smiles("CCO")
print(f"IR atoms: {len(ir.atoms)}")

for atom_ir in ir.atoms:
    print(f"  element={atom_ir.element}, aromatic={atom_ir.aromatic}")

mol = smilesir_to_atomistic(ir)
print(f"Atomistic atoms: {len(mol.atoms)}")
```

```text
IR atoms: 3
  element=C, aromatic=False
  element=C, aromatic=False
  element=O, aromatic=False
Atomistic atoms: 3
```

BigSMILES 也适用同样思路。转换前检查 IR，可以看清端口标记怎么解析的、哪些原子被分配了连接角色。

```python
from molpy.parser import parse_bigsmiles, bigsmilesir_to_polymerspec

ir = parse_bigsmiles("{[<]CC[>],[<]CC(C)[>]}")
spec = bigsmilesir_to_polymerspec(ir)
print(f"topology: {spec.topology}")
```

```text
topology: random_copolymer
```

## 怎么选解析器

四种记号从完全指定的化学结构一路覆盖到纯拓扑结构。拿到具体的小分子 SMILES，用 `parse_molecule` 或 `parse_mixture`。连接端口开始变得重要——也就是分子成了聚合物重复单元时——改用 `parse_monomer` 或 `parse_polymer`。`parse_smarts` 留给原子类型器做匹配规则用，不要拿它创建结构。如果需要表达命名嵌块的架构排列，把内部化学留给片段定义，就用 `parse_cgsmiles`。

解析后需要 3D 坐标的话，用 `mp.adapter.RDKitAdapter(mol).generate_3d()`（需要 RDKit）。

另请参见：[逐步构建聚合物](02_polymer_stepwise.md)、[原子与拓扑结构](../tutorials/01_atomistic_and_topology.md)。
