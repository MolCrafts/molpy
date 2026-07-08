[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/molcrafts/molpy/blob/master/docs/user-guide/02_polymer_stepwise.ipynb)

# 逐步聚合物构建

从反应层面构建聚合物：选中哪些原子、形成哪根键、移除哪些原子——以及为何每次偶合后都要重新生成拓扑和类型。先手动操作一遍，顺序使用 `merge`、`def_bond`、`del_atom`、`get_topo`、`typify`；再用 `Reacter` 实现同样的过程，它把这一串操作封装成了可复用的规则。

!!! note "前置条件"
    本指南需要 RDKit（用于 `RDKitAdapter.generate_3d`）和 `oplsaa.xml` 力场文件。

单体是**环氧乙烷（EO）**，在 BigSMILES 中写作 `{[][<]OCCOCCOCCO[>][]}`。每个 EO 单元有两个反应性端口标记：左端氧原子上标 `<`，右端标 `>`。这些标记并非独立的原子，而是对现有原子的注释——指明偶合发生的位置。

---

## 第一部分 — 手动偶合两个单体

### 准备并检查两个单体

每个单体在参与偶合之前，须具备含显式氢的 3D 坐标、完整的拓扑结构和 OPLS-AA 类型。


```python
import molpy as mp
import numpy as np
from pathlib import Path
from molpy.reacter import find_neighbors, find_port
from molpy.typifier import OplsTypifier

output_dir = Path("02_output")
output_dir.mkdir(exist_ok=True)

ff = mp.io.read_xml_forcefield("oplsaa.xml")
typifier = OplsTypifier(ff, strict_typing=False)

MONOMER_BIGSMILES = "{[][<]OCCOCCOCCO[>][]}"


def make_eo_monomer():
    m = mp.parser.parse_monomer(MONOMER_BIGSMILES)
    m = mp.adapter.RDKitAdapter(m).generate_3d(add_hydrogens=True, optimize=True)
    m = m.get_topo(gen_angle=True, gen_dihe=True)
    m = typifier.typify(m)
    return m


mon_a = make_eo_monomer()
mon_b = make_eo_monomer()
mon_b.move([10.0, 0.0, 0.0])  # 平移到远离 merged 后坐标重叠的位置

print(f"单体: {len(mon_a.atoms)} 个原子, {len(mon_a.bonds)} 根键")
print(f"      {len(mon_a.angles)} 个键角, {len(mon_a.dihedrals)} 个二面角")
print(f"未分类: {sum(1 for a in mon_a.atoms if a.get('type') is None)} 个原子")
```

```text
2026-06-30 21:08:06,136 - molpy.io.forcefield.xml - INFO - Using built-in force field: /Users/roykid/work/molcrafts/molpy/src/molpy/data/forcefield/oplsaa.xml
```


```text
2026-06-30 21:08:06,140 - molpy.io.forcefield.xml - INFO - Parsing force field: OPLS-AA v0.1.0


2026-06-30 21:08:06,140 - molpy.io.forcefield.xml - INFO - Combining rule: geometric


2026-06-30 21:08:06,146 - molpy.io.forcefield.xml - INFO - Parsed 825 atom types


2026-06-30 21:08:06,147 - molpy.io.forcefield.xml - INFO - Parsed 307 bond types (OPLS-AA with unit conversion)


2026-06-30 21:08:06,149 - molpy.io.forcefield.xml - INFO - Parsed 964 angle types (OPLS-AA with unit conversion)


2026-06-30 21:08:06,151 - molpy.io.forcefield.xml - INFO - Parsed 1089 dihedral types (OPLS-AA with unit conversion)


2026-06-30 21:08:06,155 - molpy.io.forcefield.xml - INFO - Parsed 825 nonbonded parameters (OPLS-AA with unit conversion)


2026-06-30 21:08:06,155 - molpy.io.forcefield.xml - INFO - Parsed 825 atom types (by type)


monomer: 24 atoms, 23 bonds
         40 angles, 45 dihedrals
untyped: 0 atoms
```


将单体导出为第一个检查点：


```python
def save_step(name: str, chain, ff, output_dir: Path) -> None:
    frame = chain.to_frame()
    atoms = frame["atoms"]
    atoms["mol_id"] = np.ones(atoms.nrows, dtype=int)
    coords = np.column_stack([atoms["x"], atoms["y"], atoms["z"]])
    lo = coords.min(axis=0) - 5.0
    hi = coords.max(axis=0) + 5.0
    frame.box = mp.Box(matrix=hi - lo, origin=lo)
    mp.io.write_lammps_system(output_dir / name, frame, ff)
    print(f"  已保存: {output_dir / name / (name + '.data')}")


save_step("peo1", mon_a, ff, output_dir)
```

```text
  已保存: 02_output/peo1/peo1.data
```


### 识别偶合反应中的四个参与者

脱水缩合每形成一根 C–O 键就脱去一个水分子。修改拓扑前，先定位涉及的四个原子：

1. **锚定碳** — `mon_a` 右端参与新 C–O 键的碳原子
2. **mon_a 的端口氧**（`>`）— 羟基氧，将作为 –OH 的一部分离去
3. **羟基氢** — 连接到该端口氧的氢，随氧一同离去
4. **mon_b 的离去氢** — `mon_b` 左端端口氧（`<`）上的一个氢，以水的另一半形式离去


```python
# mon_a 的右端口和 mon_b 的左端口
port_r = find_port(mon_a, ">")
port_l = find_port(mon_b, "<")

# 锚定碳是右端口氧的碳邻居
anchor_C = [a for a in find_neighbors(mon_a, port_r) if a.get("element") == "C"][0]

# 离去基团 –OH：端口氧及其单个氢
leaving_OH_oxygen = port_r
leaving_OH_hydrogen = find_neighbors(mon_a, leaving_OH_oxygen, element="H")[0]

# 离去 H：mon_b 左侧端口上的一个氢
leaving_H_b = find_neighbors(mon_b, port_l, element="H")[0]

print(f"新键将形成:  {anchor_C.get('element')} — {port_l.get('element')}")
print(
    f"从 mon_a 离去:  {leaving_OH_oxygen.get('element')} + {leaving_OH_hydrogen.get('element')}  (= OH)"
)
print(f"从 mon_b 离去:  {leaving_H_b.get('element')}  (= H)")
print(f"净脱除:         H₂O")
```

```text
new bond will form:  C — O
leaving from mon_a:  O + H  (= OH)
leaving from mon_b:  H  (= H)
net removal:         H₂O
```


### 合并、形成键、移除离去原子

偶合只需三步：


```python
# 1. 将两个单体合并为单个 Atomistic（两个单体的原子和键都合到一起）
dimer = mon_a.merge(mon_b)
print(
    f"合并后: {len(dimer.atoms)} 个原子  (= {len(mon_a.atoms)} + {len(mon_b.atoms)})"
)

# 2. 在锚定碳和 mon_b 的端口氧之间形成新 C–O 键
new_bond = dimer.def_bond(anchor_C, port_l)
print(
    f"新键: {new_bond.endpoints[0].get('element')}—{new_bond.endpoints[1].get('element')}"
)

# 3. 移除三个离去原子；它们关联的键会自动删除
dimer.del_atom(leaving_OH_oxygen, leaving_OH_hydrogen, leaving_H_b)
print(
    f"移除后: {len(dimer.atoms)} 个原子  (= {len(dimer.atoms)} = {len(mon_a.atoms) + len(mon_b.atoms)} − 3)"
)
```

```text
after merge: 48 atoms  (= 48 + 24)
new bond: C—O
after removal: 45 atoms  (= 45 = 69 − 3)
```


`del_atom` 删除所有引用被移除原子的键、键角和二面角。新 C–O 键虽已建好，但连接处的拓扑还不完整。

### 重新生成拓扑并重新分类连接区域

新键引入了两个单体中原本没有的三体与四体路径。`get_topo` 负责枚举这些路径，`typify` 为枚举出的每一项分配力场类型。


```python
# 检查修复前连接区域中未解析的内容
print(
    f"get_topo 前未分类的键:      {sum(1 for b in dimer.bonds if b.get('type') is None)}"
)
print(
    f"get_topo 前未分类的键角:     {sum(1 for a in dimer.angles if a.get('type') is None)}"
)
print(
    f"get_topo 前未分类的二面角:  {sum(1 for d in dimer.dihedrals if d.get('type') is None)}"
)

angles_before = len(dimer.angles)
dihe_before = len(dimer.dihedrals)
dimer = dimer.get_topo(gen_angle=True, gen_dihe=True)

print(
    f"\n键角:    {angles_before} → {len(dimer.angles)}  (+{len(dimer.angles) - angles_before} 跨连接)"
)
print(
    f"二面角: {dihe_before} → {len(dimer.dihedrals)}  (+{len(dimer.dihedrals) - dihe_before} 跨连接)"
)

dimer = typifier.typify(dimer)

print(f"\n未分类的键:      {sum(1 for b in dimer.bonds if b.get('type') is None)}")
print(f"未分类的键角:     {sum(1 for a in dimer.angles if a.get('type') is None)}")
print(f"未分类的二面角:  {sum(1 for d in dimer.dihedrals if d.get('type') is None)}")
```

```text
untyped bonds before get_topo:      1
untyped angles before get_topo:     0
untyped dihedrals before get_topo:  0

angles:    75 → 79  (+4 cross-junction)
dihedrals: 81 → 90  (+9 cross-junction)
```



```text
untyped bonds:      0
untyped angles:     0
untyped dihedrals:  0
```


全部相互作用已分配完毕。导出二聚体：


```python
save_step("peo2", dimer, ff, output_dir)
```

```text
  已保存: 02_output/peo2/peo2.data
```


### 循环构建任意链长

上述五步流程——`merge` → `def_bond` → `del_atom` → `get_topo` → `typify`——可直接套用到后续每个单元。循环从二聚体延伸到五聚体，每步保存一个快照。


```python
chain = make_eo_monomer()
save_step("peo1", chain, ff, output_dir)

for i in range(1, 5):
    unit = make_eo_monomer()
    unit.move([10.0 * i, 0.0, 0.0])

    # 识别当前链端和传入单元上的反应原子
    p_r = find_port(chain, ">")
    p_l = find_port(unit, "<")
    anc = [a for a in find_neighbors(chain, p_r) if a.get("element") == "C"][0]
    l_O = p_r
    l_H1 = find_neighbors(chain, l_O, element="H")[0]
    l_H2 = find_neighbors(unit, p_l, element="H")[0]

    # 偶合
    chain = chain.merge(unit)
    chain.def_bond(anc, p_l)
    chain.del_atom(l_O, l_H1, l_H2)
    chain = chain.get_topo(gen_angle=True, gen_dihe=True)
    chain = typifier.typify(chain)

    name = f"peo{i + 1}"
    save_step(name, chain, ff, output_dir)
    print(f"{name}: {len(chain.atoms)} 个原子, {len(chain.bonds)} 根键")
```

```text
  已保存: 02_output/peo1/peo1.data
```


```text
  已保存: 02_output/peo2/peo2.data
peo2: 45 atoms, 44 bonds


  已保存: 02_output/peo3/peo3.data
peo3: 66 atoms, 65 bonds


  已保存: 02_output/peo4/peo4.data
peo4: 87 atoms, 86 bonds


  已保存: 02_output/peo5/peo5.data
peo5: 108 atoms, 107 bonds
```


循环运行完后，`02_output/` 下有五个数据文件——`peo1.data` 到 `peo5.data`，每个都是不同链长的有效 LAMMPS 输入。

---

## 第二部分 — 使用 Reacter 自动进行相同的偶合

第一部分将每一步决策都显式化了。这套五步流程——识别参与者、merge、def_bond、del_atom、get_topo、typify——正是 `Reacter` 要编码为规则的内容。选择逻辑只需写一次，`rxn.run` 负责执行。

### 定义反应规则

`Reacter` 需要四个选择器函数：左锚定原子、右锚定原子，以及左右两侧的离去原子各一个。


```python
from molpy.core.atomistic import Atom, Atomistic
from molpy.reacter import (
    Reacter,
    form_single_bond,
    select_hydrogens,
    select_neighbor,
    select_self,
)


def select_hydroxyl_group(struct: Atomistic, reaction_site: Atom) -> list[Atom]:
    """返回 [O, H] — 锚定碳上的羟基离去基团。"""
    for neighbor in find_neighbors(struct, reaction_site):
        if neighbor.get("element") != "O":
            continue
        h_neighbors = find_neighbors(struct, neighbor, element="H")
        if h_neighbors:
            return [neighbor, h_neighbors[0]]
    raise ValueError("反应位点附近未找到羟基")


rxn = Reacter(
    name="dehydration",
    anchor_selector_left=select_neighbor("C"),  # 等同于：找到端口 > 旁边的 C
    anchor_selector_right=select_self,  # 等同于：端口 < 的 O 本身
    leaving_selector_left=select_hydroxyl_group,  # 等同于：l_O + l_H1
    leaving_selector_right=select_hydrogens(1),  # 等同于：l_H2
    bond_former=form_single_bond,  # 等同于：def_bond(anc, p_l)
)
```

每个选择器直接对应到第一部分手动执行的操作。

### 每步一次调用构建相同的五聚体


```python
chain = make_eo_monomer()
save_step("peo1_rxn", chain, ff, output_dir)

for i in range(1, 5):
    unit = make_eo_monomer()
    unit.move([10.0 * i, 0.0, 0.0])

    result = rxn.run(
        left=chain,
        right=unit,
        port_atom_L=find_port(chain, ">"),
        port_atom_R=find_port(unit, "<"),
        compute_topology=True,
    )
    chain = result.product
    chain = chain.get_topo(gen_angle=True, gen_dihe=True)
    chain = typifier.typify(chain)

    name = f"peo{i + 1}_rxn"
    save_step(name, chain, ff, output_dir)
    print(f"{name}: {len(chain.atoms)} 个原子")
```

```text
  已保存: 02_output/peo1_rxn/peo1_rxn.data
```


```text
  已保存: 02_output/peo2_rxn/peo2_rxn.data
peo2_rxn: 45 atoms


  已保存: 02_output/peo3_rxn/peo3_rxn.data
peo3_rxn: 66 atoms


  已保存: 02_output/peo4_rxn/peo4_rxn.data
peo4_rxn: 87 atoms


  已保存: 02_output/peo5_rxn/peo5_rxn.data
peo5_rxn: 108 atoms
```


`rxn.run` 内部执行的就是 merge、`def_bond`、`del_atom`——与第一部分相同的三个操作，顺序一致。调用方仍需在每步之后自行调用 `get_topo` 和 `typify`，因为这两步依赖具体的力场上下文，不在反应规则的职责范围内。

## 故障排除

**未找到端口**

检查端口标记是否存在于预期原子上：


```python
for a in mon_a.atoms:
    if a.get("port"):
        print(f"  element={a.get('element')}  port={a.get('port')}")
```

```text
  element=O  port=<
  element=O  port=<
  element=O  port=>
```


**"未找到羟基"（仅第二部分）**

打印锚定碳的邻居，查看选择器正在搜索的内容：


```python
p_r = find_port(mon_a, ">")
anc = [a for a in find_neighbors(mon_a, p_r) if a.get("element") == "C"][0]
for nb in find_neighbors(mon_a, anc):
    print(f"  element={nb.get('element')}")
```

```text
  element=C
  element=O
  element=H
  element=H
```


**偶合后存在未分类的相互作用（两部分均适用）**

`get_topo` 必须在 `typify` 之前调用，且两者都必须在合并 + 键形成 + 移除之后：

```python
chain = chain.merge(unit)
chain.def_bond(anc, p_l)
chain.del_atom(l_O, l_H1, l_H2)
chain.get_topo(gen_angle=True, gen_dihe=True)   # 第一步
chain = typifier.typify(chain)                  # 第二步
```

**导出失败：缺少 `mol_id`**

LAMMPS `full` 原子风格需要 `mol_id`。`save_step` 辅助函数已设置好，但手动导出时：

```python
frame["atoms"]["mol_id"] = np.ones(frame["atoms"].nrows, dtype=int)
```

参见：[基于拓扑的组装](03_polymer_topology.md)，[交联网络](04_crosslinking.md)。
