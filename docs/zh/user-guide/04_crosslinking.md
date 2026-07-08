[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/molcrafts/molpy/blob/master/docs/user-guide/04_crosslinking.ipynb)

# 交联网络

LAMMPS `fix bond/react` 运行时需要一组配套文件——局域反应模板、紧密堆积的初始构型、拓扑和力场文件，且三者编号必须统一。MolPy 一条工作流就能生成全部内容。

!!! note "前置条件"
    本指南需要 RDKit、Packmol 和 `oplsaa.xml` 力场。假设读者已熟悉[逐步聚合构建](02_polymer_stepwise.md)。

## 反应性 MD 需要四个输出文件

LAMMPS 的 `fix bond/react` 在 MD 过程中驱动成键反应。它需要以下四样东西：反应模板，能捕获每次成键事件前后的局域拓扑；堆积构型，在合理密度下放置分子；力场系数，覆盖初始构型和反应后模板中的所有类型；以及一个 LAMMPS 输入脚本，把上述要素串起来依次做最小化、平衡和反应性 MD。MolPy 的 `BondReactReacter` 生成模板文件，`Packmol` 处理堆积，`write_lammps_bond_react_system` 一次调用就能导出编号统一的 data 文件、力场和模板。

## 两种单体共享一个反应模板

体系包含两种单体：**EO2**（线性，两个 `$` 端口）和 **EO3**（支化，三个 `$` 端口）。分子种类虽然不同，但只需要**一个**反应模板。

原因在于 `fix bond/react` 匹配的是*局域*拓扑，而非整个分子身份。模板只捕获反应位点附近几根键的范围，宽度由 `radius` 参数控制。EO2 和 EO3 的臂化学结构完全相同——每个反应性端口都位于 `...COCCO[$]` 片段的末端。BFS 在 `radius=4` 处停止时，尚未触及 EO3 的支化碳，因此提取的子图完全一致，不论该端口来自 EO2 还是 EO3 分子。

```text
EO2:   HO–OCCOCCOCCO–OH           ← 2 个相同的臂
            [$]          [$]

EO3:        COCCO–OH               ← 3 个相同的臂
           /     [$]
     C–COCCO–OH
           \     [$]
            COCCO–OH
                  [$]

模板半径只捕获端口邻域：
     ...COCCO–OH   (每个臂相同)
         ^^^^
         radius=4 在此停止
```

如果单体端口周围化学环境*不同*（例如胺与环氧基团反应），每种反应类型就需要单独一个模板。

```python
from pathlib import Path
import numpy as np
import molpy as mp
from molpy.core.atomistic import Atom, Atomistic
from molpy.core.element import Element
from molpy.typifier import OplsTypifier

ff = mp.io.read_xml_forcefield("oplsaa.xml")
typifier = OplsTypifier(ff, strict_typing=False)
```

```text
2026-07-01 03:07:38,585 - molpy.io.forcefield.xml - INFO - Using built-in force field: /Users/roykid/work/molcrafts/molpy/src/molpy/data/forcefield/oplsaa.xml
```

```text
2026-07-01 03:07:38,589 - molpy.io.forcefield.xml - INFO - Parsing force field: OPLS-AA v0.1.0


2026-07-01 03:07:38,589 - molpy.io.forcefield.xml - INFO - Combining rule: geometric


2026-07-01 03:07:38,596 - molpy.io.forcefield.xml - INFO - Parsed 825 atom types


2026-07-01 03:07:38,597 - molpy.io.forcefield.xml - INFO - Parsed 307 bond types (OPLS-AA with unit conversion)


2026-07-01 03:07:38,599 - molpy.io.forcefield.xml - INFO - Parsed 964 angle types (OPLS-AA with unit conversion)


2026-07-01 03:07:38,601 - molpy.io.forcefield._rb_opls - WARNING - RB coefficients do not lie on the ideal 4-term OPLS manifold (C0+C1+C2+C3+C4 = 10.041600, expected ≈ 0). Conversion will preserve forces and relative energies exactly, but will introduce a constant energy offset of ΔE = 10.041600 kJ/mol. This does not affect MD simulations.


2026-07-01 03:07:38,603 - molpy.io.forcefield.xml - INFO - Parsed 1089 dihedral types (OPLS-AA with unit conversion)


2026-07-01 03:07:38,605 - molpy.io.forcefield.xml - INFO - Parsed 825 nonbonded parameters (OPLS-AA with unit conversion)


2026-07-01 03:07:38,605 - molpy.io.forcefield.xml - INFO - Parsed 825 atom types (by type)
```

每个单体从 BigSMILES 解析，扩展为含氢原子的 3D 结构，再用 OPLS-AA 分配类型。原子 ID 必须在类型分配之前赋值，因为力场写入器依赖它们。

```python
eo2 = mp.parser.parse_monomer("{[][$]OCCOCCOCCO[$][]}")
eo2 = mp.adapter.RDKitAdapter(eo2).generate_3d(add_hydrogens=True, optimize=True)
eo2 = eo2.get_topo(gen_angle=True, gen_dihe=True)
for idx, atom in enumerate(eo2.atoms, start=1):
    atom["id"] = idx
eo2 = typifier.typify(eo2)
```

```python
eo3 = mp.parser.parse_monomer("{[]C(COCCO[$])(COCCO[$])COCCO[$][]}")
eo3 = mp.adapter.RDKitAdapter(eo3).generate_3d(add_hydrogens=True, optimize=True)
eo3 = eo3.get_topo(gen_angle=True, gen_dihe=True)
for idx, atom in enumerate(eo3.atoms, start=1):
    atom["id"] = idx
eo3 = typifier.typify(eo3)
```

```text
/Users/roykid/work/molcrafts/molpy/src/molpy/adapter/rdkit.py:719: UserWarning: UFF optimization returned code 1. Code 1 typically means convergence not reached within 200 iterations. The structure may still be improved.
  warnings.warn(msg, UserWarning)
```

确认 3D 生成后端口标记仍在：

```python
for label, mon in [("EO2", eo2), ("EO3", eo3)]:
    ports = [a.get("port") for a in mon.atoms if a.get("port")]
    print(f"{label}: atoms={len(mon.atoms)}, ports={ports}")
```

```text
EO2: atoms=24, ports=['$', '$']
EO3: atoms=38, ports=['$', '$', '$']
```

## 模板生成捕获局域反应环境

`BondReactReacter` 继承自 `Reacter`，增加了子图提取功能。它先在两个单体之间执行一次代表性反应，然后提取每个反应位点周围的局域环境（范围由 `radius` 控制），生成反应前和反应后的分子模板以及原子等价映射。

`radius` 参数控制 BFS 从锚原子出发延伸多少根键。取 4 时提供了足够的缓冲区，所有类型变化的原子距模板边缘至少有两根键——这是 LAMMPS `fix bond/react` 的要求。

反应遵循**脱水缩合**机理，与[逐步聚合构建](02_polymer_stepwise.md)中的化学过程相同。左侧锚原子是端口原子的碳邻接原子，离去基团为羟基（OH）。右侧锚原子就是端口原子本身，离去基团为一个氢原子。

```python
from molpy.reacter import (
    BondReactReacter,
    find_neighbors,
    find_port,
    form_single_bond,
    select_hydrogens,
    select_hydroxyl_group,
    select_neighbor,
    select_self,
)
```

类型分配器必须传给 `run()` 方法。若不传入，新生成的 C–O 键就没有力场类型，导出后模板时会静默丢弃——LAMMPS 会移除离去基团的原子，但交联键也永远创建不出来。

```python
reacter = BondReactReacter(
    name="rxn1",
    anchor_selector_left=select_neighbor("C"),
    anchor_selector_right=select_self,
    leaving_selector_left=select_hydroxyl_group,
    leaving_selector_right=select_hydrogens(1),
    bond_former=form_single_bond,
    radius=4,
)

left = eo2.copy()
right = eo2.copy()
result = reacter.run(
    left=left,
    right=right,
    port_atom_L=find_port(left, "$"),
    port_atom_R=find_port(right, "$"),
    compute_topology=True,
    typifier=typifier,
)
template = result.template
```

```text
2026-07-01 03:07:47,385 - molpy.reacter.bond_react - WARNING - Charge not conserved in bond/react template: sum(q_post) - sum(q_pre) = 0.278 e exceeds tolerance 1e-06 e.
```

反应前和反应后模板的原子数相同。离去基团中的原子在 `.map` 文件中标记为删除——LAMMPS 应用后模板拓扑后会将它们移除。

```python
print(f"pre:  {len(template.pre.atoms)} atoms")
print(f"post: {len(template.post.atoms)} atoms")
```

```text
pre:  23 atoms
post: 23 atoms
```

## 在合理密度下堆积

根据总分子量和目标密度算出盒子尺寸，然后让 Packmol 找出无重叠的摆放方案。27 个双官能团单体加 9 个三官能团单体最多可提供 81 个反应性端口。

```python
from molpy.pack import InsideBoxConstraint, Packmol

N_EO2, N_EO3 = 27, 9
TARGET_DENSITY = 1.1  # g/cm³ (无定形 PEO ≈ 1.1–1.2)

total_mass_g = (
    N_EO2 * sum(Element(a.get("element")).mass for a in eo2.atoms)
    + N_EO3 * sum(Element(a.get("element")).mass for a in eo3.atoms)
) / 6.022e23
box_length = ((total_mass_g / TARGET_DENSITY) * 1e24) ** (1 / 3)
```

```python
packer = Packmol(workdir=Path("04_output/packmol"))
constraint = InsideBoxConstraint(
    length=np.array([box_length] * 3),
    origin=np.zeros(3),
)
packer.def_target(eo2.to_frame(), number=N_EO2, constraint=constraint)
packer.def_target(eo3.to_frame(), number=N_EO3, constraint=constraint)

packed = packer(max_steps=20000, seed=42)
packed.box = mp.Box.cubic(length=box_length)
print(f"packed: {packed['atoms'].nrows} atoms in {box_length:.1f} Å box")
```

```text
packed: 990 atoms in 21.1 Å box
```

## 一次调用导出 data、力场和模板

`write_lammps_bond_react_system` 会收集堆积框架以及每个模板中的所有原子、键、角和二面角类型，构建统一的类型映射，将所有内容写入同一个目录。这样 `04.data`、`04.ff`、`rxn1_pre.mol` 和 `rxn1_post.mol` 中的数值类型 ID 就能互相一致。

### 模板有效性保证

`BondReactReacter` 按照 REACTER 协议（Gissinger 等, *Polymer* **128** (2017) 211–217, DOI: 10.1016/j.polymer.2017.06.038; Gissinger 等, *Macromolecules* **53** (2020) 9953–9961, DOI: 10.1021/acs.macromol.0c02012）和 LAMMPS [`fix bond/react`](https://docs.lammps.org/fix_bond_react.html) 的契约验证每个生成的模板：

- **等价关系**是反应前和反应后模板原子之间的双射。
- **InitiatorIDs** 恰好包含 2 个原子（成键锚原子），按确定性顺序写入（左侧锚原子在前）。若锚原子落在模板边界上，抛出 `ValueError` 并建议增大 `radius`。
- **边缘原子**在反应前和反应后的 `type` 和 `charge` 必须完全相同——若不匹配（通常因类型分配器对反应位点附近原子重新分配类型），抛出 `ValueError`；应增大 `radius`，让类型重新分配的壳层落在模板内部。
- **Improper 二面角**会传播到后模板中，确保 sp2 中心（酰胺、乙烯基、芳香环）在反应后仍保留其平面性项。
- **总电荷**（以元电荷为单位）在 `CHARGE_CONSERVATION_TOL = 1e-6` e 容差范围内检查是否守恒；违反时记录一条警告。

已知限制：可选的 `Constraints` 和 `ChiralIDs` 映射段不会被生成。

```python
from pathlib import Path

output_dir = Path("04_output")
output_dir.mkdir(exist_ok=True)

mp.io.write_lammps_bond_react_system(
    output_dir,
    packed,
    ff,
    templates={"rxn1": template},
)
```

该目录现在包含：堆积体系构型文件（`04.data`）、覆盖初始状态和两个模板中所有类型的力场系数文件（`04.ff`）、反应前和反应后分子模板（`rxn1_pre.mol` 和 `rxn1_post.mol`），以及原子等价、边缘和删除 ID 映射文件（`rxn1.map`）。

## LAMMPS 输入脚本执行五阶段协议

模拟按顺序经过五个阶段。阶段 1 用共轭梯度能量最小化消除堆积构型中的空间冲突。阶段 2 在 NVT 系综下将体系加热到 300 K，运行 5 ps，让动能在盒子允许弛豫之前分布均匀。阶段 3 切换到 1 atm 下的 NPT 系综再跑 5 ps，使密度达到平衡值。阶段 4 激活 `fix bond/react`：`stabilization yes` 关键字将新反应的原子放入单独的热浴组做短暂稳定，`molecule inter` 将反应限制在不同分子原子之间，防止分子内环化。阶段 5 清理收尾，根据新的键拓扑重新计算分子 ID，写入最终快照。下面脚本中 `run` 的长度有意缩短到几步，这样本页面几秒就能跑完——它生成的是有代表性的日志骨架，而非平衡后的网络；生产运行时请使用上述每阶段的推荐时长。

```python
lammps_script = """\
# ====================================================================
# PEO 交联网络 – OPLS-AA / fix bond/react
# ====================================================================
units           real
atom_style      full
boundary        p p p

read_data       04_output.data
include         04_output.ff

# -- 长程静电 --
kspace_style    pppm 1.0e-4

# -- OPLS-AA 1-4 缩放 --
special_bonds   lj 0.0 0.0 0.5 coul 0.0 0.0 0.5

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

# ====================================================================
# 阶段 1：能量最小化（消除堆积重叠）
# ====================================================================
thermo          100
thermo_style    custom step temp pe ke etotal press vol density
min_style       cg
minimize        1.0e-4 1.0e-6 2000 20000

reset_timestep  0

# ====================================================================
# 注意：运行步数已缩短（每阶段仅几步），以便本文档页面快速渲染；
# 实际交联运行请使用 5 / 5 / 10+ ps。
# 阶段 2：NVT 平衡（300 K）
# ====================================================================
variable step equal "step"
variable T    equal "temp"
variable rho  equal "density"
variable rxn1 equal "f_rxns[1]"

fix out all print 100 "${step} ${T} ${rho}" &
    file thermo_rxn.dat screen no &
    title "# step temp density"
velocity        all create 300.0 12345 dist gaussian

fix             nvt_eq all nvt temp 300.0 300.0 100.0
timestep        1.0
run             200
unfix           nvt_eq

# ====================================================================
# 阶段 3：NPT 平衡（300 K, 1 atm）
# ====================================================================
fix             equil all npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0
run             200
run             200
run             100
unfix           equil

# ====================================================================
# 阶段 4：使用 fix bond/react 进行反应性 MD（300 K NPT）
# ====================================================================
molecule        rxn1_pre rxn1_pre.mol
molecule        rxn1_post rxn1_post.mol

# bond/react: 每一步尝试，5 Å 截断，仅分子间反应
fix             rxns all bond/react stabilization yes npt_grp 0.03 &
                react rxn1 all 1 0.0 5.0 rxn1_pre rxn1_post rxn1.map prob 0.01 1234

fix             npt_grp_react npt_grp_REACT npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0

# -- 原子轨迹转储（OVITO 兼容） --
dump            traj all custom 200 traj.lammpstrj &
                id mol type x y z vx vy vz
dump_modify     traj sort id

# -- 键拓扑转储（OVITO：作为拓扑加载） --
compute         bnd all property/local batom1 batom2 btype
dump            bonds all local 200 bonds.dump c_bnd[1] c_bnd[2] c_bnd[3]
dump_modify     bonds colname 1 batom1 colname 2 batom2 colname 3 btype

# -- 热力学输出包含来自 fix bond/react 的反应计数 --
thermo          200
thermo_style    custom step temp pe ke etotal press density f_rxns[1]
run             1000

# ====================================================================
# 阶段 5：写入最终状态
# ====================================================================
write_data      final.data
"""
```

末尾的 `reset_mol_ids all` 命令根据当前键拓扑重新计算分子 ID。若交联成功，原始的 36 个分子将塌缩为少量连通组分。

`LAMMPSEngine` 将脚本包装为 `Script` 对象，在输出目录（所有数据文件和模板文件已放置在此）中运行 `lmp_serial`。

```python
from molpy.engine import LAMMPSEngine

script = mp.Script.from_text("run", lammps_script, language="other")
script.tags.add("input")

engine = LAMMPSEngine(executable="lmp_serial", check_executable=False)
proc = engine.run(
    script,
    workdir=output_dir,
    capture_output=True,
    check=False,
    timeout=600,
)

for line in proc.stdout.splitlines()[-20:]:
    print(line)
assert proc.returncode == 0, f"LAMMPS failed:\n{proc.stderr or proc.stdout[-1000:]}"
```

!!! tip "在 OVITO 中可视化"
    加载 `traj.lammpstrj` 作为主文件，再加载 `bonds.dump` 作为拓扑叠加层。键转储使用 OVITO 可直接识别的 `batom1`/`batom2`/`btype` 列。更多细节见 [OVITO 手册](https://www.ovito.org/manual/reference/file_formats/input/lammps_dump_local.html)。

## 验证输出

读回最终快照，检查交联是否确实形成。脚本末尾的 `reset_mol_ids` 命令根据键连通性重新分配分子 ID——数量从 36 下降就意味着网络已经形成。

```python
final = mp.io.read_lammps_data(output_dir / "final.data", atom_style="full")
n_atoms_final = final["atoms"].nrows
n_mols_final = len(set(final["atoms"]["mol_id"]))
print(f"final: {n_atoms_final} atoms, {n_mols_final} molecules (started with 36)")
```

```text
final: 891 atoms, 3 molecules (started with 36)
```

## 故障排除

**模板生成失败**

打印选中的位点和离去基团原子，看看选择器找到了什么：

```python
site = find_port(left, "$")
carbon = [a for a in find_neighbors(left, site) if a.get("element") == "C"][0]
print(f"site: {carbon.get('element')} name={carbon.get('name')}")
for nb in find_neighbors(left, carbon):
    print(f"  neighbor: {nb.get('element')} name={nb.get('name')}")
```

```text
site: C name=None
  neighbor: O name=None
  neighbor: C name=None
  neighbor: H name=None
  neighbor: H name=None
```

**堆积失败**

降低目标密度或单体数量。Packmol 需要足够空间来放置分子而不产生重叠。

**LAMMPS："Atom type affected by reaction is too close to template edge"**

增大 `BondReactReacter` 中的 `radius`。更大的半径能捕获更多局域环境，将模板边界推离类型发生变化的原子。

**LAMMPS：反应触发但分子数量保持不变**

后模板中缺少新的交联键。原因通常是没有将类型分配器传入 `reacter.run()`——新键没有力场类型，导出时被静默丢弃。务必传入 `typifier=typifier`。

**LAMMPS 拒绝模板**

检查 `.map` 文件是否包含有效的等价 ID：

```python
print((output_dir / "rxn1.map").read_text()[:500])
```

```text
# auto-generated map file for fix bond/react

23 equivalences
2 edgeIDs
3 deleteIDs

InitiatorIDs

2
14

EdgeIDs

6
18

DeleteIDs

1
7
19

Equivalences

1   1
2   2
3   3
4   4
5   5
6   6
7   7
8   8
9   9
10   10
11   11
12   12
13   13
14   14
15   15
16   16
17   17
18   18
19   19
20   20
21   21
22   22
23   23
```

另请参阅：[拓扑驱动组装](03_polymer_topology.md)、[多分散体系](05_polydisperse_systems.md)。
