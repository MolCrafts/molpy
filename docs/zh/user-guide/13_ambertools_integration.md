# PEO-LiTFSI 与 AmberTools 集成

用 antechamber 做离子参数化、tleap 生长 PEO 链、Packmol 在目标密度下组装 PEO-LiTFSI 电解质体系——一个由 MolPy 驱动的完整 AmberTools 工作流。

!!! warning "外部依赖"
    本指南需要 **AmberTools**（通过 conda 安装）、**RDKit** 和 **Packmol**。三个工具都必须装上，确保能跑。没有 AmberTools，此页面上的任何代码都无法运行。

??? note "设置 AmberTools"
    在专用 conda 环境中安装 AmberTools：

    ```bash
    conda create -n AmberTools25 -c conda-forge ambertools=25
    conda activate AmberTools25
    # 验证安装
    which antechamber   # 应打印路径
    which tleap         # 应打印路径
    ```

    MolPy 的包装类运行命令时会自动激活 conda 环境，不用在 shell 里一直开着。代码里的 `env="AmberTools25"` 参数指定包装器用哪个环境。

    用不同环境名时，把 `"AmberTools25"` 换成自己的。

## 工作流概览

工作流先用 Amber 小分子标准流程（antechamber、parmchk2、tleap）对 TFSI 阴离子做参数化。Li⁺ 单独处理，非键参数从 Åqvist (1990) 拿来直接写进 frcmod。两种离子都搞定之后，用 `AmberPolymerBuilder`（内部包装了 prepgen 和 tleap）建 PEO 链。接着合并各力场，用 Packmol 按目标密度放好分子，最后导出 LAMMPS 格式。


## Antechamber 为 TFSI 分配 GAFF 类型和 BCC 电荷

Amber 小分子工作流为：antechamber（分配类型 + 电荷）→ parmchk2（缺失参数）→ tleap（拓扑 + 坐标）。

```python
from pathlib import Path
import molpy as mp
from molpy.adapter import RDKitAdapter
from molpy.io.writers import write_pdb
from molpy.wrapper import AntechamberWrapper, Parmchk2Wrapper, TLeapWrapper

output_dir = Path("07_output")
ions_dir = output_dir / "ions"
ions_dir.mkdir(parents=True, exist_ok=True)

# 从 SMILES 创建 TFSI 并生成 3D 坐标
tfsi = mp.parser.parse_molecule("O=S(=O)(C(F)(F)F)[N-]S(=O)(=O)C(F)(F)F")
tfsi = RDKitAdapter(internal=tfsi).generate_3d(add_hydrogens=False, optimize=True)

# 写入 PDB 作为 antechamber 输入
write_pdb(ions_dir / "tfsi.pdb", tfsi.to_frame())
```

```python
conda_env = "AmberTools25"

# 步骤 1：antechamber — 分配 GAFF 类型和 BCC 电荷
ac = AntechamberWrapper(
    name="antechamber", workdir=ions_dir, env=conda_env, env_manager="conda"
)
ac.atomtype_assign(
    input_file=(ions_dir / "tfsi.pdb").absolute(),
    output_file=(ions_dir / "tfsi.mol2").absolute(),
    input_format="pdb",
    output_format="mol2",
    charge_method="bcc",
    atom_type="gaff2",
    net_charge=-1,
)

# 步骤 2：parmchk2 — 生成缺失参数
parmchk2 = Parmchk2Wrapper(
    name="parmchk2", workdir=ions_dir, env=conda_env, env_manager="conda"
)
parmchk2.run(args=["-i", "tfsi.mol2", "-o", "tfsi.frcmod", "-f", "mol2", "-s", "gaff2"])

# 步骤 3：tleap — 生成 prmtop 和 inpcrd
leap_script = """source leaprc.gaff2
TFSI = loadmol2 tfsi.mol2
loadamberparams tfsi.frcmod
saveamberparm TFSI tfsi.prmtop tfsi.inpcrd
quit
"""
(ions_dir / "tfsi_leap.in").write_text(leap_script)

tleap = TLeapWrapper(name="tleap", workdir=ions_dir, env=conda_env, env_manager="conda")
tleap.run(args=["-f", "tfsi_leap.in"])
```


## Li⁺ 无需电荷计算——文献参数直接写入 frcmod 文件

Li⁺ 没有键合项，也不用算部分电荷，跳过 antechamber。Åqvist (1990) 的非键参数直接写进 frcmod，再用 tleap 造 prmtop。

**Li⁺ 非键参数**——Åqvist (1990)，J. Phys. Chem. 94, 8021–8024，DOI: 10.1021/j100384a009。
这些参数针对水合自由能拟合得到，是聚合物电解质 GAFF 模拟中的标准选择。

| 参数 | 值 |
|-----------|-------|
| Rmin/2    | 1.137 Å |
| ε         | 0.0183 kcal/mol |

```python
from molpy.io import read_amber

li_dir = output_dir / "li"
li_dir.mkdir(parents=True, exist_ok=True)

# 写入 Åqvist (1990) frcmod — NONBON 使用 Rmin/2 和 epsilon
li_frcmod = """Li+ Aqvist 1990 parameters
MASS
LI    6.941               0.0000000

BOND

ANGLE

DIHE

IMPROPER

NONBON
  LI        1.137        0.0183

"""
(li_dir / "li.frcmod").write_text(li_frcmod)

# 单个 Li+ 原子的最小 mol2 文件（净电荷 = +1）
li_mol2 = """@<TRIPOS>MOLECULE
LIT
 1 0 0 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
      1 LI          0.0000    0.0000    0.0000 LI    1  LIT      1.000000
@<TRIPOS>BOND
"""
(li_dir / "li.mol2").write_text(li_mol2)

# tleap：生成 Li+ 的 prmtop
li_leap = """source leaprc.gaff2
loadamberparams li.frcmod
LIT = loadmol2 li.mol2
saveamberparm LIT li.prmtop li.inpcrd
quit
"""
(li_dir / "li_leap.in").write_text(li_leap)

tleap_li = TLeapWrapper(
    name="tleap", workdir=li_dir, env=conda_env, env_manager="conda"
)
tleap_li.run(args=["-f", "li_leap.in"])

li_frame, li_ff = read_amber(li_dir / "li.prmtop", li_dir / "li.inpcrd")
print(
    f"Li+: {li_frame['atoms'].nrows} atom, charge={li_frame['atoms']['charge'][0]:.1f}"
)
```


## 三种单体变体定义链的起始、内部和末端

BigSMILES 的端口标记（`[>]` 和 `[<]`）定义连接点。两个端口都有的单体是内部重复单元，只有一个端口的是端帽。构建器根据端口注释决定生成哪种 prepgen 变体（HEAD / CHAIN / TAIL）。

```python
def parse_monomer_3d(bigsmiles):
    mol = mp.parser.parse_monomer(bigsmiles)
    return RDKitAdapter(internal=mol).generate_3d(add_hydrogens=True, optimize=True)


# 头帽：仅有 < 端口 → 链的起始
me_head = parse_monomer_3d("{[][<]C[]}")

# 链单元：同时有 < 和 > → 内部重复单元
eo_chain = parse_monomer_3d("{[][<]COC[>][]}")

# 尾帽：仅有 > 端口 → 链的末端
me_tail = parse_monomer_3d("{[]C[>][]}")

library = {"MeH": me_head, "EO": eo_chain, "MeT": me_tail}
```


## AmberPolymerBuilder 在内部运行完整的 Amber 流水线

`AmberPolymerBuilder` 把单体库、连接器规则和 Amber 工具链（prepgen + tleap）打包成一个构建器，能直接生成完全参数化的链。不同链长会把 Amber 中间文件写进 `work_dir` 下各自的子目录，避免冲突。

```python
from molpy.builder.polymer.ambertools import AmberPolymerBuilder

polymer_dir = output_dir / "polymer"
polymer_dir.mkdir(exist_ok=True)

builder = AmberPolymerBuilder(
    library=library,
    force_field="gaff2",
    charge_method="bcc",
    env="AmberTools25",
    env_manager="conda",
    work_dir=polymer_dir,
)

result = builder.build("{[#MeH][#EO]|10[#MeT]}")
```

`AmberPolymerBuilder.build()` 在内部运行 antechamber、parmchk2、prepgen 和 tleap。结果包含聚合物 Frame、ForceField 以及中间 Amber 文件的路径。

```python
peo_frame = result.frame
peo_ff = result.forcefield
print(f"PEO 10-mer: {peo_frame['atoms'].nrows} atoms")
```


## 打包前合并三个力场可防止类型冲突

选择打包之前而非之后合并力场，是因为 Packmol 只管坐标，不关心力场类型。要是两个组分用了同名的原子类型但参数不同，打包后再合并会静默覆盖掉其中一个。先合并就能在生成坐标前把类型名冲突暴露出来。

```python
import numpy as np
from molpy.io import read_amber
from molpy.pack import Packmol, InsideBoxConstraint

# 从第一阶段生成的 Amber 文件中读取 TFSI
tfsi_frame, tfsi_ff = read_amber(
    ions_dir / "tfsi.prmtop",
    ions_dir / "tfsi.inpcrd",
)

# 合并所有三个力场：PEO + TFSI + Li+
combined_ff = peo_ff.merge(tfsi_ff).merge(li_ff)

# 打包体系
box_size = 60.0
packer = Packmol(workdir=output_dir / "packmol")
constraint = InsideBoxConstraint(length=[box_size] * 3, origin=[0.0] * 3)
packer.def_target(peo_frame, number=3, constraint=constraint)
packer.def_target(li_frame, number=10, constraint=constraint)
packer.def_target(tfsi_frame, number=10, constraint=constraint)

system = packer(max_steps=20000, seed=12345)
system.box = mp.Box.cubic(box_size)
```


## 导出时跳过 pair_style，因为长程静电需要在脚本中设置

```python
from molpy.io.writers import write_lammps_data, write_lammps_forcefield

lammps_dir = output_dir / "lammps"
lammps_dir.mkdir(exist_ok=True)
write_lammps_data(lammps_dir / "system.data", system, atom_style="full")
write_lammps_forcefield(lammps_dir / "system.ff", combined_ff, skip_pair_style=True)
```

`skip_pair_style=True` 省略力场文件中的 `pair_style` 行。用 kspace（长程静电）时必须如此，`pair_style` 只能由模拟脚本设置，力场文件里不该写。


## 故障排除

| 症状 | 检查项 |
|---------|-------|
| Antechamber 失败 | 确认 PDB 的原子名称正确，ID 不重复 |
| TFSI 电荷错误 | 使用 `charge_method="bcc"` 并确认 `-nc -1` |
| tleap 对 Li⁺ 失败 | 确认 mol2 原子类型（`LI`）与 frcmod 的 NONBON 条目一致 |
| 聚合物构建失败 | 检查单体 SMILES 中的端口标记 |
| 力场合并冲突 | 检查 PEO 与 TFSI 的原子类型名称是否冲突 |
| 打包失败 | 增大盒子尺寸或减少分子数量 |

参见：[力场类型化](06_typifier.md)、[Wrapper 与 Adapter](../tutorials/07_wrapper_and_adapter.md)。
