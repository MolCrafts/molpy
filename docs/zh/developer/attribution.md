# 第三方贡献与许可

MolPy 以 [BSD 3-Clause 许可证](https://github.com/MolCrafts/molpy/blob/master/LICENSE) 发布
（Copyright © 2024–2025, Roy Kid）。以下项目中的代码和参考数据是 MolPy 移植或改编的来源。
数值计算内核通过 **molrs** Rust 后端执行，其归属见
[molrs `docs/attribution.md`](https://github.com/MolCrafts/molrs/blob/master/docs/attribution.md)；
本页只列出 MolPy Python 层复用的部分。

> 通过子进程调用或库导入方式使用的外部工具（Packmol、AmberTools、LAMMPS、CP2K、RDKit、OpenBabel 等）
> 属于运行时依赖，不在本页范围内。

## 移植/改编 — 宽松许可证

| 项目 | SPDX | 版权 | 使用位置 | 上游 |
|---|---|---|---|---|
| **foyer** | `MIT` | © 2015 Vanderbilt University | `typifier.graph` (`SMARTSGraph`)、`typifier.matcher`、`typifier.atomistic`（OPLS SMARTS 原子类型引擎）；`data/forcefield/oplsaa.xml`、`tip3p.xml` | [mosdef-hub/foyer](https://github.com/mosdef-hub/foyer) |
| **OpenMM** | `MIT` (`openmm/app/element.py`) | Stanford University and the Authors | `core.element` — 元素名称/符号/质量表 | [openmm/openmm](https://github.com/openmm/openmm) |
| **moltemplate** | `MIT` | © 2013 Andrew Jewett, UC Santa Barbara | `parser.moltemplate.*`、`cli.moltemplate` — `.lt` 格式读写器及 `ltemplify` | [jewettaij/moltemplate](https://github.com/jewettaij/moltemplate) |
| **tame** | `BSD-3-Clause` | © Yunqi Shao | `compute.mcd`、`compute.pmsd`、`compute.time_series`、`compute.jacf`、`compute.onsager`、`compute.persist` | [yqshao-archive/tame](https://github.com/yqshao-archive/tame)（已归档） |

> `tame` 在 `pyproject.toml` 中声明 `license = "BSD-3-Clause"`，未附带独立的
> `LICENSE` 文件。该项目来自邵云奇（Yunqi Shao），目前已归档。

## 捆绑的参数数据

| 数据文件 | 来源 | 许可证 |
|---|---|---|
| `data/forcefield/oplsaa.xml`、`tip3p.xml` | 来自 **foyer** 的 OPLS-AA / TIP3P 力场 XML | `MIT`（foyer） |
| `data/forcefield/clp.xml` | CL&P 离子液体力场，来自 **paduagroup/clandp**（`il.ff`）的子集 | 学术许可；DOI 10.1021/jp0362133 |
| `data/forcefield/alpha.ff`、`clpol_fragments.ff` | **paduagroup/clandpol** CL&Pol 可极化数据（A. Padua, K. Goloviznina） | 学术许可；DOI 10.1021/acs.jctc.9b00689 |

## molrs 后端执行的数值内核

RDF、Steinhardt/hexatic/nematic 序参数、radical Voronoi、
结构因子和 PMFT 等分析计算均在 molrs 中完成。molrs 后端移植了
**freud**（`BSD-3-Clause`）、**voro++**（`BSD-3-Clause-LBNL`）和 **RDKit**
（`BSD-3-Clause`，MMFF）。详见
[molrs 贡献说明文件](https://github.com/MolCrafts/molrs/blob/master/docs/attribution.md)。

## 已实现的规范（引用，非许可）

`parser/grammar/` 和 `parser/smiles/grammars/` 中的文法文件实现的是公开的
**符号规范**，而非其他项目的代码：

| 规范 | 引用来源 | 实现位置 |
|---|---|---|
| **OpenSMILES** | opensmiles.org 社区规范 | `smiles.lark`、`base.lark` |
| **Daylight SMILES/SMARTS** | Daylight Theory Manual（© Daylight C.I.S. — 专有文档，仅作引用） | `smarts.lark` |
| **BigSMILES** | Lin 等，*ACS Cent. Sci.* **5**, 1523 (2019)，DOI 10.1021/acscentsci.9b00476 | `bigsmiles.lark` |
| **G-BigSMILES** | generative-BigSMILES 扩展 | `gbigsmiles.lark`、`gbigsmiles_new.lark` |
| **CGSmiles** | Grünewald 等 (2025) | `cgsmiles.lark` |
