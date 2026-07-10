# 指南

每篇指南围绕一个具体的建模任务展开，从输入到可运行的模拟输出——这是手册的操作部分。[示例画廊](../getting-started/examples.md)收录了其中若干篇的浓缩版，方便快速上手；指南本身则走完整叙述。不熟悉的术语，[数据模型教程](../tutorials/index.md)里有定义。

部分指南用高分子做示例——那是为了展示能力，不是限定范围。链增长、交联、多分散性这些场景刚好覆盖了 MolPy 编辑机制的方方面面；同样的操作手法也适用于任何复杂分子系统。


## 基础

- [化学结构解析](01_parsing_chemistry.md) — 将 SMILES、SMARTS、BigSMILES、CGSmiles 字符串转换成 `Atomistic` 结构


## 链与网络构建

- [组装](02_assembly.md) — 一个 `GraphAssembler` 完成长链、交联和成环三件事；差别只在 `Selector` 如何配对反应位点
- [多分散体系](05_polydisperse_systems.md) — 分子量分布采样、原子级链构建与盒子填充


## 参数化

- [力场类型分配](06_typifier.md) — 基于 SMARTS 的原子类型分配与力场参数解析


## 几何与填充

- [3D 构象生成](07_conformers.md) — 为解析或构建出的结构嵌入化学上有效的三维坐标
- [几何优化](08_geometry_optimization.md) — 运行力场驱动的结构最小化并读取优化报告
- [体系填充](09_packing.md) — 通过 Packmol 后端在几何约束下将分子填充进模拟盒子
- [极化与虚位点模型](10_polarizable.md) — 用虚位点构建器协议实现 Drude 壳层和 TIP4P M 位点


## 导出与引擎

- [文件 I/O](11_io.md) — 分子数据、轨迹、日志文件与力场格式的读写
- [模拟引擎](12_engine.md) — 为 LAMMPS、CP2K、OpenMM 生成输入文件并直接从 Python 驱动计算


## 工具与生态

- [AmberTools 集成](13_ambertools_integration.md) — 驱动 antechamber、parmchk2、tleap，完成完整的电解质准备流程
- [Moltemplate CLI](14_moltemplate_cli.md) — moltemplate `.lt` 文件与 MolPy 体系之间的双向转换
- [MCP 套件](15_mcp.md) — 向模型上下文协议（MCP）智能体暴露 MolPy 符号与文档
