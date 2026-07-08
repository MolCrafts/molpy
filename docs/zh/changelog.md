# 更新日志

本页记录了 MolPy 的各版本发布说明，按版本号从高到低排列。MolPy 与 molrs 共享同一版本号，两者配对发布。每个版本条目注明了所需的 `molcrafts-molrs` 版本。已标记的发布版本和可安装包见 [GitHub Releases](https://github.com/MolCrafts/molpy/releases) 页面。

## 0.5.1 - 2026-07-01

需配合 `molcrafts-molrs == 0.5.1`（从本版本起 MolPy 与 molrs 配对发布）。

### 新增

- `molpy.compute` 新增多项分析计算算子：角度/二面角/距离及组合分布函数、空间分布函数、Van Hove 相关函数 `G(r, t)`、Legendre 再定向相关、氢键检测、包含域/空穴/电荷分析的 radical Voronoi 分割、振动光谱（VDOS、红外、拉曼、VCD、ROA、共振拉曼）。
- `molpy.version.check_molrs_version()`：`import molpy` 时自动执行，若已安装的 `molcrafts-molrs` 版本不匹配则发出警告。

### 变更

- 文档改用 **Zensical** 构建，用户指南笔记本已预渲染为 Markdown。计算部分同时提供了教科书式指南和完整 API 参考。

## 0.5.0 - 2026-06-21

需配合 `molcrafts-molrs == 0.1.5`。

### 移除

- **移除 SMARTS GAFF 类型分类器。** 删除了 `molpy.typifier.GaffTypifier`（含内部 `_GaffAtomTypifier`）、`typifier/gaff.py` 模块及 `data/forcefield/gaff.xml`。GAFF 原子类型和 AM1-BCC 电荷现在完全经由 AmberTools 包装器（`antechamber` / `prepgen`）提供；保留了 41 个通用 SMARTS 匹配器测试。

### 修复

- **tip3p 水的 `theta0` 改为弧度制**，与 molrs 角度的内部弧度惯例一致。

### 新增

- PEO 聚合物电解质工作流示例。

## 0.4.2 - 2026-06-18

增量版本，无 API 重命名或破坏性变更。
需配合 `molcrafts-molrs == 0.1.4`。

### 新增

- **GROMACS TRR / XTC 和 DCD 轨迹读写器。** 新增 `read_trr_trajectory`、`read_xtc_trajectory`、`read_dcd_trajectory`，以及 `write_trr`、`write_xtc`——对 molrs 后端的轻量委派。

## 0.4.1 - 2026-06-14

基于 `0.4.0` 的维护版本。需配合 `molcrafts-molrs == 0.1.2`。

### 移除

- **移除 `molpy.legacy`。** 删除了基于纯 NumPy 的 `MSD` / `DisplacementCorrelation` 算子及 `molpy.legacy` 子模块。请改用 `molpy.compute` 中基于 molrs 的实现：`molpy.compute.MSD` 和 `molpy.compute.MCDCompute`。

### 修复

- **AMBER prmtop 读取器。** 原子连接性索引列（键/角/二面角的 `atomi`/`atomj`/`atomk`/`atoml`/`id`，以及原子的 `id`）改用无符号 `uint32` 输出，与 molrs 的 `UInt` 索引列（`get_uint`）保持一致；使用 `-1` 的掩码/哨兵列仍维持有符号类型。

### 内部

- 将 `molcrafts-molrs` 锁定为 `0.1.2`（之前为 `0.1.1`），同步更新了 CI 安装注释。
- 将三个几乎相同的 PDB 写入器必填字段测试合并为单个参数化测试。

## 0.4.0 - 2026-06-11

本版本引入了 **molrs Rust 后端**，作为 `Frame` / `Block` / `Box` / `compute` 的基础。同时完成了五阶段的 **builder/reacter 改造**：死代码整合、fix bond/react 序列化移至 io 层、REACTER 模板科学正确性修复、统一公共 API（附可执行文档）、构建循环性能优化（保持行为不变）。后续整合继续将更多功能下沉到 molrs 中：拓扑感知、轨迹存储、力场/势能模型、box/region 几何现在全部由 Rust 后端执行。**快速失败清扫**移除了所有已知的静默失败路径，并在 typifier/builder 层引入了 **CL&P 离子液体力场**及 **CL&Pol 可极化栈**的基础。
需配合 `molcrafts-molrs == 0.1.0`。

### Builder / Reacter

- **移除 `BondReactTemplate.write()` / `write_map()`。** `BondReactTemplate` 现为纯数据对象，所有 fix bond/react 序列化移至 io 层。写入完整反应系统（数据 + 力场 + 模板，类型 ID 跨文件统一）需使用 `mp.io.write_lammps_bond_react_system(workdir, frame, ff, templates=...)`；仅写入 `.map` 文件可使用 `mp.io.write_bond_react_map(template, stem)`。原有的单模板 `write()` 路径产生的模板局部类型 ID 与系统数据文件不一致，且无替代方案。
- **导入 molpy 子包不再预加载其他模块。** 顶级子模块（`molpy.io`、`molpy.engine`、`molpy.adapter`……）改为惰性加载（PEP 562）：`import molpy.reacter` 仅初始化 `core`（和 `potential`）。`mp.io.…` 属性访问和 `import molpy.io` 行为不变。`molpy.builder` / `reacter` / `pack` / `compute` 可作为惰性顶级属性访问（`mp.builder.…`）。
- **Builder/reacter 术语与 API 整合。** `polymer()` / `polymer_system()` 作为文档化的单调用入口点，`PolymerBuilder` + `Connector` 仍保留为分步 API。原 Agent 专用的 Tool 类移至 `molpy.builder.polymer.tools`（移出公开 `__all__`）；`ReactionPresets` / `ReactionPresetSpec` 现为公共扩展点。`ReactionPresetSpec.site_selector_*` 更名为 `anchor_selector_*`，`molpy.reacter.find_port_atom` 更名为 `find_port`。实验阶段不提供弃用垫片，详情见仓库根目录的 `CHANGELOG.md`。
- **REACTER 模板正确性。** `BondReactReacter` 后模板现在携带 impropers（sp2 平面性项在 `fix bond/react` 中正确处理）。`InitiatorIDs` 是确定性的且经过验证：恰好包含 2 个，且绝不位于模板边界上。边缘原子在反应前后检查类型和电荷是否一致。总电荷守恒检查（`CHARGE_CONSERVATION_TOL = 1e-6` e）。`run()` 不再修改调用方持有的 `left` / `right` 结构。
- **molrs 现在是强制依赖。** `molcrafts-molrs` 从可选依赖移入核心 `dependencies`。移除了 `molpy[molrs]` 额外安装键——安装 molpy 时始终附带 molrs。`Frame`、`Block`、`Box` 由 molrs 类型支持（并继承自它们），`compute` 算子直接转发到 Rust 内核，不再提供纯 Python 后备方案。
- **移除基于 RDKit 的计算节点。** 删除了 `molpy.compute.rdkit`（通过 `RDKitAdapter` 的 `Generate3D` / `OptimizeGeometry`）。`molpy.compute.Generate3D` 改为基于 molrs 的主干算子，接收 `Atomistic` 图并返回新的 3D 结构。RDKit 适配器（`molpy.adapter.rdkit`）作为**可选**外部后端保留；`rdkit` 仍是可选依赖，非常驻依赖。
- **`Frame` / `Block` 现在是 molrs 类型，而非 molpy 子类。** `molpy.core.frame.Frame` 就是 `molrs.Frame`，`Block` 就是 `molrs.Block`（轻量重导出）。Python 侧的对象列溢出（`_objects`）不复存在：列是**仅 numpy** 类型（float / int / bool / str）。写入对象 / `None` / 不规则列时会引发 `molrs.BlockDtypeError`，不再静默存储在 Python 侧。`frame.box` 返回 `molrs.Box`（携带 `is_free` / `style` / `volume()`）。molpy 原有更丰富的 box 几何通过 `molpy.Box` 保持可用，可通过 `Box.from_box(frame.box)` 升级。

### 类型分类器与力场

- **每个力场只暴露一个公共类型分类器类。** 原先的原子级/Atomistic 双层结构合并为每个力场一个完整的全流水线类：`OplsTypifier` 和 `GaffTypifier`（atom → pair → bond → angle → dihedral）。编排器基类 `ForceFieldAtomisticTypifier` 更名为 `ForceFieldTypifier`。各力场的原子类型分类器改为私有（`_OplsAtomTypifier` / `_GaffAtomTypifier`）。移除了 `Opls{Bond,Angle,Dihedral}Typifier` 类及 `GaffTypifier = GaffAtomisticTypifier` 别名。行为不变。
- **新增 CL&P 离子液体力场。** `ClpTypifier(OplsTypifier)` 通过完整流水线从内置 `clp.xml` 分类离子液体（咪唑阳离子 + BF4 / PF6 / NTf2 / dca 阴离子）。该文件生成自权威的 paduagroup CL&P `il.ff`，包含精确的电荷/LJ 参数和手工编写的环位置区分 SMARTS（CR/CW/NA）。通过现有的 OPLS 读取器读取，`oplsaa.xml` 未作修改。阳离子 `[C4C1im]+` 总电荷为 +1，各阴离子为 −1。
- **新增 CL&Pol 可极化力场基础。** `VirtualSite` / `DrudeParticle` / `MasslessSite` 是新的纯数据 `Atom` 子类，携带持久的 `vsite` 标记（标记存放在字段中，因此能随 molrs 存储持久化，`Atom.is_virtual` 读取它）。`VirtualSiteBuilder` ABC（复制 → 选择 → 构建位点 → 重新分配，不修改输入）附带 `DrudeBuilder`（CL&Pol 极化器，由 `alpha.ff` 极化率数据驱动）和 `Tip4pBuilder`（刚性 M 位点）。
- **CL&Pol scaleLJ SAPT epsilon 缩放。** `molpy.core.ops.scale_lj` 通过 SAPT 导出的因子 k_ij 缩放跨碎片 LJ epsilon（匹配 paduagroup/clandpol scaleLJ）。深拷贝 `ForceField` 后仅修改碎片间 pair epsilon，sigma 和电荷不变。碎片数据以 `clpol_fragments.ff` 形式提供；`FragmentScaling`、`compute_k_ij` 和 `load_fragment_scaling_data` 均为公共 API。
- 本开发周期内还编写了临时的纯 Python **Thole 和 Tang–Toennies 阻尼评估器**，并针对 paduagroup/clandpol 和 LAMMPS `pair_thole` / `pair_coul_tt` 进行了验证。但发布前它们因力场向 molrs 迁移而被取代——**不在** 0.4.0 的公共 API 中，后续将以 molrs 原生内核的形式回归。

### molrs 整合

- **力场模型完全迁移到 molrs 中；`molpy.potential` 变为外观模块。** `molpy.core.forcefield` 是原生 molrs `ForceField` / Style / Type 层级结构的轻量重导出（外加 `AtomisticForcefield` 别名和命名的专用 Style 类）。删除了 `potential/` 下的 Python 内核和能量计算代码。`molpy.potential[.bond|.angle|…]` 重导出基于 molrs 的 Style 类和 `Potentials`，用户无需直接导入 molrs。能量/力通过 `ff.to_potentials(frame).calc_energy(frame)` / `.calc_forces(frame)` 计算；`optimize.ForceFieldPotential` 包装了此流程（移除了 `potential_wrappers` 中各内核的包装器）。I/O 格式器改为按样式/内核名称分发（不再需要各内核的 `Type` 类），`def_type` 参数改为仅关键字参数。
- **拓扑感知改用 molrs 图内核，移除了单独的 igraph 引擎。** `get_topo` 的角度/二面角感知委托给 `molrs.Atomistic.generate_topology`；`get_topo_neighbors` / `get_topo_distances` 和 `extract_subgraph` 使用 molrs 的 BFS 及邻接内核。删除了 `core/topology.py` 及公开的 `molpy.Topology` / `molpy.core.Topology` 导出。`get_topo` **始终返回 `Atomistic`**（无标志位表示简单复制），修复了无标志路径可能泄漏原始图对象的潜在缺陷。igraph 仅在 SMARTS 类型分类器内部保留。关系枚举改用 molrs 权威来源 `relation_ids()`——Python 侧的 `_rel_handles` 镜像及其句柄范围探测启发式已移除。
- **`Trajectory` 改为继承 `molrs.Trajectory`。** 即时容器、惰性读取、LAMMPS/XYZ 轨迹解析等功能全部迁移到 molrs 中（`read_lammps_trajectory` / `read_xyz_trajectory`），删除了 molpy 中的重复读取器和 mmap 索引基础设施。molpy 保留了分割扩展（`SplitStrategy` / `TrajectorySplitter`）、拓扑/切片/映射便捷方法、XYZ 写入器和 HDF5 路径。`TimeIntervalStrategy` 读取原生 `.time` 数组（Python 的 `frame.metadata` 不会双向保留 molrs 存储）。
- **`Compute` 简化为普通类。** `__init__` 接收配置，`__call__` 接收输入，`dump()` 持久化结果。移除了单输入的 `_compute` 钩子、molexp 的 `execute()` / `input_key` / `output_key` 垫片、`Compute[InT, OutT]` 泛型，以及 17 个基于 molrs 的算子包装器中的废弃 `_compute` 存根。
- **Box 和区域几何委托给 Rust 内核。** `Box.make_fractional` / `make_absolute` / `isin` 转发到继承的 molrs `to_frac` / `to_cart` / `isin`；体积、长度/倾斜、包装及最小映像差异使用 molrs 属性和 `Box.wrap` / `Box.delta`。`SphereRegion` 和 `BoxRegion` / `Cube` 的点归属判断基于 `molrs.Sphere.contains` / `molrs.Cuboid.contains`——molpy 仅保留布尔代数 / `MaskPredicate` 层。移除了 Python 侧的 `_is_free` 标志：自由 box 状态派生自 molrs `cell_defined`，因此非周期性边界框（`from_bounds`）现在能正确报告一个具有体积/长度的真实晶胞。`Atomistic.scale` 和 `align` 改用 `molrs.scale` / `molrs.rotate`（删除了逐原子的 Rodrigues 循环）。每次下沉前均验证了正交和三斜结果与之前 NumPy 路径的一致性。
- **规范字段注册表改为使用 `molrs.fields`。** `molpy.core.fields` 不再定义并行的 `FieldSpec` / `FieldFormatter` 集，而是重导出统一的规范注册表（`molpy.core.fields.CHARGE` *就是* `molrs.fields.CHARGE`），仅在其上保留力场特定的 `ForceFieldFormatter`。这解决了三个注册表因长久未同步而漂移的问题。

### 快速失败

- **选择器在缺少列时抛出异常。** `AtomIndexSelector` / `ElementSelector` / `CoordinateRangeSelector` / `DistanceSelector` 现在对拼写错误的字段抛出 `KeyError`，指明缺少的列，不再静默匹配零个原子。
- **`ForceField.to_potentials()` 不再静默丢弃样式。** 合法为空的样式（无类型）被显式跳过；任何真正的失败（未知内核、类型缺少必需参数）都会传播。
- **类型分类器输入经过验证。** 无法解析的 SMARTS 模式现在会抛出异常，不再仅仅告警后丢弃（那会导致该原子类型静默未分类）；无效的元素符号或原子序数抛出 `ValueError`，不再降级为通配符匹配（`*` 仍保留用于显式无元素/无数值的情况）。
- **LAMMPS 数据读取器保留力场系数。** Pair/Bond/Angle/Dihedral/Improper Coeffs 部分现在存储在元数据 `ForceField` 上（位置键控，感知样式元数），因此读写可双向保留；之前解析后即丢弃。格式错误（非数值）的 coeff 行会抛出 `ValueError`，不再静默忽略。
- 删除了废弃的 `molpy.op` 包（未使用的几何辅助函数）。

### 新增

- `molpy.compute.NeighborList`：链接细胞邻居搜索（molrs 后端）。
- `molpy.compute.RDF`：针对单个或多个帧的径向分布函数。
- `molpy.Box` 继承自 `molrs.Box`，molpy box 可直接用于任何 molrs API，无需转换。
- molrs 分析功能以 molpy 算子形式暴露：`MSD`、`Cluster`、`ClusterCenters`、`CenterOfMass`、`GyrationTensor`、`InertiaTensor`、`RadiusOfGyration`、`Pca`、`KMeans`。
- `molpy.typifier.ClpTypifier` + 内置 `clp.xml`——基于 OPLS 流水线的 CL&P 离子液体力场（咪唑 + BF4/PF6/NTf2/dca）。
- `VirtualSite` / `DrudeParticle` / `MasslessSite` 原子、`Atom.is_virtual`、`VirtualSiteBuilder` / `DrudeBuilder` / `Tip4pBuilder`（CL&Pol Drude 极化器 + TIP4P M 位点），附带 `alpha.ff` 极化率数据。
- `molpy.core.ops.scale_lj`（+ `FragmentScaling`、`compute_k_ij`、`load_fragment_scaling_data`、`clpol_fragments.ff`）——CL&Pol SAPT 跨碎片 LJ epsilon 缩放。
- `UnitSystem.register_preset(name, base_units, *, overwrite=False)`：注册可通过 `preset()` 使用的自定义 LAMMPS 风格单位预设；预设表不再是封闭字典。

### 变更

- `compute.mcd` 和 `compute.pmsd` 改用 molrs `Box.delta(minimum_image=True)` 计算最小映像位移，公开签名不变。

### 迁移

- 将 `pip install "molcrafts-molpy[molrs]"` 改为 `pip install molcrafts-molpy`。
- `from molpy.compute.rdkit import Generate3D` 改为 `from molpy.compute import Generate3D`（基于 molrs，`Atomistic -> Atomistic`）。如需使用 RDKit 适配器流程，改用 `from molpy.adapter import Generate3D`。
- 构建 `Block` 列时需使用 NumPy 可表示的数据：将 `np.array([...], dtype=object)` 字符串列替换为原生的 `np.array([...])`（NumPy 会自动推断出 `U` dtype）。稀疏的逐实体属性不能再以 `None` 填充到列中——请使用类型化默认值或直接省略该列。`Atomistic.to_frame()` / `CoarseGrain.to_frame()` 现在会丢弃无法用 NumPy 表示的列（例如 CG bead 的不规则 `atoms` 映射），而非生成对象数组。
- `molpy.Topology` / `molpy.core.Topology` 已移除，`get_topo()` 始终返回 `Atomistic`。k 跳图查询请改用 `get_topo_neighbors` / `get_topo_distances`（基于 molrs BFS）。
- 类型分类器：`OplsAtomTypifier` / `GaffAtomTypifier` / `Opls{Bond,Angle,Dihedral}Typifier` / `GaffAtomisticTypifier` 统一替换为 `OplsTypifier` / `GaffTypifier`。自定义编排器应继承 `ForceFieldTypifier`（原名 `ForceFieldAtomisticTypifier`）。
- 势能：移除了 `molpy.potential` 下的 Python 内核类。从相同路径导入基于 molrs 的 Style 类，通过 `ff.to_potentials(frame).calc_energy(frame)` 评估。参数名称遵循 molrs 命名（`k` 而非 `k0`）。使用 `ff.get_style(category, name)` 查找样式，`def_type` 参数改为仅关键字参数。
- `Compute` 子类：配置放入 `__init__`，数据放入 `__call__`。移除了 `execute()` / `input_key` / `output_key` molexp 垫片和 `_compute` 钩子。
- 先前依赖选择器在缺失列时返回空掩码、依赖不可解析的 SMARTS 被跳过、依赖 `to_potentials()` 忽略损坏样式的代码，现在必须处理抛出的 `KeyError` / `ValueError`（这些均属隐藏性错误，也可直接修复输入）。

有关详细信息，请参阅 [molrs 后端开发人员指南](developer/molrs-backend.md)。
