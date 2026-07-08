## 词汇表

MolPy 核心术语速查。每个条目都链接到深入介绍该概念的页面。

### 数据结构

**Atomistic（原子结构）**
:   一张可编辑的分子图，原子是节点，化学键是边。构建阶段使用——加原子、除离去基团、查邻域。参见[原子结构与拓扑](01_atomistic_and_topology.md)。

**Entity（实体）**
:   原子和珠子的基类。类似字典：方括号读写属性。用身份做哈希（`id(self)`），不做值相等判断。

**Atom（原子）**
:   表示单个原子的 `Entity` 子类。可携带任意键值属性（`element`、`charge`、`type` 等）。

**Bead（珠子）**
:   表示粗粒化位点的 `Entity` 子类。

**Link（连接）**
:   拓扑连接的基类。持有一个有序的 `Entity` 端点元组。子类包括：`Bond`、`Angle`、`Dihedral`、`Improper`。

**Struct（结构容器）**
:   聚合实体和连接的基类。子类包括：`Atomistic`、`CoarseGrain`。

**Topology（拓扑）**
:   molrs Rust 内核从 `Atomistic` 键图派生的视图。`get_topo()` 返回新的 `Atomistic`，其中已从化学键识别出角度和二面角；`get_topo_neighbors()` / `get_topo_distances()` 回答 k 跳图查询。没有独立的拓扑类。参见[原子结构与拓扑](01_atomistic_and_topology.md)。

**Block（数据块）**
:   字符串键到 NumPy 数组的列式表格，所有列行数相同。`Frame` 内部用它存储原子、化学键、角度等。参见[数据块与数据帧](02_block_and_frame.md)。

**Frame（数据帧）**
:   一组命名的 `Block` 对象加自由格式元数据，代表一个完整的系统快照。I/O 的通用交换对象。参见[数据块与数据帧](02_block_and_frame.md)。

**Box（盒子）**
:   由 3×3 晶格矩阵和周期性边界条件定义的模拟单元。支持包裹、最小镜像距离和坐标变换。参见[盒子与周期性](03_box_and_periodicity.md)。

**Trajectory（轨迹）**
:   一个有序的 `Frame` 序列。支持通过生成器和 `map` 变换做惰性访问。参见[轨迹](05_trajectory.md)。


### 力场

**ForceField（力场）**
:   保存分子系统全部样式、类型和参数的容器。可手动创建，也可从 XML/LAMMPS/AMBER 文件加载。

**Style（样式）**
:   力场中的一种相互作用家族——例如 harmonic 键或 lj126/cut 对，定义期望的参数集合。子类包括：`BondStyle`、`AngleStyle`、`DihedralStyle`、`PairStyle`。

**Type（类型）**
:   一个样式下的具体参数记录。例如键类型 "CT-OH" 携带 `k=320.0` 和 `r0=1.41`。子类包括：`AtomType`、`BondType`、`AngleType`、`DihedralType`、`PairType`。

**Potential（势能）**
:   力场样式和类型的数值实现，可直接用于能量/力计算。通过 `ff.to_potentials()` 生成（返回延迟计算的 `Potentials`），然后对已类型化的 `Frame` 执行 `pots.calc_energy(frame)` / `pots.calc_forces(frame)` 求值；内核跑在 molrs Rust 扩展中。参见[力场](04_force_field.md)。


### 模块

**Parser（解析器）**
:   将字符串表示（SMILES、SMARTS、BigSMILES、CGSmiles）转成 MolPy 结构。参见[解析化学](../user-guide/01_parsing_chemistry.md)。

**Reacter（反应器）**
:   在两个 `Atomistic` 对象的指定端口原子之间建立连接，移除离去基团并形成新化学键，从而执行化学反应。参见[逐步聚合物构建](../user-guide/02_polymer_stepwise.md)。

**Port（端口）**
:   原子上的标记（`<`、`>` 或 `$`），表示该原子是聚合反应的关键接点。

**Typifier（类型化器）**
:   通过 SMARTS 模式匹配为原子、化学键、角度和二面角分配力场类型。子类包括：`OplsTypifier`、`ClpTypifier`、`MMFFTypifier`、`PairTypifier`。（GAFF 原子类型*不是* Typifier——它们来自 AmberTools/antechamber；参见[AmberTools 集成](../user-guide/13_ambertools_integration.md)。）参见[力场类型化](../user-guide/06_typifier.md)。

**Selector（选择器）**
:   可组合的谓词，按元素、类型、坐标范围或距离筛选 `Block` 中的原子。支持 `&`、`|`、`~` 组合。参见[选择器](06_selector.md)。

**Wrapper（包装器）**
:   将外部可执行程序（antechamber、tleap、Packmol）作为子进程运行并捕获结果。跨越执行边界。参见[包装器与适配器](07_wrapper_and_adapter.md)。

**Adapter（适配器）**
:   在 MolPy 对象与另一个库的内存对象（RDKit、OpenBabel）之间做转换。跨越表示边界。参见[包装器与适配器](07_wrapper_and_adapter.md)。


### 命名约定

**atomi / atomj / atomk / atoml**
:   `Frame` 和 `Block`（数据交换层）中使用的整数原子索引。总是从 0 开始。不存对象引用。

**itom / jtom / ktom / ltom**
:   `Entity` 级别拓扑（Bond、Angle、Dihedral）中使用的原子对象引用。不存整数。参见[命名约定](naming-conventions.md)。


### 计算术语

以下缩写用于[计算](../compute/index.md)分析中。

**RDF** — 径向分布函数 `g(r)`
:   距参考原子 `r` 处找到邻居的概率，相对于理想气体。参见[结构分析](../compute/structure.md)。

**MSD** — 均方位移
:   `⟨|r(t) − r(0)|²⟩`；其斜率给出自扩散系数。参见[扩散与离子输运](../compute/transport.md)。

**VACF** — 速度自相关函数
:   `⟨v(0)·v(t)⟩`；积分（Green–Kubo）得扩散系数，FFT 得 VDOS。参见[速度自相关与 VDOS](../compute/vacf.md)。

**VDOS** — 振动状态密度
:   原子运动的谱密度，`∝ FFT[VACF]`。参见[速度自相关与 VDOS](../compute/vacf.md)。

**MCD** — 平均位移相关（区别扩散）
:   不同物种之间的互相关位移——扩散中超越单粒子 MSD 的*区别*部分。参见[扩散与离子输运](../compute/transport.md)。

**PMSD** — 极化均方位移
:   集体电荷偶极子的 MSD；离子电导率的 Einstein 途径。参见[介电谱](../compute/dielectric.md)。

**SDF** — 空间分布函数
:   参考框架周围邻居的三维密度（角度结构，不只是径向）。参见[分布函数](../compute/distributions.md)。

**CDF** — 组合分布函数
:   两个几何观测量（例如距离 × 角度）的联合直方图。参见[分布函数](../compute/distributions.md)。

**PMFT** — 平均力与扭矩势
:   相对位置/取向坐标上的自由能 `−k_BT ln g`。参见[分布函数](../compute/distributions.md)。

**ROA / VCD** — 拉曼光学活性 / 振动圆二色性
:   由极化率/磁偶极响应的相关函数导出的手性振动光谱。参见[振动光谱](../compute/spectra.md)。
