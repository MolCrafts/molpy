# Compute

**Compute** 层从 `Trajectory` 或 `Frame` 计算物理观测量，涵盖结构分布、动力学关联和光谱。所有分析方法共用一套模式，学会一个就能通晓全部。

## Compute → Result 模式

每种分析都是一个可配置的小对象，构建一次后*调用*到数据上。返回带类型的 `Result` 对象——不是裸元组——输出自带含义，一眼就能看懂。部分分析（`RDF`、密度、序参数）还需要 `NeighborList` 做空间查询，构造方式与分析方法相同，随后跟帧一起传进去：

```python
from molpy.compute import NeighborList, RDF

nlist  = NeighborList(cutoff=10.0)(frame)   # 1. 空间查询（成对邻居）
rdf    = RDF(n_bins=100, r_max=10.0)        # 2. 配置分析
result = rdf([frame], [nlist])              # 3. 调用于数据 -> 带类型 RDFResult
result.rdf, result.bin_centers              # 4. 读取自描述字段
```

不需要成对距离的分析（`MSD`、分布函数等）直接调用帧就行——还是先配置再调用。

数值密集的内核（自相关、FFT、谱前因子）在 `molrs` 中用 Rust 实现；MolPy 类负责数据提取、周期镜像解卷和向量化组装，物理计算则交给底层处理。

!!! note "单位"
    Compute 内核使用 LAMMPS *real* 单位：长度 Å、电荷 $e$、时间 ps、
    体积 Å³、温度 K、角频率 rad·ps⁻¹。GROMACS 轨迹以原生 nm 读取
    —— 分析前将长度乘以 10。

## 可用分析方法

| 方法 | 类 / 入口点 | 返回 | 测量量 |
|--------|---------------------|---------|----------|
| **介电光谱** | `DielectricSusceptibility` | `DielectricSusceptibilityResult` | $\varepsilon^*(\omega)$, $\varepsilon(0)$, $\sigma$ |
| **离子电导率** | `IonicConductivity` | `ConductivityResult` | $\sigma$ (S/m)，通过 Einstein-Helfand 方法 |
| 自相关 | `ACFAnalyzer` | `ACFResult` | 时间关联 $C(t)$ |
| 时间 → 频率 | `SpectralAnalyzer` | `SpectralResult` | 加窗频谱 |
| 平均位移关联 | `MCDCompute` | `MCDResult` | 每组的扩散 / MSD |
| 极化 MSD | `PMSDCompute` | `PMSDResult` | 集体电荷输运 |
| Onsager 系数 | `Onsager` | `OnsagerResult` | $L_{ij}$ 集体位移互关联 |
| 电流-ACF 电导率 | `JACF` | `JACFResult` | $\sigma$ (S/m)，通过 Green-Kubo $\langle J(0)\cdot J(t)\rangle$ |
| 成对持久性 | `Persist` | `PersistResult` | 驻留时间 / 存活 $C(\tau)$ |
| 径向分布 | `RDF` | 结构 $g(r)$ | 成对结构 |
| 静态结构因子 | `StaticStructureFactorDebye` | $S(k)$ | 倒空间结构 |
| 均方位移 | `MSD` | 时间序列 | 单粒子扩散 |
| 速度自相关 (VACF) | `compute_acf`, `PowerSpectrum` | ACF / VDOS | 速度记忆、Green–Kubo 扩散、振动模式 |
| 近邻列表 | `NeighborList` | 成对列表 | 截断近邻查询 |
| 局部 / 网格密度 | `LocalDensity`, `GaussianDensity` | 密度场 | 数密度场 |
| 序参数 | `Steinhardt`, `Hexatic`, `Nematic`, `SolidLiquid` | 每个粒子的序参量 | 结晶度 / 相 / 取向 |
| 键取向图 | `BondOrder` | $(\theta,\phi)$ 直方图 | 局部键合几何 |
| 平均力势与力矩 | `PMFTXY` | 自由能场 | 取向分辨 PMF |
| 形状描述符 | `RadiusOfGyration`, `GyrationTensor`, `InertiaTensor`, `CenterOfMass` | 每帧张量/标量 | 分子形状 |
| 聚类 / 分解 | `Cluster`, `ClusterCenters`, `Pca`, `KMeans` | 标签 / 成分 | 分组与降维 |
| 几何分布 | `DistanceDistribution`, `AngleDistribution`, `DihedralDistribution` | $p(r)$, $p(\theta)$, $p(\phi)$ | 键角 / 二面角结构 |
| 联合分布 | `CombinedDistribution` | N 维直方图 | 关联可观测量（联合分布函数, CDF） |
| 空间分布 | `SpatialDistribution` | 体固定密度 | 三维取向分辨结构（空间分布函数, SDF） |
| Van Hove 关联 | `VanHove` | $G(r,t)$ | 时间分辨结构 / 动力学 |
| 重取向 TCF | `LegendreReorientation` | $C_1(t)$, $C_2(t)$ | 向量重取向时间 |
| 氢键 | `HBonds`, `HBondCriterion` | 每帧键列表 | 氢键网络与计数 |
| 自由基 Voronoi | `RadicalVoronoi`, `VoronoiIntegration`, `voronoi_domains`, `voronoi_voids` | 胞 / 域 / 矩 | 曲面细分、域、空隙、电荷 |
| 振动光谱 | `PowerSpectrum`, `IRSpectrum`, `RamanSpectrum`, `VcdSpectrum`, `RoaSpectrum` | 光谱 | 振动态密度 (VDOS)、IR、Raman、振动圆二色 (VCD)、拉曼光学活性 (ROA)，均基于 ACF |

用 [`Workflow`](workflows.md) 可以把多个分析步骤串成有向图，搭成多步流水线（例如偶极矩 → ACF → 频谱）。详见 **[Compute 工作流](workflows.md)**。

## 专题指南

以下指南从基本原理出发，完整地推导每种方法——读完就能理解分析方法*为什么*能工作，而不只是*怎么*调用。

### 结构

- **[结构分析](structure.md)** — 对分布函数 $g(r)$、配位数、静态结构因子 $S(k)$（Debye 方程）、局部与网格数密度、共用的近邻列表原语，以及平均力势。涵盖 `RDF`、`StaticStructureFactorDebye`、`LocalDensity`、`GaussianDensity`、`NeighborList` 和 `PMFTXY`。
- **[分布函数](distributions.md)** — 角度 (ADF)、二面角 (DDF)、距离、联合 (CDF) 和空间 (SDF) 分布函数。涵盖 `AngleDistribution`、`DihedralDistribution`、`DistanceDistribution`、`CombinedDistribution` 和 `SpatialDistribution`。
- **[键取向序](order.md)** — 基于键球谐函数的 Steinhardt $q_\ell$/$w_\ell$、fcc/hcp/bcc 区分、六角 $\psi_6$、固液判别以及向列 $Q$ 张量。涵盖 `Steinhardt`、`Hexatic`、`SolidLiquid`、`Nematic` 和 `BondOrder`。
- **[形状、聚类与分解](descriptors.md)** — 回转张量和惯性张量、形状各向异性、聚集检测以及描述符集上的 PCA / k-means。涵盖形状描述符、`Cluster`、`ClusterProperties`、`Pca` 和 `KMeans`。
- **[氢键网络](hbonds.md)** — 几何氢键检测及其与寿命的关系。涵盖 `HBonds` 和 `HBondCriterion`。
- **[自由基 Voronoi](voronoi.md)** — 自由基曲面细分、域和空隙分析以及电子密度电荷积分。涵盖 `RadicalVoronoi`、`VoronoiIntegration`、`voronoi_domains` 和 `voronoi_voids`。

### 动力学

- **[扩散与离子输运](transport.md)** — 从随机行走和 Einstein 关系出发，覆盖均方位移、自身扩散与区别扩散（平均位移关联, MDC）、Onsager 唯象系数，以及两条等效的电导率路径（PMSD / 电流 ACF）。涵盖 `MCDCompute`、`Onsager`、`PMSDCompute`、`JACF` 和 `IonicConductivity`。
- **[速度自相关与 VDOS](vacf.md)** — 速度记忆函数：气/液/固特征与笼效应、扩散系数的 Green–Kubo 路径以及通过 Fourier 变换的振动态密度。涵盖 `compute_acf` 和 `PowerSpectrum`。
- **[Van Hove 与重取向动力学](van-hove.md)** — 时间分辨的 $G(r,t)$ 以及 Legendre 重取向 TCF $C_1$/$C_2$。涵盖 `VanHove` 和 `LegendreReorientation`。
- **[成对持久性](persistence.md)** — 驻留时间关联函数：存活指示符、连续 vs 间歇 vs 稳态持久性 (SSP) 定义、配位数以及配对扩散的关系。涵盖 `Persist`。

### 光谱学

- **[介电光谱](dielectric.md)** — $\varepsilon^*(\omega)$ 和离子电导率 $\sigma$ 的完整推导：涨落–耗散基础、Einstein–Helfand 和 Green–Kubo 路径、数值实现选项（加窗、FFT、无偏 ACF）、电解液偶极矩分解以及谱拟合方案（Debye、Cole–Cole、Havriliak–Negami）。
- **[源自 MD 的振动光谱](spectra.md)** — 基于时间关联函数的 IR、Raman、VDOS、VCD 和 ROA。涵盖 `PowerSpectrum`、`IRSpectrum`、`RamanSpectrum`、`VcdSpectrum`、`RoaSpectrum` 和 `ResonanceRamanSpectrum`。

## 相关

- [API 参考：Compute](../../api/compute.md) — 上述类的自动文档。
- [教程：轨迹](../tutorials/05_trajectory.md) — 输入数据模型。
- [教程：盒子与周期性](../tutorials/03_box_and_periodicity.md) — 动力学分析中使用的最小镜像约定。
