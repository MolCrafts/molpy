# 结构分析：对分布函数、结构因子与密度

本文系统介绍 MolPy **结构**计算算子：径向分布函数 $g(r)$、静态结构因子 $S(k)$、局域密度与网格密度、近邻列表原语以及平均力势。这些工具指向同一个问题："物质如何排列？"——与[扩散与离子输运](transport.md)指南中的动力学分析形成静动态对照。读完本文只需了解对分布的基本概念。

与 `compute` 中其他模块一样，计算密集型内核（配对分箱、Debye求和、网格涂抹）都在 `molrs` 的 Rust 中运行；MolPy 层提取坐标和盒子尺寸，返回类型化结果。

!!! note "全文约定"
    - 长度单位为 Å，波数 $k$ 单位为 Å⁻¹，数密度 $\rho$ 单位为 Å⁻³。
    - $g(r)$ 和 $S(k)$ 为无量纲量。
    - **帧（Frame）**必须带周期性 `box`；所有配对距离均采用最小像约定。
    - 结构算子属于*帧*分析——传入一帧 `Frame`，或传入多帧列表做平均。

---

## 1. 对分布函数以理想气体为基准统计近邻数

径向分布函数 $g(r)$ 给出距离参考粒子 $r$ 处找到另一个粒子的概率，**用同密度理想气体的概率归一化**。对于体积 $V$ 中的 $N$ 个粒子，数密度 $\rho = N/V$，有

$$
g(r) = \frac{V}{N^2}\Big\langle\sum_{i}\sum_{j\ne i}\delta\big(r - r_{ij}\big)\Big\rangle
       \Big/ 4\pi r^2 .
$$

$4\pi r^2$ 是半径为 $r$ 的球壳表面积。把壳层内的原始配对计数除以壳体积 $4\pi r^2\,\Delta r$ 再除以体密度 $\rho$，就完成了从直方图到 $g(r)$ 的转换。几个重要极限行为：

- $g(r)\to 0$ 当 $r\to 0$ ——体积排斥效应；粒子不能重叠。
- $g(r)\to 1$ 当 $r\to\infty$ ——长程上结构逐渐消失，局域密度趋近于体密度。
- **峰值对应配位壳层。** 第一个峰是最近邻距离；其后的第一个极小值定义第一溶剂化壳层。

---

## 2. 配位数是 g(r) 的累积积分

对 $g(r)$ 做壳层积分得到距离 $R$ 以内的**近邻数**：

$$
n(R) = 4\pi\rho \int_0^{R} r^2\, g(r)\, \mathrm{d}r .
$$

在 $g(r)$ 的第一个极小值处算 $n(R)$ 即得**配位数**——第一壳层的平均粒子数。这是从 RDF 中提取的最有用的单一数值，也是选取[对持久性](persistence.md)分析或聚类分析截断距离的自然依据。

---

## 3. 使用 `RDF` 计算 g(r)

`RDF` 对 `NeighborList` 提供的配对距离做分箱，因此工作流始终是*构建近邻列表 → 绘制直方图*。近邻截断距离至少为 `r_max`。

```python
from molpy.compute import NeighborList, RDF

nlist = NeighborList(cutoff=10.0)(frame)        # 10 Å 内的配对
rdf = RDF(n_bins=200, r_max=10.0)               # 配置参数
result = rdf([frame], [nlist])                  # 对数据调用 → RDFResult

result.rdf          # g(r)，形状为 (n_bins,)
result.bin_centers  # 每个分箱中心的 r 值，Å
```

传入帧和近邻列表的并行列表可对轨迹做平均：

```python
nlists = [NeighborList(cutoff=10.0)(f) for f in frames]
result = RDF(n_bins=200, r_max=10.0)(frames, nlists)   # 系综平均 g(r)
```

`result.bin_edges`、`result.volume` 和 `result.n_frames` 记录直方图几何信息和归一化参数。

---

## 4. 结构因子是倒空间中的 g(r)

散射实验测的不是 $g(r)$，而是**静态结构因子** $S(k)$——同一对关联的傅里叶空间像。MolPy 用 **Debye 散射方程**从坐标直接计算：

$$
S(k) = \frac{1}{N}\Big\langle\sum_{i}\sum_{j}\frac{\sin(k\,r_{ij})}{k\,r_{ij}}\Big\rangle .
$$

各向同性系统中，$S(k)$ 与 $g(r)$ 通过傅里叶变换关联：

$$
S(k) = 1 + 4\pi\rho\int_0^\infty r^2\,[g(r)-1]\,\frac{\sin(kr)}{kr}\,\mathrm{d}r ,
$$

两者信息等价。需要与 X 射线或中子衍射对比时、确定结构长度尺度（第一尖锐衍射峰）时、或诊断长波密度涨落（$S(k\to 0)$ 由等温压缩率决定）时，$S(k)$ 都是合适的分析工具。

---

## 5. 使用 `StaticStructureFactorDebye` 计算 S(k)

选定波数网格后，Debye 求和直接给出 $S(k)$。它不做分箱，对给定 $k$ 值精确，但每帧计算量为 $\mathcal{O}(N^2)$。

```python
import numpy as np
from molpy.compute import StaticStructureFactorDebye

k = np.linspace(0.2, 12.0, 300)             # Å^-1；避免 k = 0
sk = StaticStructureFactorDebye(k)([frame]) # 对一帧或多帧调用
```

计算开销随粒子数平方增长，因此 Debye 方法适合中小型系统或子采样帧。超大盒子推荐使用基于网格的 FFT 结构因子（本文不涉及）。

---

## 6. 局域密度与网格密度确定物质的空间分布

两个算子把位置信息转换成密度场：

- **`LocalDensity`** 计算每个粒子周围截止球体内的数密度——得到逐粒子标量值，用于检测界面、空穴或局域堆积变化。与 `RDF` 类似，它使用近邻列表。
- **`GaussianDensity`** 用宽度为 `sigma` 的高斯函数将每个粒子涂抹到固定的三维网格上，产生连续的 $\rho(\mathbf r)$ 场，适合可视化或定位吸附位点。

```python
from molpy.compute import LocalDensity, GaussianDensity

nlist = NeighborList(cutoff=4.0)(frame)
local = LocalDensity(r_max=4.0)([frame], [nlist])   # 逐粒子密度

grid = GaussianDensity(nx=64, ny=64, nz=64, sigma=1.0)([frame])  # 网格上的 ρ(r)
```

截断距离（`r_max`）和涂抹宽度（`sigma`）决定分辨率：太小则密度场表现如散粒噪声；太大则实际特征被抹平。

---

## 7. 近邻列表是共享原语

`RDF`、`LocalDensity`、序参量、聚类分析和 PMFT 都依赖 **`NeighborList`**——一种空间查询，在最小像约定下返回截断距离内的所有配对。理解它的两个要点：一是可在不同分析间复用，二是它的截断距离同时决定计算开销和物理意义。

```python
nlist = NeighborList(cutoff=5.0)(frame)
nlist.n_pairs            # 找到的配对数
nlist.pairs              # (n_pairs, 2) 索引数组
nlist.distances          # 配对距离，Å
```

每帧构建一次，供该帧上所有基于截断距离的分析使用。

---

## 8. 平均力势将结构转化为自由能

对分布本质上就是玻尔兹曼因子。**平均力势**（PMF）为

$$
w(r) = -k_\mathrm{B}T \ln g(r),
$$

即平均掉所有其他自由度后沿粒子间距的有效自由能——极小值对应 §1 中的配位壳层，势垒对应壳层间的去溶剂化能垒。

`PMFTXY` 将这一思想推广到 **二维平均力势与力矩**：不只依赖单一的 $r$，而是把每个参考粒子周围的近邻位置累积到其局域 $(x, y)$ 坐标系中，揭示各向同性 $g(r)$ 平均掉的方向性结构——比如面对面接触与边对边接触。帧中带 `orientations` 拓扑块时（每个粒子记录一个 `(head, tail)` 原子对），每根键会旋转到该粒子的局域坐标系——角度由 `head - tail` 轴的 `atan2` 算出。没有该块时，分析器在实验室坐标系下工作。

```python
from molpy.compute import PMFTXY

nlist = NeighborList(cutoff=6.0)(frame)
pmft = PMFTXY(x_max=6.0, y_max=6.0, n_x=120, n_y=120)
result = pmft([frame], [nlist])   # 如果帧有 orientations 块则对齐，否则为实验室坐标系
```

---

## 9. 参数与超参数

### 9.1 参数及其含义

| 参数 | 计算算子 | 含义 |
|---|---|---|
| `n_bins` | `RDF` | $[r_\text{min}, r_\text{max}]$ 区间内直方图分箱数（典型值 100–300） |
| `r_max` | `RDF` | 最后一个分箱的上边界，**Å**；应保持 $\le L/2$（最小像） |
| `r_min` | `RDF` | 第 0 个分箱的下边界，**Å**（默认 `0.0`） |
| `k_values` | `StaticStructureFactorDebye` | 波数网格，**Å⁻¹**；从大于 0 开始——最小有物理意义的 $k$ 约为 $2\pi/L$ |
| `cutoff` | `NeighborList` | 配对搜索半径，**Å**；必须 **≥ 所有使用该列表的分析算子各自的 `r_max`** |
| `r_max`, `diameter` | `LocalDensity` | 计数球半径，**Å**；`diameter`（默认 `0.0`）应用粒子尺寸修正——`0.0` 仅对中心计数 |
| `nx`, `ny`, `nz`, `sigma` | `GaussianDensity` | 每轴网格分辨率；高斯涂抹宽度，**Å** |

分箱几何与 Rust 内核一致：分箱宽度 $\Delta r = (r_\text{max} - r_\text{min})/n_\text{bins}$，分箱中心 $r_i = r_\text{min} + (i + \tfrac12)\,\Delta r$——也就是 `result.bin_centers` 返回的值。

### 9.2 超参数的影响

- **`n_bins`：分箱宽度与噪声的权衡。** 每个分箱的计数与 $1/n_\text{bins}$ 成正比，相对散粒噪声 $\sigma_\text{bin} \propto 1/\sqrt{\text{计数}} \propto \sqrt{n_\text{bins}}$。分箱数 **太少** 导致第一个峰欠采样——峰高偏低，进而影响 §2 中的配位数（确保第一个峰上至少有 5 个分箱）。分箱数 **太多** 则用帧数换分辨率。
- **`r_max` 超过盒子一半。** 大于 $L/2$ 的距离在最小像约定下受周期映像污染；$g(r)$ 尾部失去物理意义。这是 *正确性* 限制，不是分辨率旋钮。
- **`r_max` / `cutoff` 与计算开销。** `NeighborList` 返回的配对数随 $\rho\,r^3$ 增长；链表构建和下游直方图的计算量同步增长。不要为了绘制 6 Å 的 $g(r)$ 而查询 12 Å 的近邻。
- **Debye $k$ 网格密度。** Debye 和对每个请求的 $k$ 值都精确，但开销为 $\mathcal{O}(N^2)$ **每 $k$ 每帧**——300 个点的网格对应 300 次双重求和。只在有特征结构的区域加密网格（第一尖锐衍射峰），且 $k < 2\pi/L$ 探测的始终是大于盒子的涨落。
- **帧数。** $g(r)$ 和 $S(k)$ 是系综平均值。对于 *不相关* 的帧，每个分箱的噪声随 $1/\sqrt{N_\text{frames}}$ 下降。采样间隔小于结构弛豫时间的帧只会增加计算开销，不改善统计。
- **`LocalDensity.r_max` / `GaussianDensity.sigma`（场分辨率）。** 两者决定密度场的平滑长度：太小 → 散粒噪声主导，太大 → 界面和空穴被抹平（参见 §6）。应根据物理长度（第一个 $g(r)$ 极小值、界面宽度）选取，而非网格。

这些旋钮的失效模式总结在 [§10](#10) 中。

---

## 10. 常见陷阱清单

1. **`r_max`（或最大 $k$ 特征）超过盒子一半** → 周期映像污染结果。保持 `r_max ≤ L/2` 以满足最小像约定。
2. **近邻截断距离小于 `r_max`** → RDF 截断不完整；近邻列表 `cutoff` 必须至少等于直方图的 `r_max`。
3. **分箱数太少** → 尖锐的第一峰被抹平，配位数偏低；分箱数太多则各分箱噪声大。典型值为 100–300 个分箱。
4. **Debye 网格中包含 $k = 0$** → 除零错误；网格应从小的正 $k$ 开始。
5. **仅用单帧计算系综量** → $g(r)$ 和 $S(k)$ 依赖统计；峰高可信之前应平均多个不相关帧。
6. **无边界盒子** → 上述所有分析都需要周期性 `box`；自由帧会报错。

---

## 11. 参考文献

- M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed., Oxford (2017) — $g(r)$、配位数和结构因子。
- J.-P. Hansen, I. R. McDonald, *Theory of Simple Liquids*, 4th ed., Academic (2013) — $g(r)\!\leftrightarrow\!S(k)$ 关系和压缩率求和规则。
- P. Debye, *Ann. Phys.* **351**, 809 (1915) — Debye 散射方程。
- G. van Anders et al., *ACS Nano* **8**, 931 (2014) — 平均力势和力矩。
- V. Ramasubramani et al., *Comput. Phys. Commun.* **254**, 107275 (2020) — freud 库，本文内核以此为基础建模。

## 参见

- [扩散与离子输运](transport.md) — 动力学对应分析（MSD、Onsager、电导率）。
- [键取向序与局域环境](order.md) — 当*哪些*近邻重要，而不仅是数量。
- [对持久性](persistence.md) — 从第一个 $g(r)$ 极小值选取其截断距离。
- [计算模块概览](index.md) — Compute → Result 模式。
- [API 参考：计算模块](../../api/compute.md)。
