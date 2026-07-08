# 分布函数：角度、二面角、联合分布与空间分布

MolPy 从参考实现移植了一组**几何分布函数**：角度分布函数 (ADF)、二面角分布函数 (DDF)、距离分布函数、它们的联合**组合分布函数** (CDF)，以及面向取向的**空间分布函数** (SDF)。如果说[径向分布函数](structure.md)回答的是"有多远？"，那么这些函数回答的是"在什么角度、以何种组合、朝哪个方向？"——它们解读局域有序性的*几何形态*，而非仅仅径向范围。

直方图核心在 Rust (`molrs`) 中运行，与所有 `compute` 算子一致。MolPy 层负责提取坐标和盒子，返回类型化的结果。几何分布函数接受**一个输入**——帧。需要统计的原子元组从每帧的核心拓扑块读取：`bonds`（对）用于距离分布，`angles`（三元组）用于 ADF，`dihedrals`（四元组）用于 DDF。无需传递独立的索引数组。

!!! note "全文通用约定"
    - 距离以 Å 为单位，角度和二面角以度为单位，密度以 Å⁻³ 为单位。
    - 原子元组来自帧的拓扑块——`bonds` 中的对 `(i, j)`，`angles` 中的三元组 `(i, j, k)`（顶点在 `j`），`dihedrals` 中的四元组。通过 `Atomistic.get_topo(gen_angle=True, gen_dihe=True)` 感知它们。
    - 角度分布带有平凡的 `sin θ` 立体角权重；结果的 `density_sin_corrected` 将其去除，使无结构分布恢复平坦。

---

## 1. 距离、角度和二面角分布对内坐标进行直方图统计

这些函数都对选定原子元组集合上的某个内坐标做归一化直方图。**角度分布函数**对一组三元组定义为

$$
p(\theta) = \frac{1}{N_\text{groups}}\Big\langle\sum_{(i,j,k)}\delta\big(\theta - \theta_{ijk}\big)\Big\rangle,
$$

其中 $\theta_{ijk}$ 是顶点 $j$ 处的角度。**二面角分布**将 $\theta$ 换成四元组的扭转角，**距离分布**换成原子对的距离——一种限制在特定原子对上的 RDF，不包含 $4\pi r^2$ 壳层归一化。三者共同揭示键角的刚性、构象群体分布（DDF 中 gauche 与 anti 的对比）以及选择性原子对的结构特征。

---

## 2. 计算几何分布

先感知拓扑结构（让帧携带 `bonds` / `angles` / `dihedrals`），然后在单个或多个帧上调用算子：

```python
from molpy.compute import AngleDistribution, DihedralDistribution, DistanceDistribution

# 帧必须携带相关的拓扑块，例如来自已构建的结构：
frame = mol.get_topo(gen_angle=True, gen_dihe=True).to_frame()

adf = AngleDistribution(n_bins=180, min=0.0, max=180.0)
result = adf([frame])        # 角度三元组从 frame["angles"] 读取

result.bin_centers           # 每个 bin 处的角度，单位为度
result.density               # 归一化的 p(theta)
result.density_sin_corrected # 立体角修正后的分布
```

`DistanceDistribution(n_bins, min, max)` 和 `DihedralDistribution(n_bins, min=-180, max=180)` 遵循相同的调用模式，分别读取帧的 `bonds` 和 `dihedrals` 块。

---

## 3. 组合分布函数揭示坐标间的关联性

一维分布会平均掉关联性。例如，O···H 距离与 O···H–C 角度之间的*联合*分布能判断短接触是否也呈现线性——这正是氢键的特征。**组合分布函数** (CDF) 将多个坐标同时直方图统计到一个多轴网格中：

$$
p(x_1, x_2, \dots) = \frac{1}{N}\Big\langle\sum_\text{groups}\prod_a \delta\big(x_a - x_a^\text{group}\big)\Big\rangle.
$$

每个轴声明为一个 `(kind, n_bins, min, max, sin_weight)` 元组：

```python
from molpy.compute import CombinedDistribution

cdf = CombinedDistribution([
    ("distance", 100, 2.0, 4.0, False),   # 从 frame["bonds"] 读取
    ("angle",     90, 90.0, 180.0, True), # 从 frame["angles"] 读取（含 sin 权重）
])
result = cdf([frame])   # 每个轴读取其对应拓扑块中的元组
```

结果包含多维直方图以及辅助方法（`bin_width_product`、`flat_index`），用于对联合密度做积分或切片。

---

## 4. 空间分布函数映射三维结构

**空间分布函数** (SDF) 是全三维泛化：它在参考分子上建立体固连坐标系——将参考原子通过 Kabsch 对准到模板几何结构——然后在体坐标系网格上累积目标原子的密度。结果呈现分子周围邻居分布的三维云团——水的孤对电子瓣、芳香环的堆叠几何结构——而这些被各向同性的 $g(r)$ 平均成了单个壳层。

```python
import numpy as np
from molpy.compute import SpatialDistribution

sdf = SpatialDistribution(
    reference=[o, h1, h2],            # 定义体坐标系的原子
    template=np.array([[0,0,0],[0.76,0.59,0],[-0.76,0.59,0]]),  # 理想几何结构
    target=[o],                       # 邻近 O 原子的密度
    n=(64, 64, 64),                   # 网格分辨率
    extent=(8.0, 8.0, 8.0),           # 每个轴的一半范围，Å
    bulk_density=0.033,               # 可选 -> result.g_sdf
)
result = sdf(frames)
result.density   # 体固连坐标系上的目标密度
result.g_sdf     # 按 bulk_density 归一化（若提供了该参数）
```

如果帧携带 `orientations` 拓扑块（每个目标原子对应一个 `(head, tail)` 原子对，按 `target` 顺序排列），结果还会通过 `result.orientation` 暴露每个体素的单位 `head - tail` 向量在体固连坐标系中的平均取向；没有该块时，SDF 不包含取向信息。

---

## 5. 常见陷阱清单

1. **顶点顺序错误** —— ADF 的角度位于每个三元组的*中间*索引；`angles` 拓扑块已经按顶点在中间的方式存储（由 `get_topo(gen_angle=True)` 感知）。
2. **忘记 sin 修正** —— 原始角度密度仅因立体角效应就会在 90° 附近出现峰值；比较 `density_sin_corrected` 才能获得真正的结构信息。
3. **CDF 轴数量不匹配** —— 每个轴读取其对应 `kind` 的拓扑块（`bonds` / `angles` / `dihedrals`），所有轴必须产出相同数量的元组，否则联合样本无定义。
4. **SDF 模板未对准** —— 模板必须与 `reference` 原子的顺序和合理的几何结构匹配，否则体坐标系（以及整个映射图）毫无意义；先用一个小型对称参考验证。
5. **采样稀疏** —— 二维/三维直方图所需的样本量远大于一维直方图才能填满 bin；解读峰值高度之前，先对多个帧做平均。

---

## 6. 参考文献

- M. Brehm, B. Kirchner, *J. Chem. Inf. Model.* **51**, 2007 (2011) — 参考实现；径向/角度/二面角和组合分布函数。
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105 (2020) — 参考实现，当前功能集。
- I. M. Svishchev, P. G. Kusalik, *J. Chem. Phys.* **99**, 3049 (1993); P. G. Kusalik, I. M. Svishchev, *Science* **265**, 1219 (1994) — 空间分布函数。

## 另请参阅

- [结构分析](structure.md) — 径向分布函数和结构因子。
- [氢键网络](hbonds.md) — 距离-角度 CDF 是天然的氢键映射图。
- [计算概述](index.md) — Compute → Result 模式。
- [API 参考：计算模块](../../api/compute.md)。
