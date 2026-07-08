# 分子形状、聚类与分解

本文介绍 MolPy **描述符**算子，涵盖三大主题：描述分子与聚集体空间分布的形状张量、从轨迹中发现聚集体的聚类方法、以及 PCA / k-means 等将高维描述符降维分组的原语。典型应用包括聚合物线圈与球粒分析、胶束和聚集体检测，以及将轨迹压缩为少量可解释的结构坐标。

与所有 `compute` 算子一样，逐粒子的约简在 Rust（`molrs`）中完成；MolPy 负责提供帧、聚类分配和各粒子质量，直接返回原生结果。形状描述符的设计思路是**按聚类**定义——典型工作流为：*发现聚类 → 将每个聚类约简为一个张量*。

!!! note "本文使用的约定"
    - 长度单位为 Å；回转半径 $R_g$ 单位为 Å，回转张量单位为 Å²。
    - **聚类**是一组粒子——可以是单个分子，也可以是 `Cluster` 找到的物理连接聚集体。
    - 张量为 $3\times3$；其特征值即为主值。

---

## 1. 回转张量衡量分子尺寸

一组粒子的大小和形状用**回转张量**衡量，定义为各位置相对于组中心 $\mathbf r_\mathrm{c}$ 的平均外积：

$$
S = \frac{1}{N}\sum_{i=1}^{N}\big(\mathbf r_i - \mathbf r_\mathrm{c}\big)\otimes\big(\mathbf r_i - \mathbf r_\mathrm{c}\big).
$$

其迹就是**回转半径**的平方，也是表征线圈尺寸最常用的单标量指标：

$$
R_g^2 = \operatorname{tr} S = \frac{1}{N}\sum_{i=1}^{N}\big|\mathbf r_i - \mathbf r_\mathrm{c}\big|^2 .
$$

对聚合物链而言，$R_g$ 按 $N^\nu$ 标度，$\nu$ 即 Flory 指数。不同的 $\nu$ 值区分了溶胀线圈（$\nu\approx 3/5$）和塌缩球粒（$\nu = 1/3$）。

---

## 2. 形状各向异性来自回转张量的特征值

将 $S$ 对角化得到主值 $\lambda_1\le\lambda_2\le\lambda_3$。以下组合给出旋转不变的形状描述符：

$$
b = \lambda_3 - \tfrac{1}{2}(\lambda_1 + \lambda_2)\ \text{(非球形度)},\qquad
c = \lambda_2 - \lambda_1\ \text{(非柱形度)},
$$

$$
\kappa^2 = \frac{b^2 + \tfrac{3}{4}c^2}{R_g^4}\ \text{(相对形状各向异性)} .
$$

$\kappa^2 = 0$ 表示球体（或完全对称的排列），$\kappa^2 \to 1$ 表示棒状。有了这些量，不用看图也能区分球形胶束和蠕虫状胶束。

---

## 3. 惯量张量给出主轴

**转动惯量张量**是回转张量的质量加权版本：

$$
I = \sum_{i=1}^{N} m_i\big(|\mathbf r_i'|^2\,\mathbf{1} - \mathbf r_i'\otimes\mathbf r_i'\big),
\qquad \mathbf r_i' = \mathbf r_i - \mathbf r_\mathrm{cm},
$$

特征向量给出物体的主轴，特征值从长轴（$I$ 最小）排到短轴（$I$ 最大）。由此可定义分子坐标系，用于取向分析或结构对齐求平均。

---

## 4. 计算形状描述符

形状算子需要聚类分配和各聚类中心作为输入，因此通常接在邻居列表和聚类步骤之后。每个分子（或一个连通的聚集体）就是一个聚类。

```python
from molpy.compute import (
    NeighborList, Cluster, CenterOfMass, RadiusOfGyration,
    GyrationTensor, InertiaTensor,
)

nlist = NeighborList(cutoff=1.6)(frame)
clusters = Cluster(min_cluster_size=10)([frame], [nlist])     # 每帧一个 ClusterResult

com = CenterOfMass(masses)([frame], clusters)                 # 质量加权中心
rg  = RadiusOfGyration(masses)([frame], clusters, com)        # 每聚类的 R_g
S   = GyrationTensor()([frame], clusters, com)                # 每聚类的 3×3 张量
I   = InertiaTensor(masses)([frame], clusters, com)           # 每聚类的惯量张量
```

传 `masses=None` 则回退到单位质量（即几何描述符，不加权）。

---

## 5. 使用 `Cluster` 发现聚集体

先找到聚集体，才能描述它的形状。**`Cluster`** 基于 `NeighborList` 构建连接图，返回大于 `min_cluster_size` 的连通分量——胶束、液滴、逾渗网络等。配套的 **`ClusterProperties`** 在一次调用中将每个聚类约简为尺寸、中心、质量、回转张量和 $R_g$。

```python
from molpy.compute import ClusterProperties

clusters = Cluster(min_cluster_size=20)([frame], [nlist])
props = ClusterProperties()([frame], clusters)   # 每帧一个逐聚类属性的字典
```

邻居截断*就是*"连接"的物理定义，应根据 $g(r)$ 的第一个极小值来选择（参见[结构指南](structure.md)）。

---

## 6. PCA 将描述符集约简为其主导变化

用上述算子分析轨迹会得到一张高维表格——每个构型对应一行描述符。**主成分分析**把这表格沿最大方差的正交方向（即协方差矩阵的特征向量）重新表达，前两个分量通常就能抓住主导的结构运动模式。

```python
from molpy.compute import Pca, DescriptorRow

rows = [DescriptorRow(r) for r in descriptor_matrix]   # 每个 r 是一个 1-D float 数组
pca = Pca()(rows)                                       # 2 分量投影
```

PCA 之前需要对描述符做缩放或标准化——否则大幅值的列会主导方差，得到的 PCA 分量毫无意义。

---

## 7. K-means 将构型分组为状态

有了降维坐标后，**k-means** 通过迭代式分配——把每个点分到最近的质心，再重新计算质心（Lloyd 算法）——将数据划分成 $k$ 个聚类。这是把连续的 PCA 散点图转化为离散结构状态最直接的方法——折叠或展开、配对或游离。

```python
from molpy.compute import KMeans

labels = KMeans(k=3, max_iter=100, seed=0)(pca)
```

$k$ 是建模时自己选的参数，不是算法算出来的输出。应当试几个不同的 $k$ 值，验证聚类结果是否稳定、物理上是否可解释。

---

## 8. 注意事项清单

1. **质量加权** → $R_g$ 和惯量张量受质量约定影响。传真实质量得到物理主轴，传 `None` 得到纯几何量。
2. **周期性镜像** → 跨周期性盒子边界分裂的分子的 $R_g$ 无意义。计算形状之前，先把整个分子展开（取相对于聚类中心的最小镜像）。
3. **聚类截断** → 过大则合并不同聚集体，过小则分裂一个聚集体。从 $g(r)$ 的第一个极小值取值，再检查聚类尺寸分布是否稳定。
4. **PCA/k-means 前未缩放特征** → 先对每列做标准化；否则量级大的单位会主导结果。
5. **过度解读 $k$** → 即使数据中没有聚类结构，k-means 也总是返回 $k$ 个聚类。用留出验证指标或 PCA 散点图来确认聚类是否真实。

---

## 9. 参考文献

- M. Rubinstein, R. H. Colby, *Polymer Physics*, Oxford (2003) — 回转半径与链长标度。
- D. N. Theodorou, U. W. Suter, *Macromolecules* **18**, 1206 (1985) — 回转张量、非球形度和相对形状各向异性。
- I. T. Jolliffe, *Principal Component Analysis*, 2nd ed., Springer (2002).
- J. B. MacQueen, *Proc. 5th Berkeley Symp.* **1**, 281 (1967) — k-means。
- V. Ramasubramani et al., *Comput. Phys. Commun.* **254**, 107275 (2020) — cluster/shape 内核所基于的 freud 库。

## 另请参阅

- [结构分析](structure.md) — 从 $g(r)$ 选择聚类/邻居截断。
- [键取向序与局域环境](order.md) — 用于聚类的逐粒子序参数。
- [计算模块概述](index.md) — Compute → Result 模式。
- [API 参考：计算模块](../../api/compute.md)。
