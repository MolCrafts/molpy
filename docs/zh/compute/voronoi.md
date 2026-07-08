# 径向 Voronoi 细分：域、空腔与电荷积分

本文系统介绍径向 Voronoi 细分（Radical Voronoi，又称 Laguerre / 幂 Voronoi，MolPy 移植自 Brehm 等人的参考代码）及基于它的三种分析：连通域检测、空腔分析和电子密度积分（用于计算分子电荷和偶极矩）。Voronoi 将空间每个点分配到距离最近的原子；径向变体引入原子半径加权，对不同大小的原子给出物理更合理的分割。

细分与约简运算在 Rust 层（`molrs`）完成，直接操作位置、半径和模拟盒子，不涉及 `Frame` 对象。

!!! note "全文约定"
    - 位置和半径单位是 Å，体积是 Å³。
    - 细分采用全周期边界（最近镜像约定）。
    - *胞（cell）*：单个原子的径向 Voronoi 多面体；*域（domain）* 和 *空腔（void）*：胞的集合。

---

## 1. 径向细分用幂距离划分空间

普通 Voronoi 中两原子边界是连线的垂直平分面——这仅在原子半径相等时才成立。**径向**变体将边界设在*幂距离*相等的位置：

$$
\pi_i(\mathbf r) = |\mathbf r - \mathbf r_i|^2 - R_i^2,
$$

半径越大（$R_i$ 越大）的原子占据越大的胞。每个原子的胞体积对应有物理意义的局部体积，是局部密度、堆积分数及后续分析的基础。

```python
from molpy.compute import RadicalVoronoi

cells = RadicalVoronoi()(positions, radii, box)   # -> VoronoiCells
cells.neighbors(i)    # 与胞 i 共面的胞
```

---

## 2. 域将同类胞合并

为每个原子分配标签（种类、电荷符号、极性等），将共享面的同标签相邻胞合并，得到**域**——纳米结构液体（如离子液体中的极性/非极性网络）中连通的介观区域。`voronoi_domains` 返回每个标签对应的域数目和体积：

```python
from molpy.compute import voronoi_domains

domains = voronoi_domains(cells, labels)   # labels: 每个原子的整数标签
```

---

## 3. 空腔是细分中的空胞

将不含"真实"原子的胞标记出来，聚合为连通团簇，得到**空腔**空间——揭示与扩散、气体溶解度和孔隙率相关的空洞和自由体积：

```python
from molpy.compute import voronoi_voids

voids = voronoi_voids(cells, is_void, box_volume)   # is_void: 每个胞的布尔值
```

---

## 4. Voronoi 积分计算分子电荷与偶极矩

对每个径向 Voronoi 胞内的**电子密度**积分，再按分子求和，总电荷分配到每个原子和分子上——这是从 Voronoi 途径得到分子电荷与**偶极矩**的方法。分子偶极矩是偶极自相关计算[红外光谱](spectra.md)的输入，因此该分析将*从头算*分子动力学电子密度（如 cube 轨迹）与红外光谱预测衔接起来。

```python
from molpy.compute import VoronoiIntegration

moments = VoronoiIntegration()(
    positions, radii, atomic_numbers, atom_to_mol, n_mol, grid, box
)
# moments -> 每个分子的电荷与偶极矢
```

---

## 5. 常见陷阱

1. **半径选择**：划分结果依赖原子半径，应使用一套一致且有物理依据的半径（如范德华半径或共价半径），并在报告中说明。
2. **非周期性盒子**：构建器是周期性的，需提供模拟盒子而非自由框架，否则表面胞无界。
3. **标签/空腔数组长度**：`labels` 和 `is_void` 须按细分顺序为每个原子或胞给出一个条目。
4. **积分网格分辨率**：密度网格过粗会导致积分电荷偏差，应进行网格间距收敛测试。
5. **电荷中性**：积分得到的分子电荷之和应等于体系总电荷；偏差过大说明网格或半径设置有问题。

---

## 6. 参考文献

- B. J. Gellatly, J. L. Finney, *J. Non-Cryst. Solids* **50**, 313 (1982) — 径向（幂）Voronoi 细分。
- M. Thomas, M. Brehm, B. Kirchner, *Phys. Chem. Chem. Phys.* **17**, 3207 (2015) — 电子密度 Voronoi 积分用于分子偶极矩。
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105 (2020) — 参考实现；域与空腔分析。

## 参见

- [分子动力学振动光谱](spectra.md) — 使用 Voronoi 分子偶极矩。
- [结构分析](structure.md) — 局部密度与堆积。
- [计算模块概述](index.md) — Compute → Result 模式。
- [API 参考：计算模块](../../api/compute.md)。
