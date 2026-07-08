# 键取向序与局域环境

键取向序参数量化局域环境的**有序程度**和排列**类型**。[结构分析指南](structure.md)关心的是给定距离内有几个邻居，而本章的算子关心的是这些邻居在**角度上如何排列**——它们排成了晶格？六角形薄膜？还是沿着同一个指向矢排列？典型应用包括结晶与熔化分析、二维相变以及液晶取向。

球谐求和部分在 Rust（`molrs`）中运行，其他 `compute` 算子也是这个模式。MolPy 负责提供坐标、盒子和近邻列表，最终返回带类型的结果。每项分析接受**两个输入**——`Frame` 和 `NeighborList`，因为"有序"是相对于粒子的近邻来定义的。

!!! note "全文通用约定"
    - 长度单位为 Å。序参数无量纲。
    - 粒子的**键**是指向 `NeighborList` 中近邻的向量；定义这些近邻的截断半径是最关键的选择（见"陷阱"一节）。
    - $Y_{\ell m}$ 为球谐函数；$\ell$ 为谐函数阶数。

---

## 1. 局域序存在于键的球谐展开中

Steinhardt、Nelson 和 Ronchetti（1983 年）的核心思路是：把粒子键的方向展开成球谐函数。对于有 $N_b(i)$ 个近邻、方向为 $(\theta_{ij}, \phi_{ij})$ 的粒子 $i$，

$$
q_{\ell m}(i) = \frac{1}{N_b(i)}\sum_{j=1}^{N_b(i)} Y_{\ell m}\big(\theta_{ij}, \phi_{ij}\big).
$$

这些复系数会随粒子坐标一起旋转，本身不是可直接观测的物理量。物理信息藏在它们的**旋转不变量**中。

---

## 2. q_ℓ 和 w_ℓ 是旋转不变的序参数

二阶和三阶不变量是与坐标系无关的局域对称性指纹：

$$
q_\ell(i) = \sqrt{\frac{4\pi}{2\ell+1}\sum_{m=-\ell}^{\ell}\big|q_{\ell m}(i)\big|^2},
\qquad
w_\ell(i) = \sum_{m_1+m_2+m_3=0}
   \begin{pmatrix}\ell&\ell&\ell\\ m_1&m_2&m_3\end{pmatrix}
   q_{\ell m_1}q_{\ell m_2}q_{\ell m_3}.
$$

阶数 $\ell$ 决定了所关注的对称性：

- **$q_6$**——密堆积序的主力参数，对 fcc、hcp 和 bcc 取较大且特征明显的值，在液体中取小值。
- **$q_4$**——与 $q_6$ 联用，可区分 fcc、hcp 和 bcc。
- **$w_\ell$**（三阶不变量）——进一步增强区分能力；其符号可区分 $q_\ell$ 值相近的晶体结构。

Lechner 与 Dellago（2008 年）提出的**局域平均**变体先在粒子及其近邻上对 $q_{\ell m}$ 做平均，再算不变量。代价是多包一层近邻球壳，换来固/液分离效果的显著提升。

---

## 3. 使用 `Steinhardt` 计算 Steinhardt 序

```python
from molpy.compute import NeighborList, Steinhardt

nlist = NeighborList(cutoff=1.5)(frame)        # 第一近邻壳层
q = Steinhardt(l=[4, 6])([frame], [nlist])     # 每个粒子的 q_4 和 q_6
```

通过构造函数参数切换为平均化变体和三阶变体：

```python
q_avg = Steinhardt(l=[6], average=True)          # Lechner–Dellago 平均化 q_6
w = Steinhardt(l=[6], wl=True, wl_normalize=True)  # 归一化 w_6
```

把每个粒子的序参数画成直方图，就成了相态诊断工具：双峰的 $q_6$ 分布是固液共存的特征。

---

## 4. 二维序：六角序参数

在二维薄膜中，相关的对称性是六重对称，序参数为 **六角序参数** $\psi_k$（其中 $k=6$）：

$$
\psi_k(i) = \frac{1}{N_b(i)}\sum_{j=1}^{N_b(i)} e^{\,i k\,\theta_{ij}},
$$

其中 $\theta_{ij}$ 是面内键角。完美三角晶格上 $|\psi_6|\to 1$，各向同性液体中 $|\psi_6|\to 0$。它的空间关联函数是 KTHNY 熔化理论的序参数。

```python
from molpy.compute import Hexatic

psi6 = Hexatic(k=6)([frame], [nlist])
```

---

## 5. 逐粒子区分固相与液相

`SolidLiquid` 实现了 ten Wolde–Ruiz-Montero–Frenkel 判据，给每个粒子打上"类固"或"类液"的标签。当两个粒子各自环境的归一化复向量 $\mathbf q_\ell$ 充分对齐时，它们共享一个**类固键**：

$$
s_{ij} = \frac{\sum_m q_{\ell m}(i)\,q_{\ell m}^{*}(j)}
              {\big|\mathbf q_\ell(i)\big|\,\big|\mathbf q_\ell(j)\big|} > q_\text{threshold},
$$

一个粒子拥有至少 `n_threshold` 个这样的键时，便被判定为**固态**。

```python
from molpy.compute import SolidLiquid

sl = SolidLiquid(l=6, q_threshold=0.7, n_threshold=6)([frame], [nlist])
```

这是跟踪熔体中结晶核生长的标准方法。

---

## 6. 各向异性粒子的取向序：向列 Q-张量

当粒子具有内禀方向 $\mathbf u_i$（杆状分子、介晶基元、成键片段）时，集体取向由**向列序张量**衡量：

$$
Q = \frac{1}{N}\sum_i \Big(\tfrac{3}{2}\,\mathbf u_i\otimes\mathbf u_i - \tfrac{1}{2}\,\mathbf I\Big).
$$

最大特征值为标量向列序参数 $S$（0 为各向同性，1 为完全对齐），对应的特征向量就是**指向矢**（director）。`Nematic` 从 `Frame` 的 `orientations` 拓扑块中读入每个粒子的取向轴（每个粒子对应一个 `(head, tail)` 原子对；指向矢为单位向量 `head - tail`），返回序参数、特征值、指向矢以及完整的 $Q$ 张量：

```python
from molpy.compute import Nematic

# `frame` 必须包含 `orientations` 块，例如每个粒子对应一行 (head, tail)。
order, eigenvalues, director, q_tensor = Nematic()([frame])
```

---

## 7. 键取向图可视化局域几何

不变量把局域环境压缩成一个数值，而 **`BondOrder`** 保留了完整信息：它把键方向投影到球面 $(\theta, \phi)$ 网格上，在所选粒子和 `Frame` 上累积。生成的图表直接展示配位壳层的角分布——四面体环境的四叶图案，八面体环境的六叶图案。

```python
from molpy.compute import BondOrder

diagram = BondOrder(n_theta=80, n_phi=160)([frame], [nlist])
```

---

## 8. 陷阱检查清单

1. **近邻截断决定结果。** $q_\ell$ 强烈依赖哪些键被计入。截断应选在 $g(r)$ 的第一个极小值处（参见[结构分析指南](structure.md)），或使用固定的近邻数目并在不同系统间保持一致。
2. **选错 $\ell$ = 选错对称性。** 密堆积用 $q_6$，二维六角用 $\psi_6$，区分 fcc/hcp/bcc 用 $q_4\!+\!q_6$。单个 $\ell$ 很少能解决所有问题。
3. **不要跳过平均化变体。** 未经平均的 $q_6$ 固/液峰宽而重叠；Lechner–Dellago 平均化通常值得多算一层壳的代价。
4. **归一化约定。** 归一化和未归一化的 $w_\ell$ 取值范围不同；报告中需明确注明使用的是哪个版本（`wl_normalize`）。
5. **有限尺寸与表面效应。** 自由表面或界面附近的粒子近邻壳层被截断，序参数会人为偏低；应将它们排除或标记出来。
6. **向列轴端点反转。** 指向矢为 `head - tail`（块列 `atomi`/`atomj`）。$Q$ 张量与符号无关，$S$ 不受影响，但报告的指向矢方向沿 head 到 tail 的指向。

---

## 9. 参考文献

- P. J. Steinhardt, D. R. Nelson, M. Ronchetti, *Phys. Rev. B* **28**, 784 (1983) —— 键取向序参数 $q_\ell$、$w_\ell$。
- D. R. Nelson, B. I. Halperin, *Phys. Rev. B* **19**, 2457 (1979) —— 六角序与二维熔化（KTHNY）。
- P. R. ten Wolde, M. J. Ruiz-Montero, D. Frenkel, *J. Chem. Phys.* **104**, 9932 (1996) —— 类固键判据用于成核分析。
- W. Lechner, C. Dellago, *J. Chem. Phys.* **129**, 114707 (2008) —— 局域平均序参数。
- P. G. de Gennes, J. Prost, *The Physics of Liquid Crystals*, 2nd ed. (1993) —— 向列 $Q$ 张量与序参数。
- V. Ramasubramani et al., *Comput. Phys. Commun.* **254**, 107275 (2020) —— freud 库，本文所涉内核基于该库进行建模。

## 参见

- [结构分析](structure.md) —— 通过 $g(r)$ 选择近邻截断，以及 $S(k)$ 和密度场。
- [分子形状、聚类与分解](descriptors.md) —— 将有序粒子分组为晶核和畴区。
- [Compute 概述](index.md) —— Compute → Result 模式。
- [API 参考：Compute](../../api/compute.md)。
