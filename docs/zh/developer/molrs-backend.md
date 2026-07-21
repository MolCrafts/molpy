# molrs 后端

MolPy 的分析算子是 [molrs](https://github.com/MolCrafts/molrs) 的薄 Python 封装。molrs 是一个用 Rust 实现的列存储与计算内核，也是**必需**的运行时依赖——调用方必须直接从 `molrs` 导入 `Frame` 和 `Block`。两者都由 Rust 的 `Store` 支撑，`compute` 算子也直接转发到 Rust。不存在纯 Python 回退方案，也没有启用或禁用的开关。

以下几个场景展示该后端在日常分析中的具体体现：box 类型如何与 molrs 共享，近邻列表和径向分布函数如何构建，以及从 Python 视角看 molrs 分析目录的其余内容。

## 安装自动引入 molrs

molrs 以 PyPI 包 `molcrafts-molrs` 的形式发布。作为硬依赖，常规安装就会自动带上它：

```bash
pip install molcrafts-molpy
```

无需记忆 `molpy[molrs]` 这类额外标记——该 extra 键已被移除。从旧版本升级的用户请查看[更新日志](../../changelog.md)：molrs 从可选变为必需，属于破坏性变更。

## box 是 molrs 对象，而非其副本

`molpy.Box` 不包装一个 molrs box；它**继承**自 molrs：

```python
import molrs
from molpy.core.box import Box

class Box(molrs.Box):
    ...
```

这意味着 molpy box 可以直接传给任何 molrs API，无需修改——不存在 `.to_molrs()` 桥接或坐标转换：

```python
import molrs
import molpy as mp

box = mp.Box.cubic(10.0)
assert isinstance(box, molrs.Box)   # 它 *就是* 一个 molrs box
```

同样，`frame.box` 可以直接传给 Rust 侧的函数，比如 `molrs.NeighborQuery`。molpy 在继承的 Rust 核心之上增加的方法（`Style`、`cubic`、`from_lengths_angles`、`diff_dr`……）也仍然可用。

## 近邻列表来自 linked-cell 内核

`NeighborList` 使用 molrs 的 linked-cell 算法搜索给定截断半径内的所有近邻对（O(N)，N 为原子数）。它直接返回 molrs 的 `NeighborList` 结果对象——molpy 不会重新包装它：

```python
import numpy as np
import molpy as mp
from molpy.compute import NeighborList

rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 20.0, size=(500, 3))

frame = molrs.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box.cubic(20.0)

neighbors = NeighborList(cutoff=8.0)(frame)
print(neighbors.n_pairs)          # 找到的对数
print(neighbors.distances[:5])    # 原子对距离，从 Rust 借用
```

周期性 box 必不可少：在无 box 的框架上调用 `NeighborList` 会抛出 `ValueError`，因为没有可供参照的最小镜像约定。

## RDF 重用传入的近邻列表

`RDF` 从一帧（或多帧）及其对应的近邻列表计算径向分布函数：

$$ g(r) = \frac{V}{N\,N_q}\,\frac{\langle n(r)\rangle}{4\pi r^2\,\Delta r} $$

显式传入近邻列表，可以把耗时的近邻搜索挡在直方图循环之外，让多次分析共用同一次搜索结果：

```python
from molpy.compute import RDF

result = RDF(n_bins=50, r_max=8.0)(frame, neighbors)
print(result.bin_centers)   # 每个 bin 中心处的 r
print(result.rdf)           # g(r)
```

对于理想气体（均匀随机点），`result.rdf` 中间几个 bins 的值应接近 1.0，这是校验归一化是否正确的常规手段。传入列表时，多帧结果会被平均：`RDF(...)(frames, neighbor_lists)`。

## 分析目录以 molpy 算子形式暴露

一批标准轨迹分析已内置在 molrs 中。MolPy 将每种分析暴露为一个 `Compute` 算子，算子转发参数并直接返回 molrs 结果类型：

| 算子 | 计算内容 |
|----------|------------------|
| `MSD` | 相对于滞后时间的均方位移 |
| `Cluster`、`ClusterCenters`、`ClusterProperties` | 连通分量聚类、聚类中心，以及各聚类的大小/质量/回转半径 |
| `CenterOfMass` | 质量加权中心 |
| `GyrationTensor`、`RadiusOfGyration`、`InertiaTensor` | 形状描述符 |
| `Pca`、`KMeans` | 降维与划分 |
| `Steinhardt`、`Hexatic`、`SolidLiquid`、`Nematic` | 键取向序、六角序、固液分类、向列 Q 张量 |
| `LocalDensity`、`GaussianDensity` | 逐粒子局域密度和高斯涂抹密度网格 |
| `StaticStructureFactorDebye` | 基于 Debye 方程的静态结构因子 S(k) |
| `BondOrder` | 在 (θ, φ) 网格上的键方向图 |
| `PMFTXY` | 二维平均力势与扭矩 |

这些算子遵循与 `NeighborList` / `RDF` 相同的调用约定。基于近邻的算子接受 `(frames, nlists)`；少数算子接受其他输入（`GaussianDensity` 和 `StaticStructureFactorDebye` 只接受 `frames`，`Nematic` 从框架的 `orientations` 拓扑块读取逐粒子指向矢，`ClusterProperties` 接受 `Cluster` 结果）：

```python
from molpy.compute import MSD, GyrationTensor, Steinhardt, StaticStructureFactorDebye

msd = MSD()(frames)                            # 轨迹上的时间序列
rg2 = GyrationTensor()(frame)                  # 单帧的回转张量
q6 = Steinhardt([6])(frame, neighbors)         # 每个粒子的 Steinhardt q6
sk = StaticStructureFactorDebye(k_values)(frame)   # 结构因子 S(k)
```

## 一次坐标拷贝，且仅此一次

molpy 与 molrs 之间的边界刻意保持零拷贝。坐标只跨越该边界一次，发生在 `frame["atoms"][["x", "y", "z"]]` 内部：三个独立的列通过 `numpy.column_stack` 堆叠成连续的 `(N, 3)` 数组。只要坐标仍以独立的 `x`/`y`/`z` 列存储，这种重组就不可避免。下游的所有内容——原子对索引、距离、直方图 bins——都是对 Rust 所拥有缓冲区的借用只读视图，因此算子不会防御性地对输入做 `.copy()`，也不会修改传入的框架。

## 三维结构通过 molrs embed 生成

从仅有连接信息的图生成坐标同样在 molrs 上完成。`molpy.compute.Generate3D` 封装了 molrs 的距离几何 + 最小化流水线：

```python
from molpy.parser import parse_molecule
from molpy.compute import Generate3D

mol = parse_molecule("CCO")          # 乙醇，重原子图
mol_3d = Generate3D(seed=42)(mol)    # 全新结构，输入未被修改
```

RDKit 适配器（`molpy.adapter.rdkit`）仍可作为备选的外部后端，但 molrs 流水线是默认主干。
