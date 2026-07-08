# 盒子与周期性

坐标的意义依赖于它所参照的模拟单元格。`Box` 承载这个单元格，负责将坐标包裹到主像内、计算最小像距离。

## 为什么周期性很重要

分子动力学模拟用一个小盒子（通常几千到几百万个原子）来代表体相材料——液体、聚合物、晶体。如果盒子太小，表面效应会严重干扰模拟结果。解决办法是通过*周期性边界条件*将盒子在所有方向上无限复制：从右侧离开的原子，从左侧重新出现。两个原子之间的距离总是取最短路径，这条路径很可能穿过周期性边界。

这意味着原始坐标本身就包含歧义。长度为10的盒子里有两个原子，位置分别在1.0和9.0。它们之间的距离不是8.0，而是通过周期像算出的2.0。所有距离计算、近邻搜索和结构分析都必须考虑这个事实。

## 坐标本身不够

原子快照记录了位置，但周期性系统中的位置暗藏一个依赖关系：坐标只有相对于模拟单元格才有意义。原始坐标中看似相距很远的两个原子，考虑周期性包裹后可能是近邻。没有盒子，距离、位移和近邻列表都说不清楚。

**`Box` 定义模拟单元格，赋予坐标周期性的含义。**

MolPy 把盒子当作显式对象，而不是把周期性藏在某个标志或辅助函数里。盒子不是可选的标注——它是物理模型的一部分。

## 创建盒子

`Box` 为三种常用单元格类型提供了工厂构造方法。

```python
import molpy as mp
import numpy as np

cubic = mp.Box.cubic(20.0)
ortho = mp.Box.orth([10.0, 20.0, 30.0])
tric  = mp.Box.tric(lengths=[10.0, 12.0, 15.0], tilts=[1.0, 0.5, 0.2])

print(cubic)    # <Orthogonal Box: [20. 20. 20.]>
print(ortho)    # <Orthogonal Box: [10. 20. 30.]>
print(tric)     # <Triclinic Box: ...>
```

也可以直接传入3×3矩阵，列向量为晶格矢量。

```python
matrix = np.array([[10.0, 1.0, 0.5],
                   [0.0, 12.0, 0.2],
                   [0.0,  0.0, 15.0]])
box = mp.Box(matrix=matrix)
print(box.lengths)
```

每个盒子都带有一个 `pbc` 数组——三个布尔值，分别控制各轴是否周期性。默认全周期。slab 几何结构可以关闭 z 轴。

```python
slab = mp.Box.orth([20.0, 20.0, 50.0], pbc=[True, True, False])
print(slab.pbc)   # [ True  True False]
```

## 派生属性

盒子根据晶格矩阵计算出一系列几何量：`lengths`、`volume`、`origin`、`bounds`，三斜单元格还包括 `tilts` 和 `angles`。

```python
box = mp.Box.orth([10.0, 12.0, 15.0])
print(f"lengths: {box.lengths}")
print(f"volume:  {box.volume}")
print(f"style:   {box.style}")
```

## 将坐标包裹入主像

模拟过程中漂移到盒子外部的原子，可以用 `wrap` 映射回来，得到位于主像内的包裹坐标。

```python
box = mp.Box.cubic(10.0)

points = np.array([
    [12.0, -2.0, 5.0],
    [25.0,  8.0, -3.0],
])

wrapped = box.wrap(points)
print(wrapped)
# 所有点现在都在各轴的 [0, 10) 范围内
```

如果后续需要重建未包裹的轨迹，`get_images` 能查出每个坐标被移动了几个盒长，`unwrap` 则可以反转该操作。

```python
images = box.get_images(points)
unwrapped = box.unwrap(wrapped, images)
print(np.allclose(unwrapped, points))   # True
```

## 分数坐标

绝对坐标（笛卡尔）与分数坐标之间的转换，对某些分析和文件格式写入很有用。分数坐标将位置表示为晶格矢量的分数，因此对包裹系统来说，取值范围始终是 [0, 1)。

```python
absolute = np.array([[5.0, 3.0, 7.0]])
fractional = box.make_fractional(absolute)
restored = box.make_absolute(fractional)

print(fractional)                          # [[0.5, 0.3, 0.7]]
print(np.allclose(restored, absolute))     # True
```

## 最小像距离

周期性系统中两个点之间具有物理意义的间距是最短的那一个——即最小像位移。`diff` 计算位移矢量，`dist` 计算标量距离。

```python
box = mp.Box.cubic(10.0)

r1 = np.array([[1.0, 1.0, 1.0]])
r2 = np.array([[9.5, 9.5, 9.5]])
```

不考虑周期性的情况下，这两个点看起来相距约14.7埃。根据最小像约定，最短路径穿越周期性边界，实际距离要小得多。

```python
dr = box.diff(r1, r2)
d  = box.dist(r1, r2)

print(f"displacement: {dr}")
print(f"distance:     {d}")
```

对于两组点之间的成对距离，`dist_all` 返回一个 (N, M) 矩阵。

```python
set_a = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
set_b = np.array([[9.5, 9.5, 9.5], [8.0, 8.0, 8.0]])
distances = box.dist_all(set_a, set_b)
print(distances.shape)   # (2, 2)
print(distances)
```

## Frame 上的 Box

盒子以 `frame.box` 的形式附加到 Frame 上，而不是存在 metadata 里。这是将模拟单元格与分子数据关联的标准方式。

```python
frame = mp.Frame(blocks={
    "atoms": {"x": [1.0, 9.5], "y": [1.0, 9.5], "z": [1.0, 9.5]},
})
frame.box = mp.Box.cubic(10.0)

# I/O 读取器会自动设置 frame.box
frame = mp.io.read_lammps_data("system.data", atom_style="full")
print(frame.box.lengths)   # 来自 data 文件头
```

所有计算算子（MSD、RDF 等）都会从 `frame.box` 读取盒子信息。

## 何时需要考虑盒子

只要系统是周期性的，就应该指定 `Box`，不要等到导出到引擎时才考虑。盒子决定了坐标的解读方式——wrap、diff 和 dist 都依赖它。周期性系统如果忽略盒子，分析结论会出错却完全不自知。

当单张快照不足以描述系统随时间的变化时，下一步的抽象是轨迹（trajectory）。

另请参阅：[Block 与 Frame](02_block_and_frame.md)、[轨迹](05_trajectory.md)。
