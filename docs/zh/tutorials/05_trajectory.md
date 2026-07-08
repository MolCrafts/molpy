# 轨迹

`Trajectory` 按时间顺序组织帧的序列，底层实现保持惰性——五帧的小列表和多 GB 的数据流，对外暴露的 API 完全一致。

## 一帧远远不够

单个 `Frame` 描述系统在某一时刻的状态。但模拟和分析几乎总是涉及一连串按时间排序的状态。把这些状态存成 Python 列表，小数据集没问题；规模一大，内存管理和惰性访问就成了绕不开的问题。

**`Trajectory` 是一组有序的 `Frame` 对象，支持惰性求值。数据量再大，系统语义也不会走样。**

核心思路是保持和 `Frame` 的连续性。轨迹的每个元素仍然是一个 `Frame`——包含命名的 `Block`、元数据和可选的 `Box`。时间没有替换快照模型，只是把快照按顺序串了起来。

## 从列表构建轨迹

最简单的轨迹来自内存中的帧列表。支持随机访问、`len` 和切片。

```python
import molpy as mp

frames = []
for i in range(5):
    f = mp.Frame()
    f["atoms"] = mp.Block({"x": [float(i)], "y": [0.0], "z": [0.0]})
    f.metadata["time"] = i * 10.0
    frames.append(f)

traj = mp.Trajectory(frames)
print(len(traj))             # 5
print(traj.has_length())     # True
print(traj[0]["atoms"]["x"]) # [0.]
```


## 基于生成器的轨迹保持惰性

处理大规模或流式数据时，传生成器而不是列表。轨迹按需生成帧，不需要一次加载全部数据到内存。代价是：生成器具体化之前，没法用 `len` 和索引。

```python
def make_frames(n):
    for i in range(n):
        f = mp.Frame()
        f["atoms"] = mp.Block({"x": [float(i)], "y": [0.0], "z": [0.0]})
        f.metadata["time"] = i * 0.5
        yield f

lazy_traj = mp.Trajectory(make_frames(1000))
print(lazy_traj.has_length())   # False

# 迭代而不一次性具体化所有帧
for frame in lazy_traj:
    if frame.metadata["time"] > 2.0:
        break
```

生成器在迭代过程中逐渐消耗。要多次读取同一批数据，可以具体化一个子集，或者每次新建一个生成器。


## 切片与索引

基于列表的轨迹支持标准的 Python 索引和切片。索引返回单个 `Frame`；切片返回一个新的 `Trajectory`。

```python
first_two = traj[:2]
print(len(first_two))   # 2

strided = traj[::2]
print(len(strided))     # 3

last = traj[-1]
print(last.metadata["time"])   # 40.0
```

步幅切片（`traj[::n]`）是降采样、快速预览数据的实用手段。


## 使用 map 进行惰性变换

`map` 对每一帧应用一个函数并返回新的轨迹。变换是惰性的——函数只在访问帧时才执行，调用 `map` 时不会触发。多个变换可以链式组合，不需要预先承担全部计算开销。

```python
def shift_x(frame):
    new = mp.Frame()
    x = frame["atoms"]["x"]
    new["atoms"] = mp.Block({
        "x": x + 10.0,
        "y": frame["atoms"]["y"],
        "z": frame["atoms"]["z"],
    })
    new.metadata = frame.metadata.copy()
    return new

shifted = traj.map(shift_x)
```

由于 `map` 返回的是基于生成器的轨迹，需要迭代或具体化才能看到结果。

```python
shifted_list = list(shifted)
print(shifted_list[0]["atoms"]["x"])   # [10.]
print(traj[0]["atoms"]["x"])           # [0.] — 原数据不变
```


## 何时使用 Trajectory

需要追踪可观测量在多个快照间的变化、计算时间相关性、或遍历 I/O 流——这些场景下时间本身就是科学问题的一部分——应该用 `Trajectory`。如果只需要单个状态，`Frame` 仍然是正确的抽象。

Trajectory 并没有创造一种新的系统状态。它在保持 Frame 语义不变的前提下引入了时间顺序。要点就在这里：同一种结构，多次出现。

另请参阅：[Block 与 Frame](02_block_and_frame.md)、[盒子与周期性](03_box_and_periodicity.md)。
