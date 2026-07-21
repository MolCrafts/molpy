# 轨迹

`Trajectory` 按时间顺序组织内存中的帧。惰性、可定位的文件访问由 molrs 轨迹读取器提供。

## 一帧远远不够

单个 `Frame` 描述系统在某一时刻的状态。但模拟和分析几乎总是涉及一连串按时间排序的状态。把这些状态存成 Python 列表，小数据集没问题；规模一大，内存管理和惰性访问就成了绕不开的问题。

**`Trajectory` 是一组即时具体化、有序的 `Frame` 对象。**

核心思路是保持和 `Frame` 的连续性。轨迹的每个元素仍然是一个 `Frame`——包含命名的 `Block`、元数据和可选的 `Box`。时间没有替换快照模型，只是把快照按顺序串了起来。

## 从列表构建轨迹

最简单的轨迹来自内存中的帧列表。支持随机访问、`len` 和切片。

```python
import molpy as mp

frames = []
for i in range(5):
    f = mp.Frame()
    f["atoms"] = mp.Block({"x": [float(i)], "y": [0.0], "z": [0.0]})
    f.meta = {"time": mp.MetaValue("f64", i * 10.0)}
    frames.append(f)

traj = mp.Trajectory(frames)
print(len(traj))             # 5
print(traj[0]["atoms"]["x"]) # [0.]
```


## 可迭代对象会立即具体化

构造器接受任意可迭代对象，但会立即将其具体化进原生容器。需要惰性、可定位的磁盘访问时，使用 `molrs.read_lammps_trajectory` 或 `molrs.read_xyz_trajectory`。

```python
def make_frames(n):
    for i in range(n):
        f = mp.Frame()
        f["atoms"] = mp.Block({"x": [float(i)], "y": [0.0], "z": [0.0]})
        f.meta = {"time": mp.MetaValue("f64", i * 0.5)}
        yield f

traj_from_iterable = mp.Trajectory(make_frames(1000))
print(len(traj_from_iterable))   # 1000
```

生成器会在构造时被消耗；文件读取器不会进行这种即时具体化。


## 切片与索引

基于列表的轨迹支持标准的 Python 索引和切片。索引返回单个 `Frame`；切片返回一个新的 `Trajectory`。

```python
first_two = traj[:2]
print(len(first_two))   # 2

strided = traj[::2]
print(len(strided))     # 3

last = traj[-1]
print(last.meta["time"].value)   # 40.0
```

步幅切片（`traj[::n]`）是降采样、快速预览数据的实用手段。


## 使用 map 进行变换

`map` 立即对每一帧应用函数并返回新的轨迹；原始帧保持不变。

```python
def shift_x(frame):
    new = mp.Frame()
    x = frame["atoms"]["x"]
    new["atoms"] = mp.Block({
        "x": x + 10.0,
        "y": frame["atoms"]["y"],
        "z": frame["atoms"]["z"],
    })
    new.meta = frame.meta
    return new

shifted = traj.map(shift_x)
```

```python
shifted_list = list(shifted)
print(shifted_list[0]["atoms"]["x"])   # [10.]
print(traj[0]["atoms"]["x"])           # [0.] — 原数据不变
```


## 何时使用 Trajectory

需要追踪可观测量在多个快照间的变化、计算时间相关性、或遍历 I/O 流——这些场景下时间本身就是科学问题的一部分——应该用 `Trajectory`。如果只需要单个状态，`Frame` 仍然是正确的抽象。

Trajectory 并没有创造一种新的系统状态。它在保持 Frame 语义不变的前提下引入了时间顺序。要点就在这里：同一种结构，多次出现。

另请参阅：[Block 与 Frame](02_block_and_frame.md)、[盒子与周期性](03_box_and_periodicity.md)。
