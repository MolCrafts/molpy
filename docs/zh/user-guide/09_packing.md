# 将体系填充到模拟盒中

把几百个分子按几何约束塞进模拟盒子，还不产生空间冲突 — 用 molpy 的 **`Packmol` 包装器**，在 Python 里直接搞定。

!!! note "前置条件"
    `Packmol` 是 molpy 对 [Packmol](https://m3g.github.io/packmol/) 可执行文件的**轻量包装器**：它负责写输入脚本、调用 `packmol` 外部进程、把结果读回 `Frame`。使用前请安装 Packmol 并将 `packmol` 加入 `PATH`（或显式指定路径：`Packmol(executable="/path/to/packmol")`）。

!!! tip "更倾向于纯 Python 的填充工具？试试 `molcrafts-molpack`"
    我们的 [**molcrafts-molpack**](https://molcrafts.github.io/molpack/) 是一个纯 Python 原生填充工具，Rust 实现的 Packmol 移植版，支持集体约束且**无需外部二进制文件**。执行 `pip install molcrafts-molpack`；它使用相同的 molrs `Frame` 接口，可以无缝接入以下工作流。不妨一试。

## 填充解决的问题

构建一个分子只能得到一个分子。模拟需要的是**盒子**——往往要几百上千个分子不重叠地排列在一起。手动操作（网格、随机插入）要么浪费体积，要么产生重叠。

**molpy 的 `Packmol` 包装器把填充目标（分子 + 数量 + 允许区域）收拢起来，交给 Packmol 无冲突地摆进盒子，最后返回一个填充好的 `Frame`。**

## 填充一个盒子

```python
from molpy.pack import Packmol, InsideBoxConstraint

p = Packmol(workdir="pack_out")                      # Packmol 临时文件的工作目录
p.def_target(
    water,                                   # 一个 molrs Frame（单个分子）
    number=500,                              # 拷贝份数
    constraint=InsideBoxConstraint(length=30.0),   # 限制在 30 Å 立方体内
)
packed = p(max_steps=1000, seed=42)  # -> 一个填充好的 Frame
```

调用 `p(...)` 就会运行 Packmol 并直接返回填充好的 `Frame`。注册多个 `def_target` 可以在一次运行中填充混合物（例如溶质 + 溶剂）。

## 组件说明

| 对象 | 作用 |
|---|---|
| `Packmol(executable=None, workdir=None)` | molpy 的 Packmol **包装器**，管理填充任务。`workdir` 存放 Packmol 的临时文件；`executable` 指向特定的 `packmol` 二进制文件。 |
| `def_target(frame, number, constraint)` | 注册 `number` 份 `frame` 拷贝，受 `constraint` 约束。返回 `Target`。 |
| `packer(max_steps=1000, seed=None, pbc=None)` | 运行 Packmol 并返回填充好的 `Frame`。`pbc` 提供周期性盒子用于最小镜像距离。 |

### 约束目录

约束限制目标的拷贝**可以放在哪里**。用 `AndConstraint` / `OrConstraint` 组合多个约束。

| 约束 | 效果 |
|---|---|
| `InsideBoxConstraint(length, origin=(0,0,0))` | 分子位于原点处边长为 `length` 的轴对齐立方体**内部**。 |
| `OutsideBoxConstraint(origin, lengths)` | 分子位于盒子**外部**（挖出一个空腔）。 |
| `InsideSphereConstraint(radius, center)` | 分子位于球体**内部**。 |
| `OutsideSphereConstraint(radius, center)` | 分子位于球体**外部**。 |
| `MinDistanceConstraint(dmin)` | 分子彼此相距至少 `dmin`（避免碰撞）。 |
| `AndConstraint(a, b)` / `OrConstraint(a, b)` | 同时满足 / 满足任一子约束。 |

```python
from molpy.pack import InsideSphereConstraint, MinDistanceConstraint, AndConstraint

# 在 20 Å 球体内，且彼此间距不小于 2.5 Å
c = AndConstraint(
    InsideSphereConstraint(radius=20.0, center=(0.0, 0.0, 0.0)),
    MinDistanceConstraint(dmin=2.5),
)
```

## 重要参数

- **`number`** — 每个目标的拷贝数。总原子数 = Σ(份数 × 每分子原子数)，直接影响内存和 Packmol 运行时间。
- **`max_steps`** — Packmol 的放置步数。盒子难以收敛时增大此值，快速草稿时可减小。
- **`seed`** — 控制随机种子，固定后可重现完全相同的填充结果。
- **盒子尺寸 vs `number`** — 体积内分子过多会让 Packmol 难以收敛甚至失败；留出余量，或分阶段填充。

## 常见陷阱

- **找不到 `packmol`** — 运行立即失败。安装 Packmol，或将路径传给 `Packmol(executable=...)`。
- **盒子过密**导致无法收敛。如果 `optimize` 停滞，放大盒子约束或减少 `number`。
- 返回的是普通 `Frame`；后续步骤如果需要周期性盒子，自行附加盒子/周期性信息。

## 另请参见

- [构建聚合物](02_polymer_stepwise.md) — 生成待填充的分子。
- [多分散体系](05_polydisperse_systems.md) — 填充链长分布。
- [API 参考 — 填充](../../api/pack.md) — 完整的类和约束参考。
