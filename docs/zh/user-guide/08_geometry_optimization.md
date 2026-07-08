# 几何优化

新构建的分子结构往往带有应变。`molpy.optimize` 通过力场弛豫将其推至局部极小值，并报告优化终止的原因。

## 何时需要

新构建或填充的结构几乎不可能刚好落在力场极小值上——键长、键角、原子间近距离接触都蕴含过量能量。生产模拟之前，或需要有意义地比较能量时，必须对几何结构做极小化处理，将原子受力降至阈值以下。

**`Optimizer` 沿势能面将原子向低处移动，直至最大力 `fmax` 低于设定阈值。** MolPy 提供 `LBFGS`，一种有限内存拟牛顿极小化器，通过 `ForceFieldPotential` 评估力场。

## 极小化结构

`molpy.optimize` 可直接导入（未通过 `mp.optimize` 暴露）：

```python
from molpy.optimize import LBFGS, ForceFieldPotential

potential = ForceFieldPotential(forcefield)      # 包装一个 molrs ForceField
opt = LBFGS(potential)
result = opt.run(frame, fmax=0.05, steps=200)    # 原地弛豫 frame

print(result.converged, result.energy, result.fmax, result.nsteps)
print(result.reason)                             # 停止原因
```

`run` 返回 `OptimizationResult`；默认**原地**优化 `frame`（`inplace=True`）。传入 `inplace=False` 则保持输入不被修改。

## 参数

`LBFGS(potential, *, maxstep=0.04, memory=20, damping=1.0)`：

| 参数 | 作用 |
|---|---|
| `maxstep` | 每步最大原子位移（Å）。值越小越稳定，但收敛也越慢；仅在收敛缓慢且体系稳定时考虑增大。 |
| `memory` | L-BFGS Hessian 近似保留的历史步数。内存越大，曲率估计越准确，但存储开销也越大。 |
| `damping` | 对每一步的提议步长缩放（`1.0` = 无阻尼）。优化器在刚性体系上容易过冲时，可适当降低此值。 |

`run(frame, fmax=0.01, steps=1000, *, inplace=True)`：

| 参数 | 作用 |
|---|---|
| `fmax` | 最大力收敛阈值（eV/Å）。当所有原子受力均低于此值时，优化停止。 |
| `steps` | 迭代次数硬上限——未达到 `fmax` 时的安全网。 |

## 解读结果

`OptimizationResult` 包含 `frame`、`energy`、`fmax`、`nsteps`、`converged` 和 `reason`。务必检查 `converged`：因达到 `steps` 上限而终止的运行（`converged = False`）**未**抵达极小值——应放宽 `fmax`、提高 `steps` 上限，或检查结构是否合理。

如需观察收敛过程，可注册每 `interval` 步触发的回调：

```python
opt.attach(lambda: print(opt.step(frame)), interval=10)   # 每次调用返回 (energy, fmax)
```

## 注意事项

- **未收敛 != 已极小化。** `converged` 为 `False` 且 `reason` 提到步数上限，说明提前终止了优化。
- **`fmax` 的单位是 eV/Å。** 对粗糙力场而言，阈值设得太紧可能永远无法收敛；设得太松则会残留应变。
- 优化需要**已完成原子类型分配**的 frame 和对应的力场——先运行 typifier，否则 `ForceFieldPotential` 无内容可评估。

## 另请参阅

- [力场](../tutorials/04_force_field.md) — 构建优化所依赖的 `ForceField`。
- [3D 构象生成](07_conformers.md) — 力场弛豫之前的图嵌入步骤。
- [引擎](12_engine.md) — 极小化后运行全动力学模拟。
