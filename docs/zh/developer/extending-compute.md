# 添加计算操作

向 MolPy 添加可重用分析操作的规范流程如下。这些操作都继承自 `Compute`。

## 选哪个基类

| 需求 | 基类 | 示例 |
|------|-----------|---------|
| 对数组数据做可重用分析 | `Compute` | `MSD`、`DisplacementCorrelation` |
| 一次性计算，用完即弃 | 普通函数 | `compute_msd(positions)` |

`Compute` 是冻结数据类。配置在构造时设定，之后只读。计算逻辑通过 `run()`（或 `__call__`）执行。

## 添加 Compute 操作

继承 `Compute`，把配置声明为冻结数据类字段，然后实现 `run()`。

```python
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from molpy.compute import Compute

@dataclass(frozen=True)
class RadiusOfGyration(Compute):
    """计算每一帧的回转半径。"""

    use_masses: bool = True

    def run(self, positions: NDArray, masses: NDArray | None = None) -> NDArray:
        """计算一组位置的回转半径 Rg。

        Args:
            positions: 形状 (n_atoms, 3)
            masses: 形状 (n_atoms,)，可选

        Returns:
            标量 Rg 值。
        """
        if self.use_masses and masses is not None:
            w = masses / masses.sum()
        else:
            w = np.ones(len(positions)) / len(positions)

        com = (positions * w[:, None]).sum(axis=0)
        dr = positions - com
        rg2 = (w * (dr ** 2).sum(axis=1)).sum()
        return float(np.sqrt(rg2))
```

使用方式：

```python
rg = RadiusOfGyration(use_masses=True)
value = rg(positions, masses)   # __call__ 委托给 run()
```

## 设计原则

1. **配置放字段** — 初始化时设一次，冻结后不改
2. **数据通过参数传入 `run()`** — 同一接口处理不同输入
3. **不修改输入** — `run()` 返回新对象，不触碰原始数据
4. **`run()` 职责单一** — 只做一件事，别写成工作流引擎
5. **便于隔离测试** — 每个 `Compute` 都能用合成数据独立验证

## 检查清单

- [ ] 继承 `Compute`
- [ ] 添加 `@dataclass(frozen=True)` 装饰器
- [ ] 带类型提示实现 `run()`
- [ ] 在 `tests/test_compute/` 中编写测试
