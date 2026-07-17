# 扩展力场

演示如何添加自定义的相互作用类型，以及如何为它们注册导出格式化器。

!!! note "构建前请先讨论"
    添加新的相互作用类型要动两个仓库：molrs 的 Rust 内核和本仓库的 Python 类型与格式化器。动手之前，先在 [GitHub issue](https://github.com/MolCrafts/molpy/issues) 里写明函数形式；[架构概览](architecture-overview.md) 说明了各组件的分布。

## 计算归属

力场模型——`ForceField`、`Style` 类型树、`Type` 类型树以及全部能量/力内核——都归属 **molrs** Rust 扩展。molpy 不维护另一套 Python 势能层。没有 `style.to_potential()` 方法，也没有 Python 侧的内核类。计算统一通过

```python
ff.to_potentials().calc_energy(frame)   # 以及 .calc_forces(frame)
```

完成。

这改变了"扩展力场"的含义：

1. **内核**——数值形式（能量 + 力）在 molrs（`molrs-ff`，Rust）中实现并注册，`ForceField` 能按类型名称自动分发。
2. **命名类型**——在 Python 侧定义一个轻量的 `Style` 子类，只负责固定类型名称。这样调用方写 `ff.def_style(BondMorseStyle())` 而非 `ff.def_bondstyle("morse")`。
3. **格式化器**——针对每个导出后端（LAMMPS、GROMACS、XML）序列化新类型的参数。

如果 molrs 已经有了需要的内核，只做步骤 2 和 3 就够了（步骤 2 可能已经有人写过）。添加全新的函数形式要从步骤 1 开始。


## 步骤 1：在 molrs 中添加内核

新的函数形式（比如 Morse 键）在 `molrs-ff` crate 中实现。写好能量和力的表达式，把内核注册到对应的类型名上，`ForceField::to_potentials` 就能找到它。然后重新构建 molrs wheel（`maturin develop` / `maturin build`）再安装；molpy 会自动识别新内核——因为它直接 re-export molrs 的层次结构。

注册完成后，类型名可直接用于通用辅助方法：

```python
import molrs
import molpy as mp

ff = mp.ForceField(name="custom", units="real")
a_style = ff.def_atomstyle("full")
c = a_style.def_type("C", mass=12.011)
o = a_style.def_type("O", mass=15.999)

bond_style = ff.def_bondstyle("morse")          # 分发到 molrs 内核
bond_style.def_type(c, o, D=100.0, alpha=1.8, r0=1.43)
```


## 步骤 2：暴露一个轻量的命名 Style

为方便使用和查找，给内核提供一个命名的 `Style` 类。这个类**不携带**内核，也**没有** `to_potential()` 方法——它只管通过 `_name_default` 固定类型名称。把类放在 `molpy/core/forcefield.py` 里已有的类型旁边（需要的话也可以从 `molpy.potential` 再导出）。

```python
from molpy.core.forcefield import BondStyle

class BondMorseStyle(BondStyle):
    """键 ``morse`` 类型（LAMMPS ``bond_style morse``）。"""

    def _name_default(self) -> str:
        return "morse"
```

类型和参数通过 molrs 原生透传，不需要写 `def_type()` 重写方法。通过 `def_style` 注册命名类型：

```python
morse = ff.def_style(BondMorseStyle())
morse.def_type(c, o, D=100.0, alpha=1.8, r0=1.43)
```


## 步骤 3：注册参数格式化器

每个导出后端都有对应的 `ForceFieldFormatter` 子类。这些子类继承自该格式的 `FieldFormatter`（负责数据字段名映射），并扩展了 `_param_formatters`（负责 Style/Type 参数序列化）。

在对应的子类上注册参数格式化器：

```python
from molpy.core.forcefield import BondMorseStyle
from molpy.io.forcefield.lammps import LammpsForceFieldFormatter

def _format_morse_bond(typ) -> list[float]:
    """为 LAMMPS 格式化 Morse 键参数：D alpha r0"""
    p = typ.params.kwargs
    return [p["D"], p["alpha"], p["r0"]]

LammpsForceFieldFormatter.register_param_formatter(BondMorseStyle, _format_morse_bond)
```

每个后端都要重复一遍。注册是**按子类隔离的**——给一个后端添加格式化器不会影响其他后端。`__init_subclass__` 通过复制注册表来保证隔离。


## 使用自定义相互作用

建好模型后，在指定了类型的 `Frame` 上计算能量：

```python
import molpy as mp
import numpy as np
from molpy.core.forcefield import BondMorseStyle

ff = mp.ForceField(name="custom", units="real")
a_style = ff.def_atomstyle("full")
c = a_style.def_type("C", mass=12.011)
o = a_style.def_type("O", mass=15.999)

morse = ff.def_style(BondMorseStyle())
morse.def_type(c, o, D=100.0, alpha=1.8, r0=1.43)

# 两个原子恰好位于 r0 → Morse 能量为 0。
frame = molrs.Frame()
atoms = molrs.Block()
atoms.insert("x", np.array([0.0, 1.43]))
atoms.insert("y", np.array([0.0, 0.0]))
atoms.insert("z", np.array([0.0, 0.0]))
frame["atoms"] = atoms
bonds = molrs.Block()
bonds.insert("atomi", np.array([0], dtype=np.uint32))
bonds.insert("atomj", np.array([1], dtype=np.uint32))
bonds.insert("type", np.array(["C-O"], dtype=str))
frame["bonds"] = bonds

pots = ff.to_potentials()
print(pots.calc_energy(frame))   # 在 r0 处为 0.0
```


## 检查清单

- [ ] 在 `molrs-ff`（Rust）中实现并注册内核，重新构建 wheel
- [ ] 验证内核：平衡位置能量 = 0，远离平衡位置单调递增
- [ ] 在 `molpy/core/forcefield.py` 中创建轻量命名 `Style` 子类（仅 `_name_default`）
- [ ] 为每个写入器后端（LAMMPS、GROMACS、XML）注册格式化器
- [ ] 编写测试：类型创建、`to_potentials().calc_energy(frame)` 值、导出往返测试
- [ ] 测试文件位于 `tests/test_core/test_forcefield.py` 和 `tests/test_io/test_forcefield/`
