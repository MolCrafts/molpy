# 单位制

`Frame` 里存的只有裸数字。`UnitSystem` 给这些数字使用的约定命名，当两种约定相遇时负责显式转换。

## 坐标本身无量纲，是约定赋予其意义

`Frame` 里的`x/y/z`、`mass`、`charge` 都是**不带单位**的普通数组。MolPy 既不关心你用什么长度单位，也不强制你选哪一种——真正给这些数字赋予物理意义的是你采用的**约定**，以及你加载的力场。用纳米写的 TIP3P 力场要求输入纳米坐标；用埃写的 OPLS 力场则要求埃坐标。混着用，结果就是静默地出错，物理上完全不对。

**`UnitSystem` 就是干这个的：它命名一种约定，然后在不同约定之间做转换。**
它是 molrs 单位引擎（`molrs.UnitRegistry` / `molrs.Unit` / `molrs.Quantity`）上的一层薄 Python 语法糖：预设名与 LJ 构造在 molpy 侧，解析、量纲运算与换算在 molrs 中执行。

## 使用预设

`UnitSystem` 把 LAMMPS 的 `units` 约定做成命名好的预设，直接拿来用：

```python
from molpy.core.unit import UnitSystem

print(UnitSystem.preset_names())
# ('real', 'metal', 'si', 'cgs', 'electron', 'micro', 'nano')

u = UnitSystem.preset("real")      # LAMMPS 'real'：埃、飞秒、kcal/mol、原子质量单位、e
length = 3.0 * u.angstrom
print(length.to(u.nanometer))      # 0.3 nanometer
```

| 预设 | 约定（LAMMPS `units`） |
|---|---|
| `real` | 埃、fs、kcal/mol、amu、e——OPLS/AMBER 的默认单位。 |
| `metal` | 埃、ps、eV、amu、e。 |
| `si` / `cgs` | SI / CGS 基本单位。 |
| `electron` | 原子（Hartree）单位。 |
| `micro` / `nano` | 微尺度和纳米尺度预设。 |

## 定义自己的预设

注册一个自定义约定后，后续可以按名称重复使用：

```python
UnitSystem.register_preset(
    "my_units",
    base_units={"length": "nm", "time": "ps", "energy": "kJ/mol", "mass": "amu"},
    overwrite=False,
)
u = UnitSystem.preset("my_units")
```

- `base_units` 把每个物理维度映射到一个单位字符串。
- `overwrite=False` 拒绝覆盖已有预设（设成 `True` 就可以替换）。

做粗粒化工作时，`UnitSystem.lj(mass=..., sigma=..., epsilon=...)` 能用你提供的参考
`molrs.Quantity` 量值（例如 `39.948 * UnitSystem().amu`）构建一套约化（Lennard-Jones）单位制。

## 转换量

量是 `molrs.Quantity`。用数字乘以单位属性得到量，再用 `.to(...)` 转换；
`.magnitude` 读出裸数字：

```python
u = UnitSystem.preset("metal")
e = 2.5 * u.eV
print(e.to("J"))                   # 转换能量（每个粒子）
print((5 * u.angstrom).to("nm"))   # 转换长度
print(e.magnitude)                 # 2.5
print((1.0 * u.kilocalorie_per_mole).to("eV").magnitude)
```

`UnitSystem` 还从原生 registry 暴露 `parse`、`define`、`quantity`、`convert`，
需要注册额外单位或相对当前 LJ 尺度换算时可以用它们。

## 注意事项

- **`Frame` 里的数字仍然不带单位。** `UnitSystem` 转换的是你*构造出来的量*，它不会去标记你的坐标数组。输入数据时自己确保单位与力场约定一致。
- **和力场匹配。** 力场用 `real`（埃）写的，就别给它喂纳米坐标。
- `register_preset(..., overwrite=False)` 如果名字已存在会抛异常——确定要替换就传 `overwrite=True`。
- **不是 Pint。** 运行时不再依赖 `pint`，也没有 Pint 特有的 context API；单位运算走 molrs 引擎。

## 参见

- [命名约定](naming-conventions.md) —— 那些无量纲数组遵循的列命名模式。
- [力场](04_force_field.md) —— 约定在这里落地为物理量。
