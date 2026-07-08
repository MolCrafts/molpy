# 极化与虚拟位点模型

Drude 壳模型和 TIP4P 的 M 位点都把电荷放在原子核以外的地方。
`molpy.builder.virtualsite` 构建器为结构副本添加这些**虚拟位点**，
并自动重分配电荷。

## 虚拟位点的用途

有些力场把相互作用位点放在原子核之外：

- **Drude 振子**（CL&Pol）在重原子上挂一个可移动的带电壳层，用来描述诱导极化。
- **TIP4P 水模型**把负电荷放在 HOH 角平分线上一个偏离氧原子的 **M 位点**，
  而不是直接放在氧原子上。

**`VirtualSiteBuilder` 复制结构、选择宿主原子、构建额外位点、重分配电荷——整个过程不改动原输入。**

## 构建器接口

每个构建器都走相同的四步流水线，入口统一为 `apply`：

```python
new_struct = builder.apply(struct)      # struct: Atomistic -> Atomistic（副本）
```

`apply` 内部按顺序执行 `select`（选宿主）→ `build_sites`（创建额外粒子）→ `redistribute`（把电荷转移到新粒子上）。通常只调用 `apply` 就够了；剩下三个方法是留给自定义构建器的扩展钩子。

## Drude 极化（CL&Pol）

`DrudeBuilder` 给每个可极化的重原子添加一个 Drude 壳层（`DrudeParticle`），
按原子类型的极化率分配参数：

```python
from molpy.builder.virtualsite import DrudeBuilder, load_polarizability

alpha = load_polarizability()                 # 内置的 alpha.ff 参数
drude = DrudeBuilder(polarizability=alpha, drude_prefix="D")
polarized = drude.apply(struct)
```

| 参数 | 说明 |
|---|---|
| `polarizability` | Drude 参数，类型为 `dict[类型 -> dict[参数 -> float]]`。设为 `None` 则回退到内置的 `alpha.ff`；`load_polarizability(path)` 读取自定义文件。 |
| `drude_prefix` | 生成的 Drude 粒子名称前缀（默认 `"D"`）。 |

## TIP4P M 位点

`Tip4pBuilder` 在每个水分子的 HOH 角平分线上放一个 `MasslessSite`——接口相同，规则不同：

```python
from molpy.builder.virtualsite import Tip4pBuilder

tip4p = Tip4pBuilder(d_om=0.1546)     # O–M 距离，单位为 nm
water4p = tip4p.apply(water)
```

`d_om` 是氧原子到 M 位点的距离；默认值与 TIP4P 几何参数一致。

## 编写自定义构建器

继承 `VirtualSiteBuilder`，实现 `select`、`build_sites` 和 `redistribute`；
`apply` 编排这些步骤并处理结构复制。`DrudeBuilder` 和 `Tip4pBuilder` 是两个现成的参考实现。

## 注意事项

- **`apply` 返回副本**——原始 `struct` 不变，使用返回值。
- Drude 输出需要能识别壳层粒子的力场，应与 CL&Pol 或可极化类型分配路径配合使用，而非普通的固定电荷力场。
- `MasslessSite` / `DrudeParticle` 是辅助粒子——下游导出器和引擎需要把它们当作虚拟位点，而不是原子。

## 另请参阅

- [力场类型分配](06_typifier.md)——参数分配，包括 CL&Pol 路径。
- [API 参考——构建器](../../api/builder.md)——完整的 `virtualsite` 参考文档。
