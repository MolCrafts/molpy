# 3D 构象生成

从 SMILES 到分子模拟，中间缺的就是三维坐标。`Conformer` 用 molrs 的生成器算出来，每步都留个记录。

## 构象生成解决的问题

解析器从 SMILES 读进来的是一个**图**——只有元素和键，没有几何信息。类型化（需要 3D 结构的 SMARTS 匹配）、填充、力场评估、导出到模拟引擎……几乎每个后续步骤都依赖真实坐标。

**`Conformer` 用 molrs 的 Rust 生成器把图嵌入三维空间，返回一个新的 `Atomistic` 对象和一份分阶段报告。** 缺氢会自动补齐，输入的图保持不变（molrs 内部做了克隆）。

## 生成构象

```python
from molpy.parser import parse_molecule
from molpy.conformer import Conformer

mol = parse_molecule("CCO")                    # 乙醇图（仅重原子）
mol_3d, report = Conformer(seed=42).generate(mol)

print(mol_3d.n_atoms)          # 9 — 重原子 + 添加的氢原子
print(report.final_energy)     # 返回结构的能量
```

`generate` 返回一个**元组**：新的 `Atomistic`（含坐标和添加的氢）和一个 `ConformerReport`。输入的 `mol` 不会变，同一张图可以反复调用，每次得到独立的构象。

## 构造函数参数

`Conformer` 继承 `molrs.Conformer`，构造函数参数完全一致：

| 参数 | 说明 |
|---|---|
| `speed` | 嵌入与优化轮次的速度/质量权衡。越快，优化步数越少。 |
| `add_hydrogens` | 嵌入前是否用显式氢补齐化合价。除非图已经带了所有氢，否则保持默认值就好。 |
| `seed` | 随机嵌入的 RNG 种子。**设好种子才能复现几何**——不设的话每次跑出来的构象都不一样。 |

带电原子需要在 `"formal_charge"` 键上设定规范的整型值（解析器会生成这个键），这样 molrs 才知道 `[N+]` / `[N-]` 该补多少个氢。

## 解读报告

`ConformerReport` 汇总一次运行的结果：

- `final_energy` — 返回结构的能量。
- `stages` — `ConformerStageReport` 列表，一个阶段一条。
- `warnings` — 生成过程中的非致命问题。

每个 `ConformerStageReport` 记录 `stage`、`steps`、`converged`、`energy_before`、`energy_after` 和 `elapsed_ms`：

```python
for s in report.stages:
    status = "converged" if s.converged else "hit step limit"
    print(f"{s.stage}: {s.energy_before:.3f} -> {s.energy_after:.3f} "
          f"({s.steps} steps, {status})")
```

如果某阶段 `converged = False`，表示它用完了步数上限——几何结构能用但没完全松弛。试试更慢的 `speed`。

## 注意事项

- **不设 `seed` 结果不可复现。** 固定 `seed` 才能确保两次结果一致。比较或缓存几何结构时别忽略这一点。
- 空图（没有原子）会抛 `ValueError`。先解析再生成构象。
- 这是**单**构象嵌入器，不是构象系综搜索。想要多个构象就用不同种子多跑几次。

## 另请参阅

- [解析化学结构](01_parsing_chemistry.md) — 构造输入图。
- [力场类型化](06_typifier.md) — 下一步，需要 3D 结构。
- [几何优化](08_geometry_optimization.md) — 分子有了力场之后做松弛。
