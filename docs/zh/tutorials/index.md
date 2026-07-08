# 教程

学习 MolPy 推荐按顺序来：先让它跑起来，再弄懂背后的数据模型。手头有具体任务（构建、类型分配、填充、导出）的话，可以直接跳到[指南](../user-guide/index.md)。不过指南默认你已经看过本教程。

## 快速上手

1. **[安装](../getting-started/installation.md)** —— 安装包并验证导入成功。*约 2 分钟*
2. **[快速入门](../getting-started/quickstart.md)** —— 六行代码跑通完整管线，然后用全控制方式构建一个 TIP3P 水盒子。*约 10 分钟*
3. **[示例集锦](../getting-started/examples.md)** —— 即用即贴的工作流：小分子、填充盒子、聚合物、虚拟位点。
4. **[常见问题](../getting-started/faq.md)** —— MolPy 为何存在，它与 RDKit / ASE / mBuild 的关系，以及何时选择其他工具更合适。

如果 MolPy 已安装，以下代码可直接运行——无需可选依赖，甚至不需要 RDKit：

```python
import molpy as mp

water = mp.Atomistic(name="water")
o  = water.def_atom(element="O", x=0.000, y=0.000, z=0.000)
h1 = water.def_atom(element="H", x=0.957, y=0.000, z=0.000)
h2 = water.def_atom(element="H", x=-0.239, y=0.927, z=0.000)
water.def_bond(o, h1)
water.def_bond(o, h2)

frame = water.to_frame()
mp.io.write_pdb("water.pdb", frame)
print(f"Wrote {frame['atoms'].nrows} atoms to water.pdb")
```

输出 `Wrote 3 atoms to water.pdb` 说明环境就绪。这几行代码触及了 MolPy 最核心的设计思路：**在图（`Atomistic`）上进行化学编辑，在数组（`Frame`）上进行计算和导出**。后面各章节会详细解释这种分工，以及搭建在它上面的所有功能。

## 数据模型

MolPy 围绕三个核心理念设计：**在图上编辑，在数组上计算和导出，参数保持分离**。下面章节逐一讲解这些概念背后的数据结构——每个对象是什么、为什么存在、界限在哪。第一次读请按顺序来；之后每章都可以独立查阅。

- **身份 vs 数据。** 实体（原子、连接键）代表唯一身份；批量数据存放在列式数组中。两个属性完全相同的原子仍然是不同的原子——它们可以参与不同的键、选择和编辑操作。
- **图 → 数组。** 以图的形式构建和编辑（`Atomistic`）；以数组的形式计算和导出（`Frame`）。转换是显式的（`Atomistic.to_frame()`）。
- **衍生拓扑。** 键角和二面角按需从键*衍生*出来，而非手动存储和维护。键图变化时，不会有过时的缓存。
- **参数是分离的。** 力场类型独立于结构，因此系统总能可靠地重建并重新分配类型。

典型流程：

> `Atomistic`（编辑）→ 衍生拓扑 → `Frame`（数组）→ I/O → 模拟 → 分析

### 表示层次结构

下图展示了 MolPy 管线中的标准数据流。每个节点代表一个核心数据结构；每条边代表一次显式转换。

```text
                    ┌─────────────────────────┐
  SMILES / 文件     │  Atomistic              │
  ────parser────>   │  (可编辑的分子图：        │
                    │   原子 + 键)              │
                    └───────────┬─────────────┘
                                │
                  typifier + ForceField
                                │
                    ┌───────────▼─────────────┐
                    │  带类型的 Atomistic      │
                    │  (原子携带类型、电荷、     │
                    │   力场参数)               │
                    └───────────┬─────────────┘
                                │
                          .to_frame()
                                │
                    ┌───────────▼─────────────┐
                    │  Frame                   │
                    │  (Block 表 +              │
                    │   Box + 元数据)           │
                    └───────────┬─────────────┘
                                │
                          io.write_*
                                │
                    ┌───────────▼─────────────┐
                    │  LAMMPS / GROMACS /      │
                    │  PDB / HDF5 文件         │
                    └─────────────────────────┘
```

**`Atomistic`** 是主要的编辑界面。原子的添加和删除、键的形成、反应执行以及结构组装，都在这个结构上操作。

**`Frame`** 是主要的数值计算界面。向量化距离计算、文件 I/O 和引擎导出都在 `Frame` 对象及其组成的 `Block` 表上操作。

**`ForceField`** 是一个独立的数据结构，随分子系统一起传递。力场参数既不嵌入原子中，也不隐式推导，而是存储在一个可查询的带类型字典中。

**`Box`** 指定周期性模拟盒子，并作为一等属性（而非元数据）附加到 `Frame` 上。

**`Trajectory`** 是一个按时间排序的 `Frame` 对象序列，为大数据集提供惰性访问模式。

### 章节地图

| 层 | 描述 | 深入学习 |
|-------|-----------|----------|
| **实体与连接** —— `Atom`、`Bond`、`Angle`、`Dihedral` | 身份优先的图模型，用于构建和编辑 | [Atomistic 与拓扑](01_atomistic_and_topology.md) |
| **拓扑** | 键角/二面角以及从键图*衍生*的 k 跳查询——没有独立的类；使用 `get_topo()` / `get_topo_neighbors()` / `get_topo_distances()` | [Atomistic 与拓扑](01_atomistic_and_topology.md) |
| **Block 与 Frame** | 列式表（`atoms`、`bonds`……）加上盒子与元数据——写入器和计算模块操作的数据载体 | [Block 与 Frame](02_block_and_frame.md) |
| **Box** | 模拟盒子 + 周期性边界（包裹、最小镜像距离） | [Box 与周期性](03_box_and_periodicity.md) |
| **ForceField 与 Typifier** | 参数目录（风格 + 类型表）以及一套将类型分配到结构上的规则引擎 | [力场](04_force_field.md) |
| **Trajectory** | 按序排列的 `Frame` 对象序列 | [轨迹](05_trajectory.md) |
| **Selector** | 在 `Block` 列上基于谓词的可组合原子过滤器 | [选择器](06_selector.md) |
| **Wrapper 与 Adapter** | 子进程执行边界以及与外部工具之间的内存表示桥接 | [包装器与适配器](07_wrapper_and_adapter.md) |
| **CoarseGrain** | 粗粒化模型的 `Bead` / `CGBond` 图 | [粗粒化结构](08_coarsegrain.md) |
| **Units** | 单位系统预设与量值显式转换 | [单位](09_units.md) |

## 附录

- [命名约定](naming-conventions.md) —— 数据模型中使用的规范字段名和拓扑键规则
- [词汇表](glossary.md) —— 核心数据结构和模块的简明定义

只记住一句话就好：**在图上编辑，在数组上计算和导出。**
