# 架构概览

MolPy 是一个分层工具包，强调数据流显式可控，尽量减少隐式行为。本文档是架构总览，每个扩展指南都以此为基础：各模块的职责划分、数据模型三层次如何配合、Python 与 molrs Rust 后端的边界在哪。开始使用 [扩展 MolPy](extending-compute.md) 系列前，建议先读完这篇。

## 模块职责

每个包职责明确，同级包之间耦合最小：

| 包 | 用途 |
|---------|---------|
| `core` | 数据结构：`Entity`、`Link`、`Struct`、`Atomistic`、`Frame`、`Block`、`Box`、`ForceField` |
| `parser` | 基于语法的解析：SMILES、SMARTS、BigSMILES、G-BigSMILES、CGSmiles |
| `builder` | 系统组装：链构建器、虚拟位点、AmberTools 集成 |
| `reacter` | 反应框架：基于模板的反应，带有锚点和离去基团选择器 |
| `typifier` | 原子类型分配：OPLS-AA、GAFF、基于 SMARTS/SMIRKS 的自定义类型分配器 |
| `pack` | 打包工作流：Packmol 集成、密度目标 |
| `io` | 文件 I/O：分子数据、轨迹和力场格式的读写器 |
| `compute` | 作用于 `Frame`/`Block` 数据的分析算子 |
| `engine` | MD 抽象：LAMMPS、CP2K、OpenMM 输入生成和执行 |
| `wrapper` | 外部 CLI 工具（antechamber、packmol 等）的子进程边界 |
| `adapter` | 外部对象模型（RDKit、OpenBabel 等）的桥接器 |
| `data` | 捆绑的包数据：力场 XML 文件、参数表 |

`core` 没有任何上层依赖，其他模块都构建在 `core` 之上。`compute`、`io`、`engine` 操作表格层（`Frame`/`Block`）；`parser`、`builder`、`reacter`、`typifier` 操作图结构层（`Atomistic`）。`wrapper` 和 `adapter` 在最外层，不允许外部类型泄漏进 `core`。

## 图结构层：Entity、Link、Struct

数据模型的可编辑部分由三个类层次构成：

1. **Entity**（节点）—— 字典风格的基类，代表原子、珠子或粒子，用身份标识做哈希（`hash()` 就是 `id()`）。即使两个原子属性完全相同，也被视为不同对象。子类：`Atom`、`Bead`。
2. **Link**（边）—— 存有 `Entity` 端点引用，以有序元组形式组织。子类：`Bond`、`Angle`、`Dihedral`、`Improper`、`CGBond`。
3. **Struct**（容器）—— 用 `TypeBucket` 管理实体和链接的增删查改。子类：`Atomistic`、`CoarseGrain`。

`TypeBucket` 按具体类型存储条目：注册 `Atom` 后，`bucket[Atom]` 返回所有 `Atom` 实例；查询父类时也会返回子类实例。新增实体或链接类型必须在 struct 的 `__init__` 中注册 —— 详见[扩展数据模型](extending-core.md)。

## 表格层：Block 和 Frame 基于 molrs

`Frame` 和 `Block` 直接重导出自 [molrs](https://github.com/MolCrafts/molrs) Rust 列存储 —— `molpy.core.frame.Frame` *就是* `molrs.Frame`。每一列都有明确类型（float / int / bool / str），通过零拷贝 NumPy 视图暴露；无法表示的列在写入时直接报错拒绝。`molcrafts-molrs` 是硬运行时依赖，没有纯 Python 回退方案。

图结构到数组的转换是显式发起的：`Atomistic.to_frame()` 委托给 molrs 的原生 `to_frame()`。盒子（box）是一级属性（`frame.box`），不会塞进 metadata。[molrs 后端](molrs-backend.md)页面介绍了邻接表、RDF 和分析算子如何从 Rust 侧暴露。

## 力场：参数与内核分离，内核驻留 Rust

`ForceField` 是一个独立、可查询的数据结构 —— 参数既不嵌入原子，也不隐式推导。模型分为三层：**Style**（函数形式）、**Type**（按类型键索引的参数集）和 **Potential**（可求值的内核）。所有能量/力内核都位于 molrs（`molrs-ff`）中；Python 侧只暴露轻量的命名 `Style` 子类，求值统一走 `ff.to_potentials()`。因此，新增一种函数形式意味着要添加 Rust 内核、Python 样式名和导出格式化器 —— 具体做法见[扩展力场](extending-forcefield.md)。

## 边界转换：格式化器层次

MolPy 内部统一使用规范字段名（`charge` 而非 `q`；`mol_id` 而非 `mol`）；格式特有的名字只出现在 I/O 边界。转换机制定义在 `core/fields.py`：

```text
FieldSpec                              — 规范字段定义（key, dtype, shape, doc）
    ↓
FieldFormatter                         — 数据字段映射：{format_key: FieldSpec}
    ↓                                     canonicalize() / localize() 作用于 Block
ForceFieldFormatter(FieldFormatter)    — 添加参数格式化器：{StyleType: Callable}
```

读取器在退出时调用 `canonicalize()`（格式转规范）；写入器在入口处调用 `localize_frame()`（规范转格式，操作在副本上）。格式特有的子类放在各自 I/O 模块中，`__init_subclass__` 保证每个子类的注册表互不干扰。完整规范名称见[命名约定](../tutorials/naming-conventions.md)附录；扩展方法见[添加 I/O 格式](extending-io.md)。

## 变更约定

核心数据模型 API 就地修改并返回 `self`（或新创建实体）以支持链式调用：`def_atom`、`def_bond`、`get_topo`、`move`、`rotate`、`merge` 都会修改当前结构。`.copy()` 用于显式创建一个独立的深度副本。`builder` 和 `op` 中的辅助函数遵循相反的约定：不能意外修改调用者传入的结构 —— 要么先拷贝，要么新建并返回。

## 构建循环的性能模型

组装的开销对链长是线性的，因为没有任何一次编辑会遍历整个体系：

- **粘贴一次，就地编辑** —— 世界只构建一次；`molrs.Reaction.apply` 就地编辑它，并返回它碰过的原子。没有逐键拷贝，也没有 `entity_map` 重映射——旧构建器四项各自 O(N)/键的开销正出在那里。
- **局部重类型化** —— 只有每根新键周围的那个球会被重新类型化，半径由 typifier 自己声明的 `TypeScope` 决定。相同的连接处按结构哈希去重，所以类型化的开销取决于**不同**化学环境的数量，而不是键的数量。
- **局部拓扑** —— 新键产生的角和二面角直接从该区域插入。组装后的世界上从不调用 `generate_topology`。
- **只匹配一次** —— 内核以 O(N) 匹配反应的模式，把匹配结果交给 `Selector`；配对（唯一的 O(位点²) 步骤）留给需要它的 selector，而 `TopologySelector` 改为按残基索引。

每次连接都不再与链长相关。旧构建器每成一根键就拷贝一次累积结构并重映射实体，仅拷贝一项就让 DP=N 的链付出 O(N²)；组装内核只粘贴一次、就地编辑，之后不再拷贝。剩下的 O(N) 只有开头那一次粘贴，以及 `assemble` 为了不修改调用者传入的世界而做的那一次 `.copy()`。

## 扩展入口

| 我想添加…… | 层 | 指南 |
|---|---|---|
| 一个分析操作 | 插件接口 | [添加计算操作](extending-compute.md) |
| 一种文件格式 | 插件接口 | [添加 I/O 格式](extending-io.md) |
| 一个外部工具集成 | 插件接口 | [添加封装器或适配器](extending-integration.md) |
| 一种实体/链接/结构类型 | 核心内部 —— 请先提交 issue | [扩展数据模型](extending-core.md) |
| 一种相互作用风格/内核 | 核心内部 —— 请先提交 issue | [扩展力场](extending-forcefield.md) |
