# 架构概览

MolPy 是一个分层工具包，强调数据流显式可控，尽量减少隐式行为。本文档是架构总览，每个扩展指南都以此为基础：各模块的职责划分、数据模型三层次如何配合、Python 与 molrs Rust 后端的边界在哪。开始使用 [扩展 MolPy](extending-compute.md) 系列前，建议先读完这篇。

## 模块职责

每个包职责明确，同级包之间耦合最小：

| 包 | 用途 |
|---------|---------|
| `core` | molrs-backed 图引用/world、`Frame`、`Block`、`Box`、单位与力场表面 |
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

## 图结构层：molrs world 的实时句柄视图

可编辑图只有一份实现：molrs world。Python 暴露三个协作表面：

1. **节点引用** —— `Atom`、`Bead` 和虚拟位点变体是由稳定 native handle
   标识的字典风格实时视图。
2. **关系引用** —— `Bond`、`Angle`、`Dihedral`、`Improper` 和 `CGBond`
   在同一个 world 中解析端点 handle。
3. **World** —— `Atomistic` 与 `CoarseGrain` 持有节点、关系、列和图算法；
   `.atoms`、`.bonds` 等属性是 molrs 实时视图集合，不是 Python bucket 或镜像 list。

系统中不再有 `Struct`/`TypeBucket` 注册层。新增可存储的节点或关系种类需要修改
molrs schema 与绑定，而不是继承一个 Python 类。详见[扩展数据模型](extending-core.md)。

## 表格层：Block 和 Frame 基于 molrs

`Frame` 和 `Block` 只由 [molrs](https://github.com/MolCrafts/molrs) Rust 列存储拥有和导出。从 molpy 导入（`from molpy import Frame, Block`）；它们是对 molrs 的 identity re-export。每一列都有明确类型（float / int / bool / str），通过零拷贝 NumPy 视图暴露；无法表示的列在写入时直接报错拒绝。`molcrafts-molrs` 是硬运行时依赖，没有纯 Python 回退方案。

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

组装的开销对链长是线性的，因为增长中的整图不会在每次编辑后重新类型化：

- **先编译，后执行** —— selector 先给出完整 binding 集。编译器在仍然完整的用户模板上叠加所有计划成键，给每个连接处物化一个有界的产物 motif。带残基信息时，motif 由完整 monomer 组成，无需用氢补全切断的分子图。
- **带根的局部缓存** —— 缓存键同时覆盖产物结构、化学标量标签和 touched 根。相同连接环境即使跨多次 build 也只类型化一次。缓存值只保存 `type`、`charge`、pair 参数等 per-atom 标量，不把局部 angle/dihedral 行复制回整图。
- **一次批量编辑** —— `molrs.Reaction.apply_many` 在完整图上解析全部离去基团，一次扫描删除它们的并集，再执行所有 transform；不存在“增长一次、重算累积聚合物、再增长”的循环。
- **显式最终化** —— `Finalization.ATOMS` 在原子写回后停止；默认的 `TOPOLOGY` 只在最后统一生成一次 angle/dihedral；`BONDED` 再用 `ForceFieldParams` 统一参数化。超大体系可一直保持 atoms-only，到 MD 输出时再生成拓扑。
- **只匹配一次** —— 内核以 O(N) 匹配反应的模式，把匹配结果交给 `Selector`；配对（唯一的 O(位点²) 步骤）留给需要它的 selector，而 `TopologySelector` 改为按残基索引。

每个 binding 只做有界的局部编译；整图只做一次批量 reaction，以及最多一次用户要求的最终化。旧构建器每成一根键就拷贝并重新处理累积结构，因此 DP=N 会退化为 O(N²)；新路径不再含这个增长循环。

## 扩展入口

| 我想添加…… | 层 | 指南 |
|---|---|---|
| 一个分析操作 | 插件接口 | [添加计算操作](extending-compute.md) |
| 一种文件格式 | 插件接口 | [添加 I/O 格式](extending-io.md) |
| 一个外部工具集成 | 插件接口 | [添加封装器或适配器](extending-integration.md) |
| 一种实体/链接/结构类型 | 核心内部 —— 请先提交 issue | [扩展数据模型](extending-core.md) |
| 一种相互作用风格/内核 | 核心内部 —— 请先提交 issue | [扩展力场](extending-forcefield.md) |
