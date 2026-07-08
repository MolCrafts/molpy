## MolPy 中的命名约定

### 约定的内容

MolPy 的命名约定在两个层面运作：命名空间组织和字段命名。命名空间层面把分子数据划入语义分组，如 `atoms`、`bonds`、`angles`、`dihedrals`。每个命名空间对应一个物理或拓扑类别，作为数据层次中的一级键。字段层面以二级键形式在父命名空间下访问各个属性，命名上区分整数原子索引和对象引用。

MolPy 支持两种互补的分子拓扑表示，各自针对不同的建模阶段做了优化。Frame 和 Block 是数据交换层，侧重序列化、存储、数值处理和外部引擎互操作。Entity 是交互构建层，侧重结构构建、图遍历和化学感知操作。两种表示描述的是同一物理拓扑，但抽象层次不同，因此使用不同的命名方案，避免语义混淆。

Frame 和 Block 层面完全用整数原子索引表示拓扑。索引从 0 开始，指向 atoms block 中的行。这一层使用 `atomi`、`atomj`、`atomk`、`atoml` 作为字段名。例如：

```python
frame["bonds"] = Block({
    "type": ["C-H"],
    "atomi": [0],
    "atomj": [1],
})
```

这里 `atomi` 和 `atomj` 是指向 atoms block 中位置的整数。规则很简单：`atomi`、`atomj`、`atomk`、`atoml` 必须存整数，不能存 Atom 对象。

### 带命名空间的 Frame 模式

下面定义 Frame 和 Block 结构的完整模式。每个命名空间按物理含义和使用模式对相关字段分组。字段通过字符串键访问，例如 `frame["atoms"]` 取 atoms Block，`frame["atoms"]["x"]` 取位置。

#### 原子属性（`atoms`）

`atoms` 命名空间存放每个原子的属性：原子序数、位置、电荷、可选标识符等。所有数组长度为 `N`，即原子总数。原子位置拆成三个独立的一维数组（`x`、`y`、`z`），这是 MolPy 读取器的标准存储方式，也是下游代码（如势能计算）预期的格式。

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| `id` | int 数组 | 原子 ID（从 1 开始，可选） |
| `mol_id` | int 数组 | 分子 ID（从 1 开始，可选） |
| `number` | int 数组 | 原子序数（可选） |
| `element` | string 数组 | 元素符号（可选） |
| `type` | int 或 string 数组 | 原子类型（可选） |
| `mass` | float 数组 | 原子质量，单位 amu（可选） |
| `charge` | float 数组 | 部分电荷，单位基本电荷（可选） |
| `x` | float 数组 (N) | 原子 x 坐标 |
| `y` | float 数组 (N) | 原子 y 坐标 |
| `z` | float 数组 (N) | 原子 z 坐标 |
| `vx` | float 数组 (N) | 原子 x 方向速度（可选） |
| `vy` | float 数组 (N) | 原子 y 方向速度（可选） |
| `vz` | float 数组 (N) | 原子 z 方向速度（可选） |
| `res_id` | int 数组 | 残基 ID（可选） |
| `res_name` | string 数组 | 残基名称（可选） |

格式特有的别名（如 LAMMPS 的 `q` 和 `mol`）只在 I/O 边界出现。读取器将其规范化（canonicalize）为 `charge` 和 `mol_id`，写入器在目标格式需要时再本地化（localize）回去。

**单位。** `mass` 用 amu，`charge` 用基本电荷。坐标（`x/y/z`）和速度（`vx/vy/vz`）**没有固有单位**——MolPy 只存原始数值。长度约定取决于应用的力场（其 `units=` 设置，如 LAMMPS `real` 对应埃）和读写文件的格式。请确保输入坐标与对应约定一致（例如 TIP3P 的 `tip3p.xml` 用纳米）。

#### 键拓扑（`bonds`）

`bonds` 命名空间用源原子和目标原子的独立索引数组存储键连接关系。这种设计简化了索引操作，也与 Entity 层面的命名一致。所有数组长度为 `E`，即键的总数。

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| `atomi` | int 数组 | 键源原子索引 |
| `atomj` | int 数组 | 键目标原子索引 |
| `type` | string 或 int 数组 | 键类型（可选） |

#### 角拓扑（`angles`）

`angles` 命名空间表示三体相互作用。原子索引分别为 `atomi`、`atomj`、`atomk`，其中 `atomj` 是中心原子。所有数组长度为 `A`，即角的数量。

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| `atomi` | int 数组 | 第一个原子索引 |
| `atomj` | int 数组 | 中心原子索引 |
| `atomk` | int 数组 | 第三个原子索引 |
| `type` | string 或 int 数组 | 角类型（可选） |

#### 二面角拓扑（`dihedrals`）

`dihedrals` 命名空间表示四体扭转相互作用。原子索引分别为 `atomi`、`atomj`、`atomk`、`atoml`，约定与 `angles` 一致。所有数组长度为 `D`，即二面角数量。

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| `atomi` | int 数组 | 第一个原子索引 |
| `atomj` | int 数组 | 第二个原子索引 |
| `atomk` | int 数组 | 第三个原子索引 |
| `atoml` | int 数组 | 第四个原子索引 |
| `type` | string 或 int 数组 | 二面角类型（可选） |

#### 异常二面角拓扑（`impropers`）

`impropers` 命名空间表示异常二面角相互作用，常用于强制平面性或手性约束。索引命名与 `dihedrals` 相同。

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| `atomi` | int 数组 | 第一个原子索引 |
| `atomj` | int 数组 | 第二个原子索引 |
| `atomk` | int 数组 | 第三个原子索引 |
| `atoml` | int 数组 | 第四个原子索引 |
| `type` | string 或 int 数组 | 异常二面角类型（可选） |

### 命名空间命名约定

命名空间名称用小写，通常用复数（`atoms`、`bonds`、`angles`、`dihedrals`、`impropers`），表明它们装的是同类项目的集合。这套约定让命名空间键自带文档含义，在整个代码库中也保持一致。通过字符串键访问命名空间：`frame["atoms"]` 拿到 atoms Block，之后再用字段名索引，如 `frame["atoms"]["x"]`。

命名空间内部的字段名用小写，多词以下划线分隔。拓扑索引字段用 `atomi`、`atomj`、`atomk`、`atoml` 标示角色（源、目标、中心、端点），同时保持无后缀命名，以区别于 Entity 层面的对象引用。

### Entity 层面的拓扑：对象引用

Entity 层面的拓扑对象（Bond、Angle、Dihedral、Improper）直接操作 Atom 实例，在分子构建、编辑和化学推理过程中使用。这一层用 `itom`、`jtom`、`ktom`、`ltom` 作为字段名，对应 Frame 层面的 `atomi`、`atomj`、`atomk`、`atoml`，但存的是对象引用而非整数索引。例如：

```python
bond = mol.def_bond(atom1, atom2)
print(bond.itom)  # Atom 对象
print(bond.jtom)  # Atom 对象
```

这里 `itom` 和 `jtom` 是 Atom 对象的直接引用，不是索引。命名刻意保持简短——这些字段在结构操作中频繁访问——而 `tom` 后缀标示该值是对象而非数字标识符。对应的规则：`itom`、`jtom`、`ktom`、`ltom` 必须存 Atom 引用，不能存整数。

这种并行命名方案保证了每个原子的语义角色（第一个、第二个、中心等）在两种表示中保持一致，而 `atomi` 与 `itom` 这样的命名区分则在每个使用点明确标示出当前用的是哪种表示。

### 为何存在此约定

命名空间和命名约定解决两个问题：数据组织和类型安全。

组织层面，命名空间消除了字段归属和含义的歧义。没有明确的命名空间，字段名只能通过前缀或后缀来编码语义类别，结果就是 `atom_z` 与 `z_atom` 与 `atomic_number` 与 `element` 等五花八门的写法。命名空间把类别显式化，让它与字段身份分离——无论上下文如何，原子序数始终通过 `frame["atoms"]["number"]` 访问。

类型安全层面，MolPy 刻意避免对索引和引用使用相同的字段名。`atom_i` 或 `atom1` 这类叫法在其他库里很常见，但它们模糊了"表中的位置"和"内存中的对象"之间的区别。Frame 层面用 `atomi`、`atomj`，Entity 层面用 `itom`、`jtom`，代码在使用点就能看出区别，也便于通过类型检查或运行时校验来强制执行。

这种分离防止了一类隐蔽错误——字段根据上下文悄然改变含义。这类错误在引入序列化、缓存、多进程或跨语言绑定时尤其麻烦，因为交换格式必须用索引，而进程内 API 用的是对象引用。Frame 层面的设计完全可序列化为 JSON、Arrow 和 HDF5，并且语言无关，在 Rust、C++ 或 TypeScript 中实现等价数据结构都很直接。它也与 LAMMPS 等 MD 引擎期望的数据布局自然对齐——这些引擎操作的是索引原子表，不是基于指针的对象图。

与此同时，Entity 层面的设计支持交互式分子构建所需的流畅 API：图遍历和化学推理需要直接访问原子属性，不能反复查索引。两个层面保持分离，边界处强制显式转换，避免了那种试图同时服务两个目的、最终两头都不合适的"万能"数据结构。

### 如何在表示之间转换

MolPy 提供了两种表示之间的显式转换路径。从 Entity 转到 Frame（如通过 `Atomistic.to_frame()`），每个 `itom`、`jtom` 等被替换为对应的原子索引，结果存入命名空间组织的 Block 中。生成的 frame 在其各命名空间 block 中只使用 `atomi`、`atomj` 等。逻辑如下：

```python
# Entity 到 Frame 的键转换
atomi_list = [atoms.index(bond.itom) for bond in bonds]
atomj_list = [atoms.index(bond.jtom) for bond in bonds]

frame["bonds"] = Block({
    "atomi": atomi_list,
    "atomj": atomj_list,
})
```

从 Frame 转回 Entity 时，每个索引通过 atoms 容器解析回 Atom 对象。Frame 中的命名空间结构指明了要构建哪些 Entity 类型：

```python
# Frame 到 Entity 的键转换
for i in range(len(frame["bonds"]["atomi"])):
    atomi = frame["bonds"]["atomi"][i]
    atomj = frame["bonds"]["atomj"][i]
    bond = Bond(itom=atoms[atomi], jtom=atoms[atomj])
```

这些转换必须在 Frame 层和 Entity 层的边界处显式、局部地进行。不允许在同一对象内混合两种表示。Frame 的命名空间结构确保转换时所有相关字段（如键索引和键类型）保持在一起，逻辑更简单，也减少了错位的可能。

### 给贡献者

扩展 MolPy 时，请把以下规则当作硬性约束：Frame/Block 拓扑始终用 `atomi/atomj/atomk/atoml`（索引）；Entity 拓扑始终用 `itom/jtom/ktom/ltom`（对象引用）；边界处显式转换，绝不在一个对象内混用两者。其设计理由（类型安全、可序列化为 JSON/Arrow/HDF5、支持跨语言后端）在[开发者指南](../developer/index.md)中有详细说明。
