# MCP：让 LLM 智能体读取 MolPy 源码

## MCP 解决的问题

LLM 智能体使用 MolPy 时遇到一个老问题：API 知识会过时。猜测的函数名、参数列表或返回类型看起来可能合理，但往往是错的——代价是在跑不起来的代码上反复调试，浪费大量时间。

MolCrafts MCP 套件把 MolPy 已安装的源码暴露为**结构化、图索引的代码视图**来解决这个问题。智能体不再靠猜，而是直接查询代码图：哪个符号实现了这个功能，签名是什么，在哪里被调用，哪些测试覆盖它，哪些示例用到它。每个答案都来自当前 Python 环境中真正可导入的 MolPy 源码树——而不是模型的训练数据。

实际工作流很简单：

1. 解析功能（或浏览包概览）。
2. 深入相关符号——签名、文档字符串、源码、调用方、测试。
3. 针对真实的 MolPy API 写标准 Python 代码。

!!! note "MCP 不是什么"
    服务器不会替智能体执行 MolPy 工作流。它是一个*发现层*：帮助智能体理解库，然后智能体自己写代码直接调用 MolPy。

## 为什么用代码图

MolPy 不是扁平的函数列表。一个真实的工作流——比如构建聚合物、分配类型、再写入 LAMMPS 文件——需要遍历*关系*：`AmberPolymerBuilder` *包含* `build` 方法，该方法*调用* `tleap`，后者*生成*一个 `Frame`，其 `box` 属性被 `write_lammps_data` 读取。对 `"build"` 做 grep 给不了这些信息，扁平符号表也不行。

MCP 服务器把 MolPy 索引为**类型化的代码图**，让智能体能提出图状问题：

- "哪个类实现了径向分布函数？" → 按功能搜索节点。
- "什么调用了 `Atomistic.get_topo`？" → 反向遍历 `calls` 边。
- "哪些测试覆盖了 `ForceField.merge`？" → 跟随 `tests` 边到 pytest 节点。
- "哪些示例展示了如何使用 `Box.orth`？" → 跟随 `exemplifies` 边到提取的文档字符串示例。
- "如果我修改 `read_xml_forcefield` 会破坏什么？" → 多跳影响分析。

这是 MCP 服务器的核心能力。后面描述的六个工具本质上都是不同形状的图查询；本节其余部分解释图中包含了什么。

### 构建图的流水线

```
MolPy 源码 ─► SourceResolver ─► Snapshot ─► Extractor ─► Resolver ─► GraphStore
  pkg:molpy                       (不可变，   (阶段 1)    (阶段 2)   (SQLite
  或本地                          内容哈希                           graph.db)
  检出                            的)
```

1. **源码解析。** MolPy 源码——无论是已安装的 `pkg:molpy` 还是本地检出——被解析成不可变的 `Snapshot`。快照基于文件的**内容哈希**做键控，而不是分支名；这样缓存的图始终绑定到精确的源码。
2. **提取（阶段 1）。** 用标准库的 `ast` 模块解析每个 Python 文件。分析器输出*节点*——模块、类、函数、方法、属性、字段、常量——包含签名、文档字符串、装饰器和文件/行范围，以及每个调用、基类和导入的*未解析引用*。
3. **解析（阶段 2）。** 解析器把这些引用链接到图中已有的节点：相对导入被解析，pytest 测试被链接到它们覆盖的符号，*文档字符串中的代码块被提升为第一级 `example` 节点*。真正动态的引用（比如 `getattr(obj, name)()`）保持标记为 `unresolved`，以免悄悄污染图。
4. **存储。** 每个快照对应一个 SQLite `graph.db` 文件，写入 `~/.cache/molmcp/discovery/` 下，附带一个派生的 FTS5 索引用于符号搜索。

### 节点和边

所有分析器输出相同模式，所以在一个源上工作的查询可以在任何源上工作。

**图中包含的节点类型：**

| 类型 | 表示的内容 |
| --- | --- |
| `package`、`module`、`file` | 作为可导入单元的包树。 |
| `class`、`function`、`method`、`property`、`field`、`constant` | MolPy 公共/私有 API 表面。每个携带 `qualname`、`signature`、`docstring`、文件/行范围。 |
| `example` | 从文档字符串中提取的代码块——随源码一起提供的*真实*用法。 |
| `test` | 一个 pytest 测试函数，带有其所在的测试文件/行。 |
| `capability` | 领域能力（例如"计算 RDF"），通过叠加层附加到一个或多个符号。 |

**边**携带你想要查询的关系：

| 边 | 用于回答 |
| --- | --- |
| `contains` | "`molpy.builder.polymer` 中有哪些符号？" |
| `calls` | "`AmberPolymerBuilder.build` 调用了什么？"——以及反向："谁调用了 `tleap`？" |
| `extends` | "哪些类继承了 `Struct`？" |
| `imports` | "哪些模块从 `molpy.core.atomistic` 导入了？" |
| `exemplifies` | "给我看一个 `Box.orth` 的使用示例。" |
| `tests` | "哪些 pytest 测试覆盖了 `ForceField.merge`？" |
| `references` | "哪里提到了 `EPS_LI`？" |
| `provides_capability` | "哪个符号实现了*径向分布函数*？" |

每条边携带一个 `provenance`（`ast` / `heuristic` / `resolved`），智能体和你都能区分可靠的结构性事实与推断。

### 快照、新鲜度、增量重索引

这个设计带来几个实用的特性：

- **本地源始终是最新的。** 每次查询时重新解析；只有每个文件的提取器有缓存。
- **增量重索引。** 内容寻址的 `ExtractCache` 让未更改的文件跳过分析器，所以编辑一个 MolPy 文件不会重新解析整个包。
- **每个工具响应都携带 `snapshot`**，包含一个 `freshness` 标志，智能体始终知道它正在看哪个 MolPy 版本。
- **图就是纯 SQLite。** 在任何 SQLite 浏览器中打开 `~/.cache/molmcp/discovery/snapshots/<slug>/graph.db`，可以直接检查 `nodes`、`edges` 和 `files` 表。

## 六个图查询工具

MCP 服务器提供六个可组合、只读的工具（全部标记 `readOnlyHint=True`，MCP 客户端可以安全地自动批准它们）。它们在执行图查询的*形状*上有所不同。

| 工具 | 图查询 | 用途 |
| --- | --- | --- |
| `molmcp_outline` | 从包根遍历 `contains` 边。 | 在 MolPy 中定位——"我应该看哪里？" |
| `molmcp_find_capability` | 按能力 + FTS + 结构信号对符号排序。 | 主要工具——描述一个任务，获得排名的符号匹配，包含签名、摘要、示例、测试和调用方。 |
| `molmcp_search_symbols` | 在名称、限定名和摘要上进行 FTS5 搜索，可选 `kind` 过滤器。 | 已经知道名称时的快速查找。 |
| `molmcp_describe_symbol` | 读取一个节点，可选择包含完整源码。 | 最后一步的详细信息：签名、清理后的文档字符串、文件/行范围、源码。 |
| `molmcp_relations` | 从一个符号遍历一种边类型（`callers`、`callees`、`implementers`、`subclasses`、`implementations`、`references`、`examples`、`tests`、`impact` 1–4 跳）。 | 扁平索引无法回答的问题。 |
| `molmcp_refresh` | 强制重新生成一个源的快照。 | 很少使用；本地源会自动刷新。 |

!!! tip "限定名规则"
    始终从之前的工具结果中获取限定名（`molpy.compute.rdf.RDF`，……）——从 `molmcp_outline`、`molmcp_search_symbols` 或 `molmcp_find_capability` 中获取——切勿猜测。错误的限定名会返回结构化的 `{"error": …}` 而非幻觉式的载荷，这是设计如此。

### 典型的遍历模式

这些工具可以组合使用。智能体在 MolPy 上反复使用的一些模式：

**"这个任务该用哪个类？"**

```text
molmcp_find_capability("compute a radial distribution function")
  → matches: [molpy.compute.rdf.RDF, …]
molmcp_describe_symbol("molpy.compute.rdf.RDF")
  → signature + docstring
```

**"这个实际上怎么用？"**

```text
molmcp_relations("molpy.core.Box.orth", relation="examples")
  → docstring examples + tests that exercise it
```

**"我的修改会破坏什么？"**

```text
molmcp_relations("molpy.io.read_xml_forcefield", relation="callers", depth=2)
  → every site, two hops out, that depends on the function
```

**"展示子包的结构。"**

```text
molmcp_outline(path="molpy/builder/polymer")
  → modules + classes + functions, scoped to one subtree
```

这些不是搜索结果——它们是图遍历。这就是把 MolPy 索引成图而不是列表的意义所在。

## 安装和注册服务器

从 PyPI 安装 `molmcp`，锁定到 0.2 版本线：

```bash
pip install "molcrafts-molmcp>=0.2,<0.3"
```

需要 Python ≥ 3.12。PyPI 发行版是 `molcrafts-molmcp`；导入名称和 CLI 入口都是 `molmcp`。发现引擎不添加任何必需的运行时依赖——它用标准库就够了。

在 stdio 模式下启动服务器（MCP 客户端的预期方式）：

```bash
python -m molmcp --source pkg:molpy
```

### Claude Code

项目级注册会在仓库根目录写入 `.mcp.json`：

```bash
claude mcp add molpy --scope project -- python -m molmcp --source pkg:molpy
```

省略 `--scope project` 可以在用户级别注册。

!!! tip "虚拟环境"
    如果 `python` 解析到了错误的解释器（或者 `molmcp` 位于项目本地的 venv 中），请显式注册二进制文件：

    ```bash
    claude mcp add molpy -- /path/to/venv/bin/python -m molmcp --source pkg:molpy
    ```

    或者让 `uv` 从项目目录中选择正确的环境：

    ```bash
    claude mcp add molpy -- uv run --directory /path/to/molpy python -m molmcp --source pkg:molpy
    ```

    启动一个新的 Claude Code 会话，然后运行 `/mcp`。应该能看到服务器及其工具名称，前缀为 `mcp__molpy__molmcp_…`。

### Claude Desktop

编辑 Claude Desktop 的 `mcpServers` 块：

```json
{
  "mcpServers": {
    "molpy": {
      "command": "python",
      "args": ["-m", "molmcp", "--source", "pkg:molpy"]
    }
  }
}
```

配置文件位置：

- **macOS**：`~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**：`%APPDATA%\Claude\claude_desktop_config.json`

保存后重启 Claude Desktop。MolPy 的发现工具应该会出现在工具选择器中。

### 索引本地检出

把 `--source` 指向 MolPy 检出目录的工作树，暴露你正在做的编辑——图会在每次查询时增量重建，所以你在编辑器中做的修改，一个工具调用后就能对智能体可见：

```bash
python -m molmcp --source /path/to/molpy
```

### 验证服务器是否正常工作

`molmcp` 附带了内置自检功能，可以索引 MolPy 并打印计数、FTS 状态以及一个示例查询——失败时以非零退出，所以它也可以用作 CI/设置检查：

```bash
molmcp discovery verify pkg:molpy
```

无需通过 MCP 客户端即可从 CLI 检查或重建图：

```bash
molmcp discovery index pkg:molpy        # 构建图
molmcp discovery outline pkg:molpy      # 高级概览
molmcp discovery query pkg:molpy "radial distribution function"
```

## 编写有效的提示词

MCP 让智能体能够访问 MolPy 的 API，但这并不能替代清晰的任务描述。最好的提示词会指定系统和约束，然后让智能体自己去发现实现方式。

### 描述系统，而非 API

告诉智能体你要构建什么，不要告诉它该调用哪个函数。

| 过于底层 | 更好的方式 |
| --- | --- |
| `使用 polymer() 构建 PEG 链` | `构建一条含有 15 个重复单元的 PEG 链` |
| `调用 Packmol 来填充分子` | `将 15 条链填充到 20 nm 的立方盒子中` |
| `使用 Box 类` | `为系统创建一个周期性模拟盒子` |

如果提示词命名了特定的 MolPy 函数，通常说明过于底层了。`molmcp_find_capability` 的意义就在于智能体把*任务*映射到正确的符号——提前给它符号就绕过了流水线中最强大的一环。

### 包含物理参数

分子工作流是由数值定义的。省略了它们，智能体就不得不自己猜。

至少需要指定：

- 分子类型
- 链长或组成
- 分子数量
- 盒子大小或密度
- 输出格式（如果重要的话）

例如：

```text
生成一个 20 x 20 x 20 nm 的盒子，包含 15 条 PEG 链，
每条链有 15 个重复单元。导出为 LAMMPS DATA 格式。
```

### 保持每个提示词对应一个工作流

不要要求智能体在一个步骤中构建系统、运行 MD 和分析结果。

把工作分解成多个阶段：

1. 构建和填充系统。
2. 准备模拟输入。
3. 运行分析。

这和真实建模工作流被调试和验证的方式一致。

### 说明约束，然后让智能体探索

指出会影响结果的约束条件，例如：

- `使用 Amber 后端和 GAFF2 参数`
- `使用 OPLS-AA 类型分配`

之后，让智能体通过 MCP 检查 API。避免：

- 强制指定特定的函数名
- 粘贴记忆中的代码片段
- 过度指定实现细节

如果智能体探索后仍然失败，这通常是对 API 或文档有用的反馈。

### 快速检查清单

发送提示词之前，检查：

- 它描述的是分子系统而不是 API 调用？
- 重要的数值是否都存在？
- 这是否是一个工作流而不是多个？
- 是否说明了实质影响结果的约束条件？

## 完整示例：TIP3P 水盒子

这个第一个示例特意保持很小。它只使用 MolPy 内置的数据，因此是确认 MCP 已连接且智能体正在读取本地 API 而非凭记忆猜测的一个很好的健全性检查。

### 提示词

```text
使用 MolPy 为 LAMMPS 构建一个小的周期性 TIP3P 水盒子。
创建 64 个水分子，排列在 4 x 4 x 4 的网格中，间距 0.32 nm。
使用 MolPy 内置的 tip3p.xml 力场，分配原子、键和角类型，
并将结果写入 quickstart-output/water_box_tip3p.data
和 quickstart-output/water_box_tip3p.ff。
```

### 智能体探索

Claude 首先向图查询任务所需的能力：力场加载、类型分配、拓扑生成和 LAMMPS 导出。

**步骤 1 — 按任务解析能力**

```
molmcp_find_capability("read an XML force field and write LAMMPS data + ff")
```

```
matches:
  • molpy.io.read_xml_forcefield        (function)
  • molpy.io.write_lammps_data          (function)
  • molpy.io.write_lammps_forcefield    (function)
  • molpy.typifier.OplsTypifier (class)
```

**步骤 2 — 确认内置 TIP3P 文件路径**

```
molmcp_describe_symbol("molpy.io.read_xml_forcefield")
```

```
signature: (filepath: str | Path, forcefield: AtomisticForcefield | None = None)
           -> AtomisticForcefield
docstring: Read an XML force field file. Relative filenames resolve against the
           package data directory, so passing "tip3p.xml" loads the built-in
           TIP3P force field.
(签名: (filepath: str | Path, forcefield: AtomisticForcefield | None = None)
       -> AtomisticForcefield
文档字符串: 读取一个 XML 力场文件。相对文件名会相对于包的数据目录解析，
           因此传入 "tip3p.xml" 会加载内置的 TIP3P 力场。)
```

**步骤 3 — 验证不明显的拓扑步骤**

```
molmcp_describe_symbol("molpy.core.atomistic.Atomistic.get_topo")
```

文档字符串阐明了一个微妙之处：当 `gen_angle=True` 时，`get_topo()` 返回一个*新的* `Atomistic`，而不是修改现有的。所以 Claude 写的是 `system = system.get_topo(...)` 而不是原地调用。

**步骤 4 — 检查盒子和类型分配 API**

```
molmcp_describe_symbol("molpy.core.Box.orth")
```

```
(lengths: ArrayLike, pbc: ArrayLike = ..., origin: ArrayLike = ...,
 central: bool = False) -> Box
```

```
molmcp_describe_symbol("molpy.typifier.OplsTypifier.__init__")
```

```
(forcefield: ForceField, skip_atom_typing: bool = False,
 skip_pair_typing: bool = False, skip_bond_typing: bool = False,
 skip_angle_typing: bool = False, skip_dihedral_typing: bool = False,
 strict_typing: bool = True)
```

有了这些信息，Claude 编写了下面的脚本。

### 生成的脚本

```python
from pathlib import Path

import numpy as np
import molpy as mp
from molpy.io import read_xml_forcefield, write_lammps_data, write_lammps_forcefield
from molpy.typifier import OplsTypifier

theta = 1.82421813418
r_oh = 0.09572  # nm

water = mp.Atomistic(name="water_tip3p")
o = water.def_atom(element="O", name="O", x=0.0, y=0.0, z=0.0, charge=-0.834)
h1 = water.def_atom(element="H", name="H1", x=r_oh, y=0.0, z=0.0, charge=0.417)
h2 = water.def_atom(
    element="H",
    name="H2",
    x=r_oh * float(np.cos(theta)),
    y=r_oh * float(np.sin(theta)),
    z=0.0,
    charge=0.417,
)
water.def_bond(o, h1, order=1)
water.def_bond(o, h2, order=1)

system = mp.Atomistic(name="water_box_tip3p")
mol_id = 1
for iz in range(4):
    for iy in range(4):
        for ix in range(4):
            mol = water.copy()
            mol.move(delta=[ix * 0.32, iy * 0.32, iz * 0.32])
            for atom in mol.atoms:
                atom["mol_id"] = mol_id
            system.merge(mol)
            mol_id += 1

system = system.get_topo(gen_angle=True, gen_dihe=False)

ff = read_xml_forcefield("tip3p.xml")
typifier = OplsTypifier(ff, skip_dihedral_typing=True, strict_typing=True)
system = typifier.typify(system)

frame = system.to_frame()
frame.box = mp.Box.orth([1.28, 1.28, 1.28])
frame["atoms"]["id"] = np.arange(1, frame["atoms"].nrows + 1, dtype=int)
frame["atoms"]["mol_id"] = np.asarray(frame["atoms"]["mol_id"], dtype=int)
frame["atoms"]["charge"] = np.asarray(frame["atoms"]["charge"], dtype=float)

out_dir = Path("quickstart-output")
out_dir.mkdir(parents=True, exist_ok=True)
write_lammps_data(out_dir / "water_box_tip3p.data", frame, atom_style="full")
write_lammps_forcefield(out_dir / "water_box_tip3p.ff", ff)
```

### 输出

```
atoms 192
bonds 128
angles 64
files ['water_box_tip3p.data', 'water_box_tip3p.ff']
```

这个示例完全在本地运行：不需要 AmberTools，不需要 Packmol，也不需要查阅文献。它通常是确认 MCP 客户端能检查 MolPy、合成正确脚本并导出真实模拟输入的最快方式。

## 完整示例：多分散 PEO/LiTFSI 电解质

下一个示例要大得多。MCP 服务器仍然在做相同的工作，但智能体需要更深入地遍历图——查找构建器、填充类以及确定 Li⁺ 参数的测试——然后才能编写完整的工作流。

### 提示词

```text
使用 MolPy 生成原子级别的 PEO/LiTFSI 聚合物电解质体系，满足以下严格约束条件。
使用 Schulz-Zimm 分布构建多分散 PEO 链，目标数均聚合度 (DP_n = 20)，
多分散指数 (PDI = 1.20)。精确构建 40 条 PEO 链。使用 AmberTools 为 PEO 单体
和聚合物链生成力场参数和连接性信息，采用 GAFF 力场并正确处理连接和端基。
以 EO:Li = 20:1 的固定组成比添加 LiTFSI 盐，并根据采样聚合物系综中的 EO
重复单元总数精确计算 LiTFSI 分子数。查阅文献获取 Li⁺ 非键参数。
使用 Packmol 以 0.10 g/cm³ 的极低初始密度进行填充。工作流应完全端到端：
定义 PEO 重复单元和 LiTFSI，从 Schulz-Zimm 分布中采样链长，构建所有 PEO 链，
使用 AmberTools 分配参数，添加 LiTFSI，填充整个体系，并导出坐标和力场文件
供后续分子动力学使用。
```

### 智能体探索

Claude 先在包中定位，然后深入到它需要的符号。

**步骤 1 — 在包中定位**

```
molmcp_outline()
```

返回 MolPy 的顶层包和模块（节选）：

```
molpy.builder   Crystal and polymer builders (AmberTools integration, stochastic generation)
molpy.io        I/O for AMBER, LAMMPS, PDB, GRO, MOL2, XYZ ...
molpy.pack      Packing (constraints, targets, Packmol integration)
molpy.parser    Parsers for SMILES, BigSMILES, CGSmiles, GBigSMILES
molpy.wrapper   External tool wrappers (antechamber, parmchk2, prepgen, tleap)
```

**步骤 2 — 找到分布和聚合物构建器类**

```
molmcp_outline(path="molpy/builder/polymer")
```

```
SchulzZimmPolydisperse    Schulz-Zimm molecular weight distribution for polydisperse polymer chains
UniformPolydisperse       Uniform distribution over degree of polymerization
PoissonPolydisperse       Poisson distribution for degree of polymerization
FlorySchulzPolydisperse   Flory-Schulz (geometric) distribution
PolydisperseChainGenerator  Middle layer: samples DP/mass, generates monomer sequences
SystemPlanner             Top layer: accumulates chains until a target total mass is reached
AmberPolymerBuilder       Polymer builder backed by the AmberTools pipeline
PolymerBuilder            CGSmiles-based polymer builder with pluggable typifier
```

**步骤 3 — 读取 Schulz-Zimm 的签名和文档字符串**

```
molmcp_describe_symbol(
    "molpy.builder.polymer.distributions.SchulzZimmPolydisperse",
    include_source=False,
)
```

```
signature: (Mn: float, Mw: float, random_seed: int | None = None)
docstring:
  Schulz-Zimm molecular weight distribution for polydisperse polymer chains.
  Implements MassDistribution — sampling is done directly in molecular-weight space.

  The PDF is:
      f(M) = z^(z+1)/Γ(z+1) · M^(z−1)/Mn^z · exp(−zM/Mn)
  where z = Mn/(Mw − Mn).  Equivalent to Gamma(shape=z, scale=Mw−Mn).

  Args:
      Mn: Number-average molecular weight (g/mol).
      Mw: Weight-average molecular weight (g/mol), must satisfy Mw > Mn.

  Methods:
      sample_mass(rng) → float     draw one mass sample
      mass_pdf(mass_array) → ndarray
```

Claude 注意到：z = 1/(PDI − 1) = 5.0，对于 PDI = 1.20。

**步骤 4 — 按能力定位 `AmberPolymerBuilder`，然后深入**

```
molmcp_find_capability("build a polymer with AmberTools / GAFF parameters")
```

挑出了 `molpy.builder.polymer.ambertools.AmberPolymerBuilder`。Claude 然后读取其构造函数和 `build` 方法：

```
molmcp_describe_symbol("molpy.builder.polymer.ambertools.AmberPolymerBuilder")
```

```
signature: (library: dict[str, Atomistic],
            force_field: str = "gaff2",
            charge_method: str = "bcc",
            work_dir: Path = Path("amber_work"),
            env: str = "AmberTools25",
            env_manager: str = "conda")
```

```
molmcp_describe_symbol("molpy.builder.polymer.ambertools.AmberPolymerBuilder.build")
```

```
docstring:
  Build a polymer from a CGSmiles string.

  Args:
      cgsmiles: CGSmiles notation, e.g. "{[#MeH][#EO]|10[#MeT]}"
                |N means N repeat units of the preceding monomer.

  Returns:
      AmberBuildResult with .frame (Frame) and .forcefield (ForceField).

  Pipeline (automatic):
      antechamber  → GAFF atom types + BCC charges (mol2 + ac files)
      parmchk2     → missing torsion/vdW parameters (frcmod)
      prepgen      → HEAD / CHAIN / TAIL residue variants (prepi)
      tleap        → build polymer and generate prmtop / inpcrd
```

**步骤 5 — 通过 `tests` 边确定 Li⁺ 参数**

Claude 不会猜测 Li⁺ 的 LJ 参数。它不使用 `grep`，而是遍历图：任何对"Åqvist"的引用都是一个节点，而它周围的 `test` 节点精确地告诉 Claude MolPy 认为哪些参数值是规范的：

```
molmcp_search_symbols("Åqvist")
```

```
test_e2e_peo_litfsi.test_li_frcmod   (test, tests/test_e2e_peo_litfsi.py:147)
    "Write Åqvist (1990) Li+ frcmod and build prmtop via tleap."
```

```
molmcp_describe_symbol(
    "test_e2e_peo_litfsi.test_li_frcmod", include_source=True
)
```

揭示了规范的参考文献：**Åqvist (1990), J. Phys. Chem. 94, 8021** — Rmin/2 = 1.137 Å, ε = 0.0183 kcal/mol。Claude 将这些值直接写入一个 frcmod 文件。

**步骤 6 — 找到填充和导出接口**

```
molmcp_outline(path="molpy/pack")
```

```
Packmol                  High-level Packmol packing interface
InsideBoxConstraint      Place molecules inside a rectangular box
OutsideBoxConstraint     Keep molecules outside a box
InsideSphereConstraint   Sphere constraint
MinDistanceConstraint    Minimum pairwise distance
Target                   One packing target (frame + count + constraint)
```

```
molmcp_describe_symbol("molpy.pack.Packmol.pack")
```

```
(max_steps: int = 20000, seed: int = 12345) → Frame
```

```
molmcp_describe_symbol("molpy.io.write_lammps_forcefield")
```

```
(path: Path | str,
 forcefield: ForceField,
 precision: int = 6,
 skip_pair_style: bool = False) → None
```

Claude 注意到：需要 `skip_pair_style=True`，以便 LAMMPS 输入脚本可以独立控制 `kspace_style`。

**步骤 7 — 通过 `examples` 边确认 `ForceField.merge`**

```
molmcp_relations(
    "molpy.core.forcefield.ForceField.merge", relation="examples"
)
```

返回文档字符串示例以及在真实力场上测试该方法的测试，确认了契约：

```
docstring:
  Merge two ForceField objects.  Returns a new ForceField containing all
  styles and parameters from both.  Raises if incompatible styles are found.
(文档字符串:
  合并两个 ForceField 对象。返回一个新的 ForceField，
  包含两者的所有样型和参数。如果发现不兼容的样型则抛出异常。)
```

**步骤 8 — 在提交之前进行影响健全性检查**

在编写脚本之前，Claude 做一次最终的图查询——从它计划调用的导出函数开始的多跳 `impact` 遍历——以确保它理解即将串联起来的依赖关系：

```
molmcp_relations(
    "molpy.io.write_lammps_data", relation="impact", depth=2
)
```

确认 `write_lammps_data` 最终读取了 `frame.box` 以及每个原子的 `charge`/`mol_id` 列——这就是生成的脚本在导出之前显式填充这些列的原因。

有了这些信息，Claude 就拥有了组装脚本所需的一切。

!!! note "完整的生成脚本"
    完整的端到端脚本位于 `docs/user-guide/08_peo_litfsi_electrolyte.py`。它运行 antechamber/parmchk2/tleap 处理 TFSI⁻，根据 Åqvist 参数构建 Li⁺ frcmod，从 Schulz-Zimm 分布中采样 40 条链长，为每个独特 DP 调用 `AmberPolymerBuilder`，合并三个力场，以 0.10 g/cm³ 用 Packmol 填充，并将 LAMMPS 数据文件和 `system.ff` 导出到 `peo_litfsi_output/lammps/`。运行它需要 AmberTools 和一个名为 `AmberTools25` 的 conda 环境中的 Packmol。

## 参见

- [多分散体系](../user-guide/05_polydisperse_systems.md) — 从分布到 LAMMPS 导出的端到端工作流
- [发现引擎参考](https://github.com/MolCrafts/molmcp) — 完整的代码图模式、快照/缓存机制和 CLI，供好奇者查阅
