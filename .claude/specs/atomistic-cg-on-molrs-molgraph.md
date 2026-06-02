# Spec: 用 molrs `MolGraph` 直接背书 `Atomistic` 与 `CoarseGrain`

| 字段   | 值                                                       |
| ------ | -------------------------------------------------------- |
| Status | Draft                                                    |
| Target | molpy 0.3.x（pre-1.0，允许 BREAKING）                    |
| Owner  | core（atomistic / cg / entity / _molrs）+ molrs-python   |
| Layers | core（单一存储改写）+ molrs-python（绑定扩面）+ molrs-core（impropers + 拓扑感知） |
| Risk   | 高：core 身份模型从「对象身份」改为「稳定 ID」，全量测试重写 |
| Date   | 2026-06-01                                               |

---

## 1. Overview / Motivation

molpy 现在维护**两套**分子图表示，靠 `core/_molrs.py` 的 `to_molrs` / `from_molrs`
在它们之间来回拷贝：

| 表示 | 位置 | 存储 | 身份 |
|---|---|---|---|
| molpy `Atomistic` | `core/atomistic.py` + `core/entity.py` | `Entity(UserDict)` 原子列表 + `Link(UserDict)` 键/角/二面角，全是 Python 对象 | **对象身份**（`bond.itom is atom`，`id(atom)`） |
| molrs `MolGraph` | `molrs-core/src/molgraph.rs`（Rust） | `SlotMap<AtomId>` / `BondId` / `AngleId` / `DihedralId`，每个挂一个 `props: HashMap<String, PropValue>` | **稳定 ID** |

`molrs-core` 的 `MolGraph` 文档明说它存「atoms (or beads), bonds, angles, and
dihedrals」，每个实体都是泛型属性袋 —— 这与 molpy `Entity`/`Link` 的「UserDict =
属性袋」**概念完全同构**。换句话说，molpy 在 Python 里重新实现了一遍 molrs 已经在
Rust 里实现的东西，再用一层会丢信息的转换胶水把两者缝起来。

这层胶水是 bug 的稳定来源（均为近期实测）：

- `from_molrs` 在 `add_hydrogens` 改变原子数时 `use_template=False`，**丢弃所有自定义
  属性**（`port` 等），逼下游在 embed 之后手动重设端口。
- `to_molrs` 一直不传 formal charge，导致带电单体 `[N+]`/`[N-]` 加错氢（已修，但属于
  「胶水必须逐字段手搬」的典型症状）。
- embed 走 `to_molrs → molrs.generate_3d → from_molrs(template=...)` 的往返，几何、
  属性、身份在三次拷贝里各丢一点。

**关键事实：molrs-core 已经实现了「通用图作基、AA/CG 作子类」这套层级**，molpy 没有把它
暴露出来而已：

| molrs-core | 角色 | 不变量 |
|---|---|---|
| `molgraph.rs::MolGraph` | **通用图基类**：节点（atoms **or** beads）+ bonds/angles/dihedrals，全为属性袋 | 无 |
| `atomistic.rs::Atomistic(MolGraph)` | 全原子**特化**（newtype + `Deref/DerefMut`，继承全部图方法） | 每节点有 `element` |
| `coarsegrain.rs::CoarseGrain(MolGraph)` | 粗粒化**特化**（同上，`add_bead`） | 每节点有 `bead_type` |

Rust 用 newtype + `Deref` 表达「is-a 通用图 + 加一条领域不变量」，即继承的等价物。
但 `molrs-python` **只导出了 `Atomistic`**（`#[pyclass(name="Atomistic")]`），既没导出
通用基 `MolGraph`，也没导出 `CoarseGrain` —— 所以 Python 看不到这个层级，molpy 只好在
Python 里另起炉灶重写一遍。

**指导原则**：

> 不要发明新层级，而是**把 molrs-core 已有的 `MolGraph ← Atomistic / CoarseGrain`
> 层级如实暴露到 Python，并让 molpy 的 `Atomistic`/`CoarseGrain` 成为它的子类**。
> 通用图是唯一事实源与公共基；AA / CG 是只加领域约定的特化；`Atom` / `Bead` 是通用
> 节点句柄 `Entity` 的子类。

判据沿用 [`cg-atomistic-mapping-redesign.md`](./cg-atomistic-mapping-redesign.md) 的
「core 方法 = 在已锁定约定上、没有第二种合理实现的操作」。本 spec 不改判据，只把
**存储后端 + 类层级**统一到 molrs 图：`Atomistic`/`CoarseGrain` 的对称性不再靠人工对齐
两套 Python 代码，而是「共同继承同一通用图」的结构性保证，与那份 spec 的 G3 合流。

## 2. Goals & Non-Goals

### Goals

- **G1（层级）**：建立单一类层级，根是 molrs 通用图：
  `Graph(通用) ← Atomistic / CoarseGrain`，`Entity(通用节点) ← Atom / Bead`。
  molpy 的 `Atomistic`/`CoarseGrain`/`Atom`/`Bead` 是这些通用基的**子类**，而非平行实现。
  唯一存储是 molrs `MolGraph`；不再有 Python 端 `Entities`/`links` 平行列表。
- **G1b（暴露通用基）**：`molrs-python` 导出通用基 `Graph`（= `MolGraph`）与
  `CoarseGrain`，与 `Atomistic` 并列，且让三者带上 PyO3 `subclass`/`extends` 关系
  （见 §5、§7-D6），使 Python 侧能看到并继承这条层级。
- **G2**：`Entity`（`Atom`/`Bead`）与 `Link`（`Bond`/`Angle`/`Dihedral`/`Improper`）
  是**轻量句柄**：内部仅 `(graph, id)`，`__getitem__`/`__setitem__`/`get`/`keys`
  代理到 molrs 实体的属性袋。`Atom`/`Bead` 仅在通用 `Entity` 上加领域访问器
  （`Atom.element`、`Bead.bead_type`）。公开的 `UserDict` 风格 API 表面**不变**。
- **G3**：`to_frame` 退化为对 molrs `MolGraph::to_frame` 的直通；删除 Python 侧逐字段
  重建。
- **G4**：删除 `core/_molrs.py` 的 `to_molrs` / `from_molrs`；embed / 几何 / 旋转等
  compute 算子**原地**作用于同一张图，无往返拷贝。
- **G5**：`CoarseGrain` 复用同一 `MolGraph` 后端（bead = 带 bead 属性的 atom 实体），
  与 `Atomistic` 逐成员对称，落实 cg-redesign spec 的 G3。
- **G6**：molrs-python 绑定补齐到「足以背书一个全功能 `Struct`」的最小面（见 §5）。
- **G7**：BREAKING 一次到位，不保留 deprecated 双后端。

### Non-Goals

- **NG1**：不改 `Atom`/`Bond`/... 的**公开**字典语义（`atom["x"]`、`bond.itom`、
  `struct.atoms["x"]` 列访问仍可用）。本 spec 换的是后端，不是表面 API。
- **NG2**：不在 core 引入任何派生量算子（COM/COG/质量求和/投影），延续 cg-redesign
  的 NG2。
- **NG3**：不改 `ForceField` —— 力场类型仍是独立对象；`MolGraph` 只存拓扑 + 每实体属性。
- **NG4**：不改 `Frame` 的列式契约（仍是 batch 层）；只是 `Atomistic→Frame` 改走 Rust。
- **NG5**：不在本 spec 内重写 parser/io/builder 的业务逻辑；只随身份模型变化做机械迁移
  （§6）。
- **NG6**：不引入「按需双向同步」的混合后端（那正是要删掉的东西）。

## 3. 现状分析（被替换的部分）

```
core/entity.py
  Entity(UserDict)            # 原子/bead：属性袋
  Link[T](UserDict)           # 键/角/二面角：持端点 Entity 引用 + 属性
  Entities[E](list[E])        # 列可访问容器：entities["x"] 返回某列
  Struct                      # self.entities[type] / self.links[type] 两个按类型分桶
  SpatialMixin/Membership/Connectivity

core/atomistic.py
  Atom(Entity) / Bond/Angle/Dihedral/Improper(Link)
  Atomistic(Struct + 3 Mixin)
    .atoms -> Entities[Atom]; .bonds/.angles/.dihedrals -> links[...]
    def_atom/def_bond/def_atoms/def_bonds
    get_topo(gen_angle, gen_dihe)   # 从键枚举角/二面角，Python 实现
    to_frame()                      # 逐字段把对象拍平成 Frame 列
    __iadd__                        # 合并两个 Atomistic（对象拷贝 + 重新建键）

core/cg.py                          # 当前 CG 容器（见 cg-redesign spec）
core/_molrs.py
  to_molrs(mol)/from_molrs(mol_rs)  # ⟵ 本 spec 删除
```

身份模型：`Bond` 内部存 `itom`/`jtom` **Atom 对象引用**；`AmberPolymerBuilder`、
parser、`__iadd__` 等多处用 `is` / `id(atom)` 做去重与匹配。这是迁移的主要冲击面。

molrs 侧已具备的完整能力（`molrs-core/src/molgraph.rs`，仅列与本 spec 相关者）：

```
add_atom/remove_atom/get_atom/get_atom_mut
add_bond/remove_bond/get_bond/get_bond_mut
add_angle/remove_angle/get_angle
add_dihedral/remove_dihedral/get_dihedral
atoms()/bonds()/angles()/dihedrals()  -> 迭代 (Id, &实体)
n_atoms/n_bonds/n_angles/n_dihedrals
neighbors(id)/neighbor_bonds(id)
rotate(axis, angle, about)
Atom/Bond/...: set/get/get_f64/get_str/get_int/remove/keys（属性袋）
```

**缺口**：molrs-core **没有 improper**（molpy 有 `Improper`）；molrs-python 绑定只暴露
`add_atom/set_atom_prop/add_bond/set_bond_order/n_atoms/n_bonds/to_frame`，
远不足以背书 `Struct`。

## 4. 目标架构（类层级）

```
 容器轴（domain）                          节点轴（domain）           拓扑轴（arity）
 ───────────────                          ───────────────           ───────────────
 Graph (通用图)                            Entity (通用节点句柄)      Link (通用连接句柄)
   ├── Atomistic   不变量: 节点有 element     ├── Atom  (+.element…)     ├── Bond     (2 端点)
   └── CoarseGrain 不变量: 节点有 bead_type   └── Bead  (+.bead_type)    ├── Angle    (3)
                                                                        ├── Dihedral (4)
   （molpy 侧 = molrs 同名类的子类 / 包装）                              └── Improper (4)

                ┌──────────────────────── Python (molpy.core) ─────────────────────────┐
 molpy.Graph(基) ← molpy.Atomistic / molpy.CoarseGrain  ← 加 SpatialMixin 等 + 领域方法
 molpy.Entity(基) ← molpy.Atom / molpy.Bead             （句柄；属性代理到图）
                └─────────────────────────────────┬──────────────────────────────────┘
                                                   │ PyO3：molrs.Graph(基) ◁ Atomistic/CoarseGrain
                ┌──────────────────────────────────▼─────────────────────────────────┐
 molrs-core   MolGraph(通用) ◁ Atomistic(MolGraph) / CoarseGrain(MolGraph)  ← 已存在
              唯一事实源：SlotMap<AtomId/BondId/AngleId/DihedralId(/ImproperId)>
              每实体一属性袋；to_frame()/rotate()/neighbors() 原生
                └────────────────────────────────────────────────────────────────────┘
```

三条轴正交：**容器轴**（AA vs CG，对应 `MolGraph` 的两个 newtype 特化）、**节点轴**
（`Atom` vs `Bead`，通用 `Entity` 的两个子类）、**拓扑轴**（Bond/Angle/Dihedral/Improper
按端点数特化通用 `Link`，AA 与 CG **共享**这一轴）。CG 的「bead = 一组原子」是
`bead["atoms"]` 约定（cg-redesign spec G4），不进 molrs 不变量。

### 4.1 句柄语义（Entity / Link）

- `Entity` 持 `(graph, atom_id)`；`__getitem__(k)`→`graph` 的属性读，`__setitem__`→属性写，
  `get/keys/__contains__/__delitem__` 同理。**不再持有自己的 dict**。
- `Link` 持 `(graph, link_id)` + 实体种类标签；端点访问（`bond.itom`、`angle.atoms`）由
  molrs 实体里存的 `AtomId` 解析回 `Entity` 句柄。
- **身份**：句柄实现 `__eq__`/`__hash__` = `(id(graph), kind, slot_id)`。为保住现有
  `is` 语义，`Struct` 维护 `WeakValueDictionary[id] -> handle`，**每个 id 只发一个句柄
  实例**，于是 `bond.itom is atom` 继续成立（§7 决策 D1）。

### 4.2 容器视图（Entities）

- `struct.atoms` 返回一个**视图对象**（不是 `list` 拷贝）：`__iter__` 走 `graph.atoms()`，
  `__len__`=`n_atoms`，`__getitem__(int)` 发句柄。
- **列访问** `struct.atoms["x"]`：新增 molrs 绑定 `atom_column(key) -> list`（或
  `to_frame()` 后取列），返回该属性的整列。写列 `atoms["type"] = arr` 走批量
  `set_atom_prop`（或新增批量 setter）。

### 4.3 关键方法重映射

| molpy 现有 | molgraph 背书后 |
|---|---|
| `def_atom(**a)` | `id = graph.add_atom(sym)`; 逐属性 `set_atom_prop`; 返回 `Atom(graph,id)` |
| `def_bond(a,b,**a)` | `graph.add_bond(a.id,b.id)` + props；返回 `Bond` 句柄 |
| `get_topo(gen_angle, gen_dihe)` | 调 molrs `neighbors()` 枚举后 `add_angle`/`add_dihedral`；**建议**把感知逻辑下沉为 molrs-core `perceive_topology()` |
| `to_frame()` | 直通 `graph.to_frame()`（molrs 原生，列式） |
| `__iadd__(other)` | molrs 图合并（新增 `MolGraph::extend(&other)`，按偏移搬 id + props） |
| `move/rotate`（SpatialMixin） | molrs `rotate()`；平移加一个 `translate()` |

## 5. molrs-python 绑定扩面（最小集）

**先把层级暴露出来**（G1b）：`molrs-python` 当前只有 `#[pyclass(name="Atomistic")]`。需：

- 导出通用基 `Graph`（= `MolGraph`），标 `#[pyclass(subclass)]`，承载下表所有通用方法。
- `Atomistic` 改为 `#[pyclass(extends=Graph)]`，只留 `add_atom`（element 便捷）+ 不变量校验。
- 新增 `CoarseGrain`（`#[pyclass(extends=Graph)]`，`add_bead` + bead_type 不变量），转发
  `molrs-core::coarsegrain::CoarseGrain`。

下表方法挂在通用基 `Graph` 上（`Atomistic`/`CoarseGrain` 经继承获得）；全部转发到已存在的
`molrs-core` 方法，无新算法（impropers 除外，见 §7-D2）：

| 类别 | 需新增的 Python 方法 |
|---|---|
| 原子属性 | `get_atom_prop(i, key)`、`atom_keys(i)`、`del_atom_prop(i, key)`、`atom_column(key)`（批量） |
| 原子 | `remove_atom(i)`、`iter_atoms()`/`atom_ids()`、`neighbors(i)` |
| 键 | `get_bond_prop/set_bond_prop/get_bond_atoms`、`remove_bond`、`iter_bonds`、`bond_ids` |
| 角 | `add_angle(i,j,k)`、`get/set_angle_prop`、`get_angle_atoms`、`remove_angle`、`iter_angles`、`n_angles` |
| 二面角 | `add_dihedral(i,j,k,l)` + 同上一套 |
| improper | 见 §7-D2（先决） |
| 几何 | `translate(vec)`、`rotate(...)`（core 已有）、`coords()`/`set_coords(arr)`（批量 x/y/z） |
| 合并 | `extend(other)`（id 偏移合并） |

`set_atom_prop`（已实现的通用 setter）是这套 API 的范式：**不为单个属性开命名参数，
统一走 `(key, value)` 通用接口**——绑定扩面必须沿用，避免 kwarg 蔓延。

## 6. 受影响的下游与迁移

机械迁移（随身份模型 `is`→视情况 `==`，对象→句柄）：

- `builder/polymer/ambertools/amber_builder.py`：`bond.itom is port_atom`、
  `id(atom)` 去重 → 改用句柄 `==` / `atom.id`。
- `parser/smiles/*`（`bigsmilesir_to_monomer` 等 `converter_*`）：构图改调 `def_atom`/
  `def_bond`（API 不变，后端变）。
- `io/*` readers/writers：`to_frame`/`from_frame` 路径简化为直通。
- `compute/embed`、`adapter/rdkit`：embed 原地改图，删 `to_molrs/from_molrs` 调用。
- `core/cg.py`：按 §4 与 cg-redesign spec 一并落地。

§7-D1 的句柄缓存策略决定了「`is` 还能否用」——若采纳缓存，多数 `is` 无需改，迁移面显著缩小。

## 7. 设计决策（需 reviewer 拍板）

- **D1 身份/`is` 兼容**：①句柄缓存（`WeakValueDictionary`，保住 `is`，推荐）
  vs ②全量把 `is`→`==`、`id(atom)`→`atom.id`（更纯，但迁移面大）。推荐 ①。
- **D2 impropers**：molrs-core 无 improper。①在 molrs-core 增 `Improper` slotmap（与
  angle/dihedral 对称，推荐，工作量中等）vs ②molpy 侧把 improper 存成带
  `kind="improper"` 属性的 dihedral（零 Rust 改动，但破坏「四级拓扑对称」且 to_frame
  需特判）。推荐 ①。
- **D3 拓扑感知位置**：`get_topo` 的角/二面角枚举留在 Python（调 `neighbors`）vs 下沉
  molrs-core `perceive_topology()`（多语言复用、更快）。建议下沉，但可分期。
- **D4 列访问性能**：`atoms["x"]` 每次 `atom_column` 走一次 FFI 拷贝 vs 暴露零拷贝
  buffer（`to_frame` 已是列式，可作为列访问的统一出口）。建议统一走 `to_frame` 切片。
- **D5 属性类型**：molrs `PropValue` 仅 `F64/Str/Int(i32)`；molpy 现可存任意 Python 值
  （如 `port="<"` 是 str，OK；但 `tuple`/`np.array` 等需约定）。需定义「可进图的属性
  类型集」，其余留在 `Struct` 级 Python 属性（非每原子）。
- **D6 层级落点（核心）**：molpy 的领域类如何「成为通用图的子类」？
  - **方案 A（跨 FFI 继承）**：molpy 类直接继承 molrs PyO3 类——
    `class Atomistic(molrs.Atomistic)`、`class CoarseGrain(molrs.CoarseGrain)`、
    `class Struct(molrs.Graph)`。最 DRY，一条层级贯穿 Rust+Python。代价：PyO3
    `extends` 对**多继承**有约束，molpy 重度依赖的 `SpatialMixin`/`MembershipMixin`/
    `ConnectivityMixin`（纯 Python mixin）与 Rust 基类的 MRO/内存布局需逐一验证可行性。
  - **方案 B（组合 + Python 继承）**：molpy 自己的 `Struct`（**包装** `molrs.Graph`）作基，
    `Atomistic(Struct, *Mixins)` / `CoarseGrain(Struct, *Mixins)` 纯 Python 继承；
    mixin 组合干净，但「is-a 通用图」体现在 molpy 自己的层级里，与 molrs 层级平行而非同一。
  - **方案 C（混合）**：`Struct(molrs.Graph)` 让 molpy 层级**根在** molrs 通用图（满足
    「Atomistic 是通用图的子类」），其余 mixin 仍是纯 Python 协作继承——前提是
    PyO3 单 Rust 基类 + 纯 Python mixin 的 MRO 可用（需先做一个最小 spike 验证）。
  - **推荐**：先做 PyO3 spike 验证 C 是否可行；可行则 C（层级最忠实于用户意图），
    否则退 B（mixin 兼容性最稳，对称性仍由「共享 molrs 图后端」保证）。

  **✅ Spike 结论（2026-06-01，已验证 C 可行）**：把 `PyAtomistic` 标
  `#[pyclass(..., subclass)]` 重建后，Python 端
  `class Struct(molrs.Atomistic, SpatialMixin, MembershipMixin)` +
  `class Atomistic(Struct)` 全部成立：
  - 协作 MRO 正常：`Atomistic → Struct → molrs.Atomistic → SpatialMixin → MembershipMixin → object`；
  - 实例化、`isinstance(a, molrs.Atomistic)`、Rust 方法/属性（`add_atom`/`add_bond`/
    `n_atoms`/`to_frame`）、mixin 方法经 `self` 调 Rust（`n_atoms_times`）、在 Rust 背书
    实例上挂 Python 属性（`self.name`/`self.label`）、领域子类方法——**全部生效**；
  - 既有 `to_molrs`/embed 路径不受影响（CAT 仍 n=32）。

  **唯一约束**：molrs 的 `#[new]` 不收参数，故 molpy `Struct` 需加
  `def __new__(cls, *a, **k): return super().__new__(cls)` 把领域参数路由给 `__init__`；
  或（更干净）把 molrs `#[new]` 签名改为接受并忽略 `*args/**kwargs`，则子类无需 `__new__` 垫片。

  **决策：采用 C**。落地时通用基用导出的 `Graph`（而非示例里的 `Atomistic`）。

  **⚠️ 落地修正（2026-06-01，P1 实测）**：`Struct(molrs.Graph)`（Graph 作 Struct 的基、
  因而在 MRO 中**早于** mixin）会让 `molrs.Graph.rotate` **遮蔽** `SpatialMixin.rotate`
  → 11 处测试 `TypeError: Graph.rotate() got an unexpected keyword argument 'entity_type'`。
  这正是 D6 预警的「mixin 名字冲突」。**最终落地方式**：把 `molrs.Graph` 列为**最后一个基**
  挂在具体类上——`class Atomistic(Struct, MembershipMixin, SpatialMixin, ConnectivityMixin,
  molrs.Graph)`、`class CoarseGrain(... , molrs.Graph)`——于是 MRO 为
  `Atomistic → Struct → *Mixins → Graph → object`：molpy 的方法**全部胜出**，同时
  `isinstance(a, molrs.Graph) == True`（子类关系保住）。`Struct` 本身不继承 Graph，保持
  mixin 友好。`PyGraph.#[new]` 吞 `*args/**kwargs`，故无需 `__new__` 垫片。

  **P1 进度**：上述继承根已落地并全绿。`Atomistic`/`CoarseGrain` 现 is-a `molrs.Graph`。

  **✅ P1 完成（2026-06-01，隔离 worktree，已独立复核全绿）**：在
  `/Users/roykid/work/molcrafts/molpy-p1`（分支 `p1-molgraph-migration`，从主 `dev` 工作树
  快照而来，主树未动）完成存储迁移：继承的 `molrs.Graph` 成为 `Struct` 实体/连接属性与拓扑的
  **唯一后端**，`Entity`/`Link` 降为薄的**双模句柄**。门禁
  `test_core/builder/io/parser` = **1440 passed + 2 既有失败（零新增）**；全量 2058 passed、
  3 既有失败（含 parmchk2 `parameter_level`，stash 验证为既有）。独立抽查确认：`def_atom` 后
  `graph.n_atoms` 反映拓扑、`atom['x']` 经 `get_atom_prop` 代理到图、`bond.itom is atom` 保持。

  实现要点（详见 worktree）：新增 `core/_handle.py`（`_AtomPropProxy`/`_LinkPropProxy`，
  `MutableMapping` over 图属性袋）；`Entity`/`Link` 保持 `UserDict` 子类，detached 时 `self.data`
  为普通 dict，`bind(graph, index)` 后把 `self.data` 换成 live proxy（所有 UserDict 方法经
  `self.data` 自动改道，零逐方法重写）；`Struct` 的 `TypeBucket` 带 binder/unbinder 回调，
  `entities.add` 调基类 `molrs.Graph.add_atom` 再 `bind`，links 按 `_link_kind` 分派到
  bond/angle/dihedral/improper；**保持把句柄对象本身留在 bucket** → `is`/id 去重/copy 重映射
  天然成立（无需 WeakValueDictionary）；molrs 删除会压缩 index，故维护插入序 registry 在 unbind
  时回填 `_index`；D5：仅 int/float/str 进图（排除 bool，因 molrs 把 True→1），其余存每句柄
  fallback dict，读时合并；给 `Entity`/`Link`/proxy 加 `__deepcopy__` 使 `copy()`/reacter 安全。

  **P2 未做（为守住门禁，刻意止步——符合 spec「P1 绿了有余量再做 P2」）**：
  (1) `to_frame` 直通：`molrs.Graph.to_frame()` 返回 molrs 自带的 `builtins.Frame`，非
  molpy IO 层消费的 `molpy.core.frame.Frame`，需先加 Frame 转换层。
  (2) embed 的 `to_molrs`/`from_molrs`：`molrs.generate_3d` 收 `molrs.Atomistic`（非我们的
  `Graph` 子类）且 `add_hydrogens` 改原子数，旧 glue 还承载 `[N+]/[N-]` 的 formal_charge
  正确性（有测试依赖）；删它是真功能改动，非机械删除。二者为 P2 专门处理。

  **✅ P2 完成（embed 部分，2026-06-01，主 dev 全绿）**：删 `core/_molrs.py`，重写唯一消费者
  `embed/__init__.py`：输入直接用 `mol`（已是 `molrs.Graph`，无需 `to_molrs`），在 `mol.copy()`
  上把每原子 `formal_charge`（取自 `formal_charge`/`charge`）写入图，使 `add_hydrogens` 对
  `[N+]/[N-]` 正确；**根因修正（在 molrs，非 molpy 硬编码）**：bond `order` 经 P1 通用
  `set_bond_prop` 把 int `2` 存成 `PropValue::Int(2)`，而 molrs 多处（`bond_order_sum`/
  `rotatable`/`stereo`/`aromaticity`）只按 `F64` 读 `order`，把 Int 当 1.0 → C=O/S=O 双键漏读、
  多加氢。修法：molrs `PropValue::as_f64()`（同时接受 `F64`/`Int`），`hydrogens::bond_order_sum`
  改用它；**不在 molpy 侧按 key 名特判 order**（那是硬编码）。输出经新 helper
  `_atomistic_from_graph` 用 `def_atom`/
  `def_bond` 重建（替代 `from_molrs`，注册表与图同步）。验证：CAT/ANI/OEG 氢数 32/33/38；门禁
  1440 passed + 2 既有；`test_embed/compute/adapter` 208 passed；无 `to_molrs/from_molrs` 残留。
  **P2 余项**：`to_frame` 仍由 `Atomistic.to_frame` 从 graph-backed 实体生成 molpy `Frame`
  （正确、全绿）；直通 `molrs.Graph.to_frame` 需 Frame 转换层，属优化，后续。
  （bond-order 读取健壮性已在 molrs `PropValue::as_f64` 根因解决——见上；同类只读 `F64` 的
  `bond_order_sum`/`rotatable`/`stereo`/`aromaticity`(bond_order + kekule snapshot) **均已**改用
  `as_f64`，接受 `Int`/`F64`；molrs-core 405 tests 全绿。）

## 8. 分期落地（Tasks）

- **P0**（先决，molrs）：D2 impropers + §5 绑定扩面 + `extend`/`translate`/`coords`。
  附 molrs-python 单测（每个新方法一条 round-trip）。
- **P1**（core 句柄层）：`Entity`/`Link` 改句柄 + `Struct` 持 `MolGraph` + 视图容器 +
  D1 句柄缓存。保持 `Atomistic` 公开 API/测试不变作为回归基线。
- **P2**（直通化）：`to_frame`→molrs 直通；`get_topo`→molrs；`__iadd__`→`extend`；
  删除 `core/_molrs.py`。
- **P3**（CG）：`core/cg.py` 按 cg-redesign + 本 spec 落地为同后端对称容器。
- **P4**（下游机械迁移）：builder/parser/io/compute（§6）。
- **P5**（清理）：删旧 Python 平行存储、相关测试重写、benchmark 对比（见 §9）。

## 9. Testing / 验收

- **等价回归**：迁移前用现有 `Atomistic` 跑一遍 `tests/test_core`、`test_builder`、
  `test_io`、`test_parser`、`test_embed` 存档；迁移后逐文件对齐（行为不变）。
- **胶水删除验证**：`grep -r to_molrs|from_molrs src/` 为空；embed 往返不再丢 `port`
  等属性（专门加一条「embed 后自定义属性保留」回归）。
- **身份语义**：D1 选定后，加 `bond.itom is struct.atoms[i]`（缓存）或对应 `==` 测试。
- **带电分子**：`[N+]`/`[N-]` 单体经 embed→GAFF 全程氢数正确、净电荷整数（本仓已有用例）。
- **性能**：100k 原子构图 / `to_frame` / `__iadd__` 的 molgraph 后端 vs 旧 Python 后端
  benchmark，记录到 §9 表（期望内存与构建时间显著下降）。

## 10. Risks / Open Questions

- **R1**：身份模型变更是 core 级 BREAKING，波及 builder/parser 全链；P1 必须以「公开
  API 不变 + 旧测试全绿」作为安全网，否则回滚。
- **R2**：molrs `PropValue` 类型集窄（D5）；若大量现有属性是非标量，进图成本高。需先
  审计 `src/` 里所有写入原子/键属性的类型分布。
- **R3**：跨 Rust 边界的每原子属性读写若不批量化，列访问/序列化可能比纯 Python 慢
  （D4）；必须以 `to_frame` 列式出口兜底。
- **R4**：molrs 与 molpy 同属 molcrafts、可同步演进，但 P0 改 molrs 需 `maturin
  develop --release` 重建并跑 `cargo test -p molcrafts-molrs-core`，纳入 CI。

---

## 附：与既有 spec 的关系

本 spec 是 [`cg-atomistic-mapping-redesign.md`](./cg-atomistic-mapping-redesign.md) 的
**后端实现层**：那份 spec 规定「CG 与 Atomistic 在 core 享有完全相同特权」（API 对称），
本 spec 让两者**共享同一 molrs 图后端**，于是对称性不再靠人工对齐两套 Python 代码维持，
而是结构性保证。两者可合并为一个 P1–P3 实施流。
