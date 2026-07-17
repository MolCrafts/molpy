---
title: "molpy.core 收成 molrs 的 re-export + Python 语法糖"
slug: molrs-core-cutover
status: in-flight
created: 2026-07-14
owner_crates: "molcrafts-molrs, molrs-python, molcrafts-molpy"
---

# `molpy.core` 不再是第二个内核

## Summary

`/Users/roykid/work/molcrafts/molpy/src/molpy/core` 中，除 `logger.py`、`script.py`、
`config.py` 仍由 molpy 实现外，其他模块全部改成以 **molrs 现有类型和算法为真值**。

`molpy.core` 只允许两种东西：

1. **re-export**：只用于 molpy 仍拥有语义的图与力场表面。`Frame` / `Block`
   不在此列；它们只由 `molrs` 导出，调用方必须直接导入。
2. **Python 语法糖**：对 molrs 类的真继承或无状态辅助层，只负责 Python
   调用体验（别名、链式返回 `self`、批量参数整形、切片、回调组合、绘图/展示）。

不允许再出现第二份坐标计算、图遍历、元素表、单位引擎、力场数据、
scaleLJ 公式或任何 `_inner` 转发门面。一句话：

> **molrs 做计算和存储；`molpy.core` 只让它更像 Python。**

本 spec 是一次所有权切换，不是把 4,890 行 Python 原样复制到 molrs-python。

## 先看 molrs 已经有什么

以当前工作树为准，molrs 并不是一个空后端：

| `molpy.core` 领域 | molrs 已有的真值 | 当前缺口 |
|---|---|---|
| `frame` | Rust `Block`/`Frame`；`molrs.frame` 已提供 dict-like、切片、排序、CSV、零拷贝 NumPy view | 无；删除 molpy 的模块与 re-export，用户直接 `from molrs import Frame, Block` |
| `entity` | `molrs.views` 已有 `NodeRef`/`RelationRef`、`Entity`/`Link`、`Atom`/`Bond`/高阶关系、`Bead`/`CGBond`、句柄实例化 | 无，molpy 可直接 re-export |
| `atomistic` | Rust ECS world、稳定句柄、原子/键/角/二面角/不正则项、拓扑生成、BFS 距离、子图抽取、copy/merge、Frame 投影、结构哈希；`molrs.views.Atomistic` 已有 `def_*` 句柄 API | 已补 native direction align 与无 callback replicate；仅 callback/chaining 留 Python |
| `cg` | Rust `CoarseGrain`、bead membership 双向映射、copy/merge、子图、Frame 投影；`molrs.views.CoarseGrain` 已有 `def_bead`/`def_cgbond` | 已补与 atomistic 对称的 align/replicate |
| `box` | Rust `SimBox`已有 free/orthogonal/triclinic、volume/style/lengths/tilts、fractional/cartesian、wrap、minimum-image delta、`isin`；PyO3 已暴露主干 | 已补 bounds/from_bounds、distance/pairwise distance、image/unwrap、构造转换与 transform |
| `region` | Rust `Sphere`/`HollowSphere`/`Cuboid`、`And`/`Or`/`Not`、`contains`/`bounds`；PyO3 已暴露 | 只缺 `Block` 坐标列选择和旧构造器名称，属 Python 语法糖 |
| `trajectory` | Rust `Trajectory` 存储、step/time 校验、Frame 索引、Zarr；PyO3 已暴露 | topology 附着、slice/map/split 为 Python 语法糖 |
| `forcefield` | Rust `ForceField`/potentials；`molrs.forcefield` 已有 Style/Type handle view、链式 `def_*style`、主要实名 style | molpy 还有一组“只固定 style name”的便利子类 |
| `fields` | Rust `keys`；`molrs.fields` 已有 `FieldSpec`、规范字段和格式边界 | molpy 自有 `SITE` 标记与 `ForceFieldFormatter` |
| `element` | Rust 已有 1..118 元素、symbol/name/Z、质量、共价/范德华半径、默认价态；crate root 的 PyO3 `molrs.Element` 是唯一公开导出 | 无；删除 molpy 模块与 re-export |
| `unit` | Rust 已有 `UnitRegistry`/`Unit`/`Quantity`、MD/LJ 单位表、复合表达式解析、维度检查和转换；PyO3 已暴露 | 无；molpy 只保留 preset 名与 LJ 构造糖 |
| `selector` | `Block` 已有 mask/fancy indexing，`NeighborQuery` 已有空间查询 | 选择器的 Python 布尔 DSL 本身可留作语法糖 |
| `utils` | `Block.from_csv`、live graph views 已覆盖真实用途 | 无生产消费者；模块整体删除 |
| `ops.geometry` | Rust 已有 translate/rotate/scale、matrix/math 基础 | align 补齐后，该 Python 数值帮助模块可删 |
| `ops.scale_lj` | molrs 已有 `FragmentScaling`、编译期 CL&Pol 表、COM/kij 与 FF clone+rewrite | molpy 只留调用形状适配 |

因此切换不是“先把 molpy 搬过去”，而是：**删掉已被 molrs 覆盖的重复，
只对上表的真缺口补 molrs/PyO3。**

## 设计边界

### 1. 什么可以留在 `molpy.core`

允许：

- `from molrs... import ...`、别名、`__all__`、类型标注；
- 为了 Python API 的真继承，例如 `class Box(molrs.Box)`；
- 将 `Atom` 对象换成 native handle、将 list/dict 批量拆成多次 native 调用；
- 别名和链式语法，例如 `move(...) -> self`、`cubic -> molrs.Box.cube`；
- Python callback 组合（`select(predicate)`、`Trajectory.map`、自定义 split strategy）；
- 展示/绘图/`repr`、`to_dict`、字段名格式化；
- `logger.py`、`script.py`、`config.py` 的完整 molpy 实现。

禁止：

- NumPy/Python 重新计算 PBC、距离、旋转、对齐、拓扑或子图；
- Python 循环改写 force-field 参数、计算 scaleLJ 或其他科学公式；
- 第二份元素表、单位定义表、字段注册表或句柄存储；
- `_inner`、`to_molrs()`/`from_molrs()`、每个方法手工转发的门面类；
- molrs 调用失败时的 Python fallback；缺能力就先补 molrs，不静默回退。

判定标准不是“代码短不短”，而是：**这段代码是否在产生新的科学/数据结果**。
产生结果属 molrs；只改变 Python 表达属 molpy 语法糖。

### 2. 模块处置

| molpy 文件 | 目标形状 | 必须删除/下沉的实现 |
|---|---|---|
| `core/frame.py` | **删除**；`Frame` / `Block` 只从 `molrs` 导入 | 所有别名、re-export 与 import hook |
| `core/entity.py` | **纯 re-export** `molrs.views` 公开句柄类 | 无（已达标） |
| `core/atomistic.py` | `molrs.Atomistic` 的薄继承，只留批量构建、别名、链式、Python predicate 等语法糖 | `_vec_*` 几何、align 数学、拓扑/抽取算法、任何存储镜像 |
| `core/cg.py` | 与 atomistic 对称的薄继承 | 同上 |
| `core/box.py` | 继承 `molrs.Box`，只留公开构造器、公开属性、array/repr/hash/plot/serialization | 所有 PBC、matrix conversion、distance/image/unwrap 计算以及 `_matrix`/`_origin`/`_pbc` 私有别名 |
| `core/region.py` | 对 `molrs.Sphere/Cuboid/Region` 的 `coord_field` + `MaskPredicate` DSL | `isin`/布尔区域/bounds 数学不再用 NumPy 复制 |
| `core/trajectory.py` | native container 薄继承 + topology/slice/map/split 语法糖 | 不自建 frame/step/time 存储 |
| `core/forcefield.py` | re-export molrs 层级；只留固定 style-name 的空子类和兼容别名 | 任何 style/type 数据、匹配、参数改写实现 |
| `core/fields.py` | re-export `molrs.fields`；只留 molpy 组装语义 `SITE` 和 I/O 格式化语法糖 | 规范字段名/dtype 的第二注册表 |
| `core/element.py` | **删除**；`Element` 只从 `molrs` 导入 | `ElementData`、`0 -> X`、Python 周期表、半径 fallback 和所有 re-export |
| `core/unit.py` | `molrs.UnitRegistry` 上的 `UnitSystem` Python 语法糖；只管 preset 名和 LJ 尺度构造 | Pint 作为第二单位引擎；单位解析/换算必须走 molrs |
| `core/selector.py` | 保留 Python mask DSL，计算通过 Block/NeighborQuery | 禁止生长成第二空间算法层 |
| `core/utils.py` | **删除** | CSV 使用 `Block.from_csv`；dead `TypeBucket` 不保留 |
| `core/ops/geometry.py` | **删除** | align 所需算法进 molrs |
| `core/ops/scale_lj.py` | 兼容函数只把 Python fragments 整形后调 `molrs` | `compute_k_ij`、质心、FF 改写、参数文件解析全部进 molrs |
| `core/__init__.py` | 组合 molrs re-export、上述 sugar 与三个 molpy-owned 模块 | 无实现 |
| `core/logger.py` / `script.py` / `config.py` | **保持 molpy-owned** | 不下沉 |

`SITE` 不进 molrs 规范字段表：它是 molpy 装配层的语义，不是分子数据内核的
通用事实。这不是第二内核，而是一个上层 schema 扩展。

### 3. molrs 必须先补齐的缺口

molpy 不允许为了切换而写 fallback。以下能力在 molrs 发版前是硬前置：

1. **Element PyO3**
   - 绑定 Rust `Element`，支持 1..118 的 name/symbol/Z 大小写无关查询；
   - 暴露 `number/name/symbol/mass/vdw/covalent`；质量真值也必须进 Rust 元素表；
   - 保留 `get_symbols`/`get_atomic_number` 便利 API，未知元素 fail-fast；
   - molpy 不再有 `_COVALENT_RADII` 或 118 行 Python 数据。

2. **Units PyO3**
   - 绑定 Rust `UnitRegistry`/`Unit`/`Quantity` 及维度错误；
   - 暴露 parse/quantity/define/conversion/arithmetic/显示和 Python 标量运算；
   - `UnitSystem` 可作为可继承的 native registry 上的语法糖，LAMMPS presets
     和 LJ 尺度只构造 native unit definitions；
   - 切换后 molpy 删除 `pint` 运行时依赖。Pint 特有 context 如无同义 native
     契约，作为明示 breaking change 记录，不伪造半套兼容。

3. **Box/geometry 暴露补齐**
   - 优先绑定 Rust `SimBox` 已有的 `shortest_vector`、distance、fractional/cartesian、
     nearest-plane distance；
   - image/unwrap、lengths+angles 转 matrix、一般晶胞转 restricted triclinic 若 Rust 尚无，
     实现在 `molrs::core::spatial`，再绑定；
   - 输入形状和 free/partial-PBC 语义只有一份。

4. **Graph geometry**
   - Kabsch/四元数刚体对齐进 `molrs::core::spatial::geometry`，Atomistic 和
     CoarseGrain 共用；
   - translate/rotate/scale 继续使用现有 native 函数；
   - 无 callback 的 replicate 路径进 native；Python callable transform 可作 molpy 语法糖，
     但每次复制/变换/合并仍调 native。

5. **CL&Pol scaleLJ**
   - `FragmentScaling` 与 `k_ij` 公式进 molrs FF 层；
   - fragment COM 和 pair-type epsilon/sigma 改写在 native 路径一次完成；
   - `clpol_fragments.ff` 不再由 Python 运行时解析。按 molrs 参数房规，真值进
     `molrs/src/ff/params/` 的编译期 Rust 表，文件头记来源；
   - molpy `scale_lj(...)` 仅整形 legacy `Mapping[str, Sequence[Atom]]` 后委托 native。

### 4. 身份、存储与错误契约

- 纯 re-export 符号必须满足 `is`，不是同名的新类。
- 语法糖类必须 `issubclass(MolpyType, MolrsType)`，不得持有 `_inner`。
- Atom/Link/Block/Frame 与坐标列不做边界拷贝；向量化写入仍写回 Rust 存储。
- molrs 错误原样跨过边界；molpy 可加调用上下文，不得捕获后 fallback。
- `molrs` 不得依赖/`import molpy`；依赖箭头永远是 `molpy -> molrs`。

### 5. Region 同名冲突

molrs 已有 native `Region` 类，molpy 也有 `MaskPredicate` ABC 叫 `Region`。不得用一个新
Python `Region` 静默覆盖 `molrs.Region`。

目标：

- `molrs.Region` 仍是 native 组合区域类；
- molrs-python 将 `Sphere`/`Cuboid`/`Region` 标成可继承，并给 native 组合结果提供
  构造 Python 子类所需的受控入口；
- `molpy.core.region.Region` 是只增 `coord_field` 与 `mask(Block)` 的 selection mixin；
  `BoxRegion`/`SphereRegion`/`AndRegion`/`OrRegion`/`NotRegion` 是对应 native 类的
  **真继承**，不持有 `_inner`/`_region` 作组合转发；
- `BoxRegion`/`SphereRegion` 的 `contains`/boolean composition/bounds 全部由 native region 完成；
- 两个公开名称都有类型测试，不用 top-level shadow 伪装统一。

## 实施顺序

### Phase A — 冻结当前契约

- 从 `molpy/tests/test_core/` 建立公开符号清单与行为 oracle；
- 将每个符号标成 `identity re-export` / `Python sugar` / `molrs gap` / `remove`；
- 记录必须 breaking 的部分（目前明确的候选是 Pint context 特有 API）。

### Phase B — 先完成 molrs

- 补 Element 与 units PyO3；
- 补 Box/geometry/align/replicate 的 native 缺口；
- 补 native scaleLJ 与编译期参数表；
- 更新 `molrs.pyi`、`molrs.__init__`、molrs-python 行为测试；
- 单独发布 molrs 新版本。不允许 molpy 先指向尚未发布的 API。

### Phase C — 收缩 `molpy.core`

- 将已覆盖的模块改为身份 re-export；
- 重写 atomistic/cg/box/region/trajectory/unit 为本 spec 白名单内的语法糖；
- 删 `ops/geometry.py`、Python Element 表、Pint 引擎、Python scaleLJ 算法与参数解析；
- `pyproject.toml` 精确 pin 新 molrs，删除 molpy 的 `pint` 依赖；
- 更新 `molpy.__init__`/`molpy.core.__init__`，用户仍只 `import molpy`。

### Phase D — 测试所有权迁移

- 核心数学/数据测试迁到 molrs Rust 或 molrs-python；
- molpy 保留 re-export 身份、语法糖、下游集成测试；
- 每个下沉算法先证明 molrs 测试在故意破坏时会红，再删 molpy 对应实现。

## Files to create or modify

### molrs / molrs-python

- `molrs/src/core/system/element.rs` — 补齐质量/旧契约所需真值（若现有表不足）
- `molrs/src/core/spatial/{geometry,region/simbox}.rs` — align 与 Box 缺口
- `molrs/src/core/units/*` — 仅补 PyO3 契约暴露所需的底层能力
- `molrs/src/ff/params/` + FF API — CL&Pol fragment table 与 scaleLJ
- `molrs-python/src/core/system/element.rs` (new)
- `molrs-python/src/core/units/` (new)
- `molrs-python/src/core/spatial/simbox.rs`
- `molrs-python/src/core/system/molgraph.rs`
- `molrs-python/src/ff/` — scaleLJ binding
- `molrs-python/python/molrs/__init__.py`
- `molrs-python/python/molrs/molrs.pyi`
- 对应 Rust 与 Python 测试

### molpy

- `src/molpy/core/` 下除 `logger.py`/`script.py`/`config.py` 的全部模块
- `src/molpy/__init__.py`
- `pyproject.toml`
- `tests/test_core/` — 收成身份/语法糖/集成测试
- `CHANGELOG.md` 与公开 API 文档

## Tasks

- [x] T1 生成当前 `molpy.core` 公开符号 + 行为清单，逐个分类
- [x] T2 molrs Element 补数据、PyO3 绑定与全 1..118 回归；0/X 明确失败
- [x] T3 molrs units PyO3；`UnitSystem` 只剩 native registry 上的 preset/LJ 语法糖
- [x] T4 补齐 Box/geometry/align/replicate native API，删 Python 数值重复
- [x] T5 scaleLJ 公式、COM、FF 改写与 fragment table 全部进 molrs
- [ ] T6 molrs-python stub/导出/测试完成，发布新版本
- [x] T7 Frame/Block/Element 直接使用 molrs 并删除 molpy 兼容面；entity 等保持 identity re-export
- [x] T8 atomistic/cg/box/region/trajectory/forcefield/fields/selector 收到语法糖白名单
- [x] T9 删 `ops/geometry.py`、`utils.py`/`TypeBucket`、Pint 引擎、Python 元素表、Python scaleLJ 算法/参数解析
- [ ] T10 molpy pin 新 molrs；全量回归、文档与 CHANGELOG

## Testing strategy

### 身份与架构门禁

- molrs-only：`Frame`/`Block`/`Element` 在 molpy 顶层、`molpy.core` 与子模块旧路径全部导入失败。
- 纯 re-export：Entity 句柄等逐个 `is`。
- 语法糖：逐个 `issubclass`，并检查无 `_inner`、无 conversion bridge。
- AST 白名单：`molpy.core` 中除三个 molpy-owned 文件外，任何新增 class/function
  都必须归入“re-export 或语法糖”清单；不允许清单外的科学函数长回来。
- 依赖门禁：`molrs` 树中 `import molpy` / `from molpy` 零命中。

### 行为与科学回归

- Atomistic/CG：句柄稳定、句柄视图写回、copy/merge/extract/to_frame、align/replicate。
- Box：free/orthogonal/triclinic + partial PBC，wrap/unwrap/image/minimum-image/distance/
  fractional round-trip 对现有 molpy oracle 数值一致。
- Region：native contains/composition/bounds 与 molpy `mask(Block)` 组合，特别测同名
  `molrs.Region` 不被 shadow。
- Units：现有 7 组 LAMMPS presets、维度错误、亲和单位、LJ 四个尺度转换；
  breaking 部分用明确负测试证明已移除，不静默改义。
- Element：1..118 全扫描；symbol/name/Z 往返；质量与半径对权威 Rust 真值；0/X/越界输入 fail-fast。
- scaleLJ：当前 CL&Pol 测试全部迁到 molrs，断言 epsilon 缩放、sigma 可选缩放、
  charge 不变、输入 FF 不变、缺 fragment fail-fast。
- 运行 molrs `fmt + clippy + check + test`、molrs-python 全测试、molpy 全测试与类型检查。

## Out of scope

- `logger.py`、`script.py`、`config.py` 的下沉或重设计。
- 把 Python callback/plot/repr 强行写成 Rust。
- 顺手改变分子模型、力场公式、Box 数学或选择语义；发现缺陷另立 spec。
- 删除 `molpy.core.*` 旧 import path；它们就是本 spec 要保留的兼容层。
- 要求用户直接 `import molrs`；用户文档仍只展示 `molpy`。
