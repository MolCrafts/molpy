# Spec: molpy 收成 ECS world 上的句柄视图（删 Struct/Mixin/镜像/拷贝）

| 字段   | 值                                                                 |
| ------ | ------------------------------------------------------------------ |
| Status | Done (2026-06-09; ac-001..004 verified)                            |
| Chain  | molgraph-ecs 3/3（依赖 molrs 01-core + 02-pybind）                 |
| Target | molpy 0.3.x（pre-1.0，允许 BREAKING；底层彻底重构，不保旧 API）    |
| Owner  | core（`atomistic` / `cg` / `entity` / `_handle` / `fields`）       |
| Risk   | 高：身份模型 → 稳定句柄；Entity/Link → 列视图；删 Struct+3 Mixin    |
| Supersedes | `atomistic-cg-on-molrs-molgraph`、`molgraph-views-02-molpy-core` |
| Date   | 2026-06-08                                                         |

---

## 1. Overview

molrs 成 ECS world（01）+ Python 暴露句柄/零拷贝列/自由函数（02）后，把 molpy 分子结构层
**收成 ECS world 上的薄视图**：

- **容器即 world**：`molpy.Atomistic` **IS-A** `molrs.Atomistic`（CG 同），molrs world 是**唯一**存储。
- **Entity/Link = 句柄视图**：`Atom`/`Bond`/… 持 `(world, handle)`，属性读写走 **component 列**
  （热路径零拷贝）；按句柄 intern 保住对象身份（`bond.itom is that_atom`）。
- **扁平化**：**删 `Struct` + `SpatialMixin` + `MembershipMixin` + `ConnectivityMixin`**，塌缩为一个
  `_GraphViews` 薄基；几何/邻接/合并继承自 molrs world 或走自由函数。
- **system 即自由函数**：molpy 算法对齐为 `mp.perceive_aromaticity(mol)` 形态（必要时 molpy 自加薄
  方法糖，但默认函数）。
- **零硬编码字段**：引用内置字段约定（`core/fields.py` canonical 名 / molrs `keys`），**直接访问、
  缺/类型错就报错**——删 `symbol = … or "X"` 占位、删 `_MOLRS_KIND` 散落映射。
- 删两套反模式：`_ordered_*` 镜像 + `_index` 平移；`from_molrs_graph` 深拷贝（→ 零拷贝 `adopt`）。

## 2. Domain basis

- 依赖 01（ECS world + 字段约定 + 自由函数）、02（句柄 + 零拷贝列 + 自由函数 + subclass + adopt）。
- 身份：按**稳定句柄 intern** 保住——同一句柄恒返回同一视图对象，`is`/`in`/`hash` 成立，无需镜像。
- 字段约定：molpy `core/fields.py` 已有 canonical 名；与 molrs `keys` 对齐，两侧引用、不散落字面量。

## 3. Design

### 3.1 扁平层级（IS-A world，无桥）

```
class _GraphViews:                                  # 唯一共享基,~40 LOC: intern + atoms/bonds 迭代 + def_* 骨架
class Atomistic(_GraphViews, molrs.Atomistic):      # + Atom/Bond 视图类型, scale/align/get_topo* 等真算法
class CoarseGrain(_GraphViews, molrs.CoarseGrain):  # + Bead/CGBond 视图类型
```
- **删 `Struct`+3 Mixin**。每方法去向：bucket/binder/`_ordered_*` 镜像/`_bind_*` → 删；
  `copy`/`merge` → 委派 molrs；`rotate`/`translate`/`neighbors` → molrs（继承方法 **或** `mp.rotate(mol)`
  自由函数，二选一，全链一致）；`scale`/`align`/`get_topo*`/`extract_subgraph` → 普通方法/自由函数。
- IS-A world、无 `.to_molrs()` 桥；`molpy.Atomistic()` 直接传给 `mp`/`molrs` 的自由函数 system。
- molpy **不重定义** molrs 存储原语名（不遮蔽）；`def_*` 调继承来的 builder/原语。

### 3.2 Entity/Link = 句柄视图（走 component 列）

- `Atom`/`Bond`/… 持 `(_world, _handle)`；`atom[key]` = `world.get(handle, key)`（节点 component）/
  `bond[key]` = relation component。热路径（typifier/compute 扫标量）走 **`world.column(key)` 零拷贝
  numpy**，不逐原子 FFI。
- **intern 用 `WeakValueDictionary`**（句柄→视图）：身份保证生命周期 =「只要你还持有任一引用」——
  现实比较 `bond.itom is a` 时你正握 `a`，故身份成立，又无 10 万视图泄漏。句柄已失效的视图下次触碰
  即丢弃（lazy invalidation）。
- **无 `_index` 平移**（句柄稳定）；**无 `_MOLRS_KIND` 映射 / `f"{kind}_keys"` 拼名**——Link 子类声明
  规范 kind 名 + 启动断言其 ∈ `world.kinds()`，拼错 bind 时大声报错。

### 3.3 构造 / adopt / 删除（直接访问，错即报错）

- `Atom(element="C")` 独立 = pending 视图（持待写 dict）；`def_atom(**props)`：`h = world.spawn()` →
  按**约定键**写 props（`world.set(h, fields.ELEMENT, "C")`，**无 `or "X"`**）→ intern；**同一对象**变
  bound 视图。`def_bond(a,b,**attrs)`：调继承 builder（`add_bond(a.handle, b.handle)`）→ intern → 写 attrs。
- `Atomistic.adopt(g)`：02 的零拷贝 `adopt` 接管 molrs 产出图（`Conformer`/SMILES）；**删
  `from_molrs_graph` 逐节点深拷贝**。视图按句柄惰性 intern。
- `remove_atom(handle)` = `world.despawn(h)` + intern 逐出；**无平移**。
- 缺字段/类型错一律抛异常（不兜底）。

## 4. Files

- `core/entity.py` — 删 `Struct`+3 Mixin；新增 `_GraphViews` 薄基；`Entity`/`Link` → 句柄列视图；
  删 `_ordered_*` 镜像/平移/dual-mode bind/`_MOLRS_KIND`。
- `core/_handle.py` — proxy 塌缩为按句柄列视图（`world.get`/`world.column`，约定键）。
- `core/atomistic.py` — `Atomistic(_GraphViews, molrs.Atomistic)`，不重定义存储原语名；删
  `from_molrs_graph` 改 `adopt`；`scale`/`align`/`get_topo*` 留；几何/邻接走 molrs。
- `core/cg.py` — `CoarseGrain(_GraphViews, molrs.CoarseGrain)`；Bead/CGBond 视图 + adopt。
- `core/fields.py` — 与 molrs `keys` 对齐；全 core 引用约定常量、不散落字面量。
- 全仓伸进 `.entities`/`.links` TypeBucket 内部的 ≥7 模块（typifier/reacter/optimize/io/parser/
  potential/adapter）：要么 `_GraphViews` 提供按句柄迭代的 `.entities`/`.links` **兼容视图**，要么
  同 PR 改写——impl 计划显式认领（非 ~40 LOC）。
- `tests/test_core/` — 身份/删除/adopt/零拷贝列；其余套件按公共 API 保持绿。

## 5. Tasks

- [ ] Failing pytest: identity via interning (`bond.itom is struct.atoms[i]`, `hash` stable, `WeakValueDictionary` boundary); remove middle atom keeps others valid no reindex; `adopt` zero-copy (source emptied); `atom[fields.CHARGE]` reads through `world.column` (zero-copy bulk path); missing field raises (no `"X"`)
- [ ] Delete `Struct`+3 Mixin → one `_GraphViews`; rebase `Atomistic(_GraphViews, molrs.Atomistic)` / `CoarseGrain(_GraphViews, molrs.CoarseGrain)`; do NOT redefine molrs storage primitives (method-identity test)
- [ ] `Entity`/`Link` → handle column-views; intern via `WeakValueDictionary`; declare canonical kind name on Link + boot assert ∈ `world.kinds()`; drop `_MOLRS_KIND`
- [ ] Replace `from_molrs_graph` with zero-copy `adopt`; `def_atom` via `world.spawn` + convention-key `set` (no `or "X"`); route every field via `core/fields` convention; missing/typed errors surface
- [ ] Decide `.entities`/`.links`: handle-iterating compat view OR rewrite the ≥7 modules in-PR; verify `__post_init__` builder templates (`Water()`, `CH3()`) instantiate through the leaf
- [ ] `pytest tests/ -m "not external"` green (CAT Generate3D n=32; `[N+]`/`[N-]` H-counts); `ruff format --check`, `ruff check`, `ty check src/molpy/`

## 6. Testing strategy

- Identity/interning: `bd.itom is a`, `a in s.atoms`, `list(s.atoms)[0] is a`, `hash(a)` stable; weakref-drop boundary (drop ref → re-iterate yields new object) documented + tested.
- Removal stability: remove middle atom → others + bonds valid, no `_index` corruption.
- Zero-copy: `s.column(fields.X)` is a numpy view; bulk typifier reads columns, not per-atom views.
- adopt: `m = molpy.Atomistic.adopt(molrs_result)` zero-copy (source emptied), properties intact, `from_molrs_graph` removed.
- Flatten: grep shows `Struct`/`SpatialMixin`/`MembershipMixin`/`ConnectivityMixin`, `_ordered_*`, `_MOLRS_KIND` gone; only `_GraphViews` remains.
- No hardcoded fields / no placeholder: field access via `core/fields`; atom without element → `get(h, fields.ELEMENT)` is null/raises, no `"X"`.
- Backward-compat (load-bearing): `pytest tests/ -m "not external"` green incl. test_core/test_io/test_parser/embed; CAT n=32; `[N+]`/`[N-]` H-counts.

## 7. Out of scope

- molrs ECS / 暴露（01/02，前置依赖）。
- 层级容器 / 分组（residue/chain、CG-mapping 行为）——独立后续 spec。**依赖提醒**：CG `bead[atoms]`
  引用另一 world 的原子；句柄 world-局部、不跨 world，跨结构引用需显式（行位置 + 源 world 引用，
  非句柄编码），CG-mapping spec 先解。
- Frame/列 buffer 改写——既有 `molrs-backend` 范畴。
