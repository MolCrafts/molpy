# Spec: 简化 CoarseGrain 至与 Atomistic 对称

| 字段   | 值                                  |
| ------ | ----------------------------------- |
| Status | Draft                               |
| Target | molpy 0.3.x（pre-1.0，允许 BREAKING）|
| Owner  | core/cg                             |
| Layers | core（单文件改写）                   |
| Risk   | 中：测试需重写，公开 API 形态变化    |
| Date   | 2026-05-03                          |

---

## 1. Overview / Motivation

`molpy.core.cg` 当前在 core 层钦定了一组"有多种合理替代实现"的领域选择：

- `Bead.atomistic: Atomistic | None` — bead 持有完整 AA 子图
- `Bead.x/y/z` 由 `from_atomistic` 自动写入（`mean(positions, axis=0)`，命名为 *center of mass* 实为 *center of geometry*）
- `CoarseGrain.from_atomistic` 强制走 partition mask + crossing-bond 推断
- `CoarseGrain.to_atomistic` 在数据结构层做反向展开

这些选择本身都不是错的；**错的是 core 不该选边**。COM / COG / weighted centroid、bond 推断策略、AA→CG 投影策略、CG→AA 反向展开重建——每一项都有多个合理实现，分别对应不同 CG 力场（Martini 2 vs 3、VOTCA、PyCGTOOL、DPD、自定义）。一旦 core 钉死其中一条，用户偏离主流时就要绕过 MolPy。

本 spec 的指导原则：

> **CG 在 core 层享有的特权应与 Atomistic 完全相同——不多、不少。**

判据：

> **core 方法 = 在已锁定的约定上、没有第二种合理实现的操作。**

`atomistic.move(delta)` 满足判据（"位置在 x/y/z" 是 SpatialMixin 锁定的约定，加 delta 没有第二种实现）。`atomistic.center_of_mass` 不满足（mass 缺失？hydrogens 包不包？），所以它不在 core 上。

CG 应用同一判据：`bead["atoms"]` 是与 `entity["x/y/z"]` 同档的约定 key（可选、不强制、不验证）；在它上面做 `beads_of(atom)` 这种"包含查询"是 core 一等公民，等价于 `move`。除此之外，所有投影 / 推断 / 派生量算子全部不在 core，**也不在 op 层**——MolPy 不提供。用户按场景自己写一行 numpy / 一个 list comprehension 即可。

## 2. Goals & Non-Goals

### Goals

- G1：`Bead` 是空 `Entity`，零强制字段；与 `Atom` 结构完全对称。
- G2：`CGBond` 是只持两个 Bead endpoints 的 `Link`；与 `Bond` 对称。
- G3：`CoarseGrain = Struct + MembershipMixin + SpatialMixin + ConnectivityMixin`；公开成员逐项对应 `Atomistic`，仅多一个 `beads_of(atom)`。
- G4：约定 `bead["atoms"]`（若存在）是 `tuple[Atom, ...]`，与"位置在 `entity["x/y/z"]`" 同档：是 core 方法依赖的约定 key，但 core 不写入、不验证、不强制。
- G5：BREAKING 一次到位，不保留 deprecated 路径。

### Non-Goals

- NG1：**不**提供 `from_atomistic` / `to_atomistic` / 任何 AA↔CG 投影函数（core、op、builder 都不写）。用户自行实现：
  ```python
  cg = CoarseGrain()
  for group in groups:
      cg.def_bead(atoms=tuple(group), type="CH3")
  ```
- NG2：**不**提供 `bead_center_of_mass` / `bead_center_of_geometry` / `bead_total_mass` / `coarsegrain_positions` 等派生量算子；MolPy 既不在 `core/ops/` 也不在 `op/` 提供。
- NG3：**不**提供 `infer_cg_bonds` / 自动 bond 推断；用户按需写一个三行 list comprehension。
- NG4：**不**引入 `Mapping` 类、`source: Atomistic | None` 属性、`_atom_to_beads_cache` 缓存。
- NG5：**不**改动 `Atom`、`Bond`、`Atomistic`、`Entity`、`Link`、`Struct`、Mixin 任何代码。
- NG6：**不**改动 `CGSmiles` parser 或新增 IO 格式。
- NG7：**不**新增 `src/molpy/core/ops/cg.py` 模块。

## 3. Background — 当前实现的具体问题

### 3.1 `Bead.atomistic` 字段是数据结构耦合

`src/molpy/core/cg.py:19-49`：构造器有两条互不相容的路径（dict copy-compat / 正常 `atomistic=` kwarg），`atomistic` 字段在 `__deepcopy__` 后被「神秘地」重设。这是「为了让 `copy.deepcopy` 工作而设计的 API」，不是为了表达 CG 的语义。

更深的问题：把"bead 来自哪些 atom"做成 typed field 强制存在，等价于 `Atom` 强制带 `element` 字段——剥夺了用户表达"纯 CG bead，无 AA 前体"（DPD、Martini 自洽运行、CGSmiles 解析）的能力。

### 3.2 `from_atomistic` 把多个领域选择捆绑

`src/molpy/core/cg.py:439-548`：

```python
positions_array = np.array(positions)
center = positions_array.mean(axis=0)            # 钦定 COG
...
bead = result.def_bead(
    atomistic=subgraph,                          # 钦定持有 AA 子图
    x=float(center[0]), y=float(center[1]), z=float(center[2]),  # 钦定写入位置
)
...
for bond in atomistic.bonds:                     # 钦定推断策略
    ...
```

四个独立的领域选择（位置算法、子图持有、自动推断、partition-only 输入）固化在一个工厂里。用户想换其中任何一个，只能整体绕开。

### 3.3 `to_atomistic` 不是数据结构层职责

`src/molpy/core/cg.py:375-437` 既要处理"bead 有 AA 子图就展开"，又要处理"无子图就按 bead 位置造一个虚拟 atom"。这种 CG → AA 反向重建是 builder 层职责（按 CG 模型挑选构象、放置、对齐），不应该是 `CoarseGrain` 自带方法。

### 3.4 与 Atomistic 不对称

`Atomistic` 没有 `from_smiles`、`to_pdb`、`compute_center_of_mass` 这些方法——它们分别属于 `parser`、`io`、`compute` 层。`CoarseGrain` 当前却把对应级别的功能全塞进 core。这是 core 层的不对称特权。

## 4. Proposed Design

### 4.1 全部 core 代码

```python
# src/molpy/core/cg.py
from __future__ import annotations
from typing import Any

from .entity import (
    ConnectivityMixin,
    Entities,
    Entity,
    Link,
    MembershipMixin,
    SpatialMixin,
    Struct,
)


class Bead(Entity):
    """Coarse-grained bead.

    Structurally identical to :class:`~molpy.core.atomistic.Atom`: an empty
    :class:`Entity` with no mandatory fields. All bead state lives in the
    underlying dict via the :class:`Entity` interface.

    Conventional keys (none enforced, all optional):

    * ``bead["atoms"]`` — ``tuple[Atom, ...]`` of atom references this bead
      represents, when derived from an :class:`Atomistic`. Required for
      :meth:`CoarseGrain.beads_of`. Same convention level as
      ``entity["x/y/z"]`` for spatial operators.
    * ``bead["x"]``, ``bead["y"]``, ``bead["z"]`` — primary position (used by
      :class:`SpatialMixin` ``move`` / ``rotate`` / ``scale`` / ``align``).
    * ``bead["type"]`` — type label.
    * ``bead["mass"]``, ``bead["charge"]`` — as user needs.

    Deriving anything (centroid from ``atoms``, total mass, etc.) is the
    user's job; MolPy does not pick a definition.
    """


class CGBond(Link):
    """CG bond between two beads."""

    def __init__(self, a: Bead, b: Bead, /, **attrs: Any) -> None:
        assert isinstance(a, Bead), f"a must be Bead, got {type(a)}"
        assert isinstance(b, Bead), f"b must be Bead, got {type(b)}"
        super().__init__([a, b], **attrs)

    def __repr__(self) -> str:
        return f"<CGBond: {self.ibead} - {self.jbead}>"

    @property
    def ibead(self) -> Bead:
        return self.endpoints[0]

    @property
    def jbead(self) -> Bead:
        return self.endpoints[1]


class CoarseGrain(Struct, MembershipMixin, SpatialMixin, ConnectivityMixin):
    """Coarse-grained structure.

    Mirrors :class:`~molpy.core.atomistic.Atomistic` in every public surface
    except for the single extra method :meth:`beads_of`, which performs the
    reverse lookup that ``Atomistic`` has no analog for.
    """

    def __init__(self, **props: Any) -> None:
        super().__init__(**props)
        # Call __post_init__ if it exists (template pattern; mirrors Atomistic).
        if hasattr(self, "__post_init__"):
            for klass in type(self).__mro__:
                if klass is CoarseGrain:
                    break
                if "__post_init__" in klass.__dict__:
                    klass.__dict__["__post_init__"](self, **props)
                    break
        self.entities.register_type(Bead)
        self.links.register_type(CGBond)

    @property
    def beads(self) -> Entities[Bead]:
        return self.entities[Bead]

    @property
    def cgbonds(self) -> Entities[CGBond]:  # type: ignore[type-var]
        return self.links[CGBond]  # type: ignore[return-value]

    def __repr__(self) -> str:
        from collections import Counter
        types = Counter(b.get("type", "?") for b in self.beads)
        comp = (
            " ".join(f"{t}:{n}" for t, n in sorted(types.items()))
            if len(types) <= 5 else f"{len(types)} types"
        )
        return f"<CoarseGrain, {len(self.beads)} beads ({comp}), {len(self.cgbonds)} bonds>"

    def __len__(self) -> int:
        return len(self.beads)

    # ---------- factory: parallel to Atomistic.def_atom / def_bond ----------

    def def_bead(self, /, **attrs: Any) -> Bead:
        """Create and register a new Bead. Parallel to ``Atomistic.def_atom``."""
        bead = Bead(**attrs)
        self.entities.add(bead)
        return bead

    def def_cgbond(self, a: Bead, b: Bead, /, **attrs: Any) -> CGBond:
        """Create and register a new CGBond. Parallel to ``Atomistic.def_bond``."""
        bond = CGBond(a, b, **attrs)
        self.links.add(bond)
        return bond

    def add_bead(self, bead: Bead, /) -> Bead:
        self.entities.add(bead)
        return bead

    def add_cgbond(self, bond: CGBond, /) -> CGBond:
        self.links.add(bond)
        return bond

    def del_bead(self, *beads: Bead) -> None:
        self.remove_entity(*beads)

    def del_cgbond(self, *bonds: CGBond) -> None:
        self.remove_link(*bonds)

    def def_beads(self, beads_data: list[dict[str, Any]], /) -> list[Bead]:
        return [self.def_bead(**a) for a in beads_data]

    def def_cgbonds(self, bonds_data, /) -> list[CGBond]:
        out = []
        for spec in bonds_data:
            if len(spec) == 2:
                a, b = spec; attrs = {}
            else:
                a, b, attrs = spec
            out.append(self.def_cgbond(a, b, **attrs))
        return out

    def add_beads(self, beads: list[Bead], /) -> list[Bead]:
        for b in beads: self.entities.add(b)
        return beads

    def add_cgbonds(self, bonds: list[CGBond], /) -> list[CGBond]:
        for b in bonds: self.links.add(b)
        return bonds

    # ---------- the one extra core method: reverse lookup ----------

    def beads_of(self, atom: "Atom") -> tuple[Bead, ...]:
        """Return all beads whose ``bead["atoms"]`` contains ``atom``.

        Empty tuple if no bead references this atom; multiple if the mapping
        has overlap. Beads without an ``"atoms"`` key are silently skipped.

        Justified as core (parallel to :meth:`move`): operates on a single
        conventional key (``"atoms"``) and has no second reasonable
        implementation. Cost is O(N_beads × mean group size) — no caching;
        callers needing speed should build their own ``id(atom) → beads``
        index.
        """
        return tuple(b for b in self.beads if atom in b.get("atoms", ()))

    # ---------- selection / type editing (parallel to Atomistic) ----------

    def rename_type(self, old: str, new: str, *, kind: type = Bead) -> int:
        if issubclass(kind, Link):
            items = self.links.bucket(kind)
        else:
            items = self.entities.bucket(kind)
        n = 0
        for it in items:
            if it.get("type") == old:
                it["type"] = new; n += 1
        return n

    def set_property(self, selector, key: str, value: Any, *, kind: type = Bead) -> int:
        if not callable(selector):
            raise TypeError("selector must be callable")
        items = self.links.bucket(kind) if issubclass(kind, Link) else self.entities.bucket(kind)
        n = 0
        for it in items:
            if selector(it):
                it[key] = value; n += 1
        return n

    def select(self, predicate) -> "CoarseGrain":
        if not callable(predicate):
            raise TypeError("predicate must be callable")
        selected = [b for b in self.beads if predicate(b)]
        sub, _ = self.extract_subgraph(selected, radius=0, entity_type=Bead, link_type=Link)
        return sub  # type: ignore[return-value]

    # ---------- spatial (return self for chaining; same as Atomistic) ----------

    def move(self, delta, *, entity_type: type[Entity] = Bead) -> "CoarseGrain":
        super().move(delta, entity_type=entity_type); return self

    def rotate(self, axis, angle, about=None, *, entity_type: type[Entity] = Bead) -> "CoarseGrain":
        super().rotate(axis, angle, about=about, entity_type=entity_type); return self

    def scale(self, factor, about=None, *, entity_type: type[Entity] = Bead) -> "CoarseGrain":
        super().scale(factor, about=about, entity_type=entity_type); return self

    def align(self, a, b, *, a_dir=None, b_dir=None, flip=False,
              entity_type: type[Entity] = Bead) -> "CoarseGrain":
        super().align(a, b, a_dir=a_dir, b_dir=b_dir, flip=flip, entity_type=entity_type)
        return self

    # ---------- system composition (parallel to Atomistic) ----------

    def __iadd__(self, other: "CoarseGrain") -> "CoarseGrain":
        self.merge(other); return self

    def __add__(self, other: "CoarseGrain") -> "CoarseGrain":
        result = self.copy(); result.merge(other); return result

    def replicate(self, n: int, transform=None) -> "CoarseGrain":
        result = type(self)()
        for i in range(n):
            r = self.copy()
            if transform is not None: transform(r, i)
            result.merge(r)
        return result
```

### 4.2 删除清单

| 当前 API                              | 处置                                              |
| ------------------------------------- | ------------------------------------------------- |
| `Bead.__init__(data_dict, atomistic=None, **attrs)` | **删除**双签名 hack；改为 `Bead(**attrs)` |
| `Bead.atomistic` 字段                 | **删除**；用户用 `bead["atoms"]` 约定键           |
| `Bead.__deepcopy__`                   | **删除**；用 `Entity` 默认 deepcopy（dict 深拷贝） |
| `CoarseGrain.from_atomistic(...)`     | **删除**；用户自己写（见 §5）                     |
| `CoarseGrain.to_atomistic(...)`       | **删除**                                          |
| `def_bead(atomistic=...)` 参数        | **删除**；统一为 `def_bead(**attrs)`              |

### 4.3 不新增的清单

明确不在本 spec 也不在后续 spec 提供：

- `core/ops/cg.py` 模块
- `op/cg.py` 模块（中的 CG 相关 helper）
- `bead_center_of_mass` / `bead_center_of_geometry` / `bead_total_mass` / `bead_total_charge` / `coarsegrain_positions` / `coarsegrain_by_groups` / `infer_cg_bonds` / `bead_centroid` 任何函数

如果未来用户反复手写同一段代码，可在那时按需起新 spec 提议加入 op 层。本次不预设。

### 4.4 Module Changes

| File                          | Action  | Description                                       |
| ----------------------------- | ------- | ------------------------------------------------- |
| `src/molpy/core/cg.py`        | Rewrite | 见 §4.1，约 150 行                                  |
| `src/molpy/__init__.py`       | No-op   | `Bead`、`CGBond`、`CoarseGrain` 已导出              |
| `tests/test_core/test_cg.py`  | Rewrite | 删除映射、派生量、to_atomistic 相关；保留 def/del/ select/spatial/merge 等结构性测试 |
| `docs/api/core.md`            | Modify  | 更新 Bead / CoarseGrain 描述：去掉"bidirectional conversion"措辞，明确"约定 key" |

### 4.5 Layer Validation

仍在 **core** 层。新 `core/cg.py` 仅依赖 `core.entity`；不再有 `from .atomistic import` 也不依赖 numpy。九层依赖规则收紧。

## 5. Behavior Examples

### 5.1 纯 CG（无 AA 前体）

```python
cg = CoarseGrain()
b1 = cg.def_bead(type="A", x=0.0, y=0.0, z=0.0, charge=-1.0)
b2 = cg.def_bead(type="B", x=3.0, y=0.0, z=0.0)
cg.def_cgbond(b1, b2, k=120.0)

cg.move([1, 0, 0])           # SpatialMixin works on bead["x/y/z"]
assert b1["x"] == 1.0
assert "atoms" not in b1     # 没有 AA 前体
```

### 5.2 从 Atomistic 投影（用户自己写）

MolPy **不提供**投影函数。用户按场景自己写：

```python
import numpy as np

def my_coarsegrain(ato, mask):
    """Partition + crossing-bond inference + COG position. User code."""
    cg = CoarseGrain()
    bead_of = {}
    for idx in np.unique(mask):
        atoms = tuple(a for a, m in zip(ato.atoms, mask) if m == idx)
        pos = np.mean([[a["x"], a["y"], a["z"]] for a in atoms], axis=0)
        bead_of[int(idx)] = cg.def_bead(
            atoms=atoms,
            x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
        )
    seen = set()
    for bond in ato.bonds:
        # naive O(N) lookup — user can index if needed
        ai = next(i for i, a in enumerate(ato.atoms) if a is bond.itom)
        aj = next(i for i, a in enumerate(ato.atoms) if a is bond.jtom)
        bi, bj = int(mask[ai]), int(mask[aj])
        if bi == bj: continue
        key = (bi, bj) if bi < bj else (bj, bi)
        if key in seen: continue
        seen.add(key)
        cg.def_cgbond(bead_of[bi], bead_of[bj])
    return cg
```

注：MolPy 不替用户做这个选择。换 Martini 3 (COG 含 H)、加权 COM、virtual site、shared-atom mapping、用 ITP 显式声明 bond——都是用户改这段函数的事，与 core 无关。

### 5.3 反向查询：唯一的 core 便利方法

```python
cg = my_coarsegrain(ato, mask)
beads = cg.beads_of(ato.atoms[0])     # tuple[Bead, ...]
assert len(beads) == 1                 # disjoint partition
```

如果用户的映射有 overlap：

```python
b1 = cg.def_bead(atoms=(a, b))
b2 = cg.def_bead(atoms=(b, c))
assert cg.beads_of(b) == (b1, b2)      # 两个 bead
```

### 5.4 PolymerBuilder 反向（未来 spec）

PolymerBuilder 看到的 CG 没有任何 MolPy 钦定语义，纯靠约定 key 驱动：

```python
# 未来 builder/cg.py 的伪代码（不在本 spec 范围）
def expand(cg, fragment_library):
    ato = Atomistic()
    for bead in cg.beads:
        frag = fragment_library[bead["type"]]   # 用 type 当 key
        place_fragment_at(frag, bead["x"], bead["y"], bead["z"])
        ato.merge(frag)
    return ato
```

bead 只暴露 `dict[key, value]`，builder 选哪些 key、怎么解释——自由。

## 6. Migration / Breaking Changes

| #  | Before                                                | After                                                   |
| -- | ----------------------------------------------------- | ------------------------------------------------------- |
| B1 | `Bead(atomistic=sub, type="CH3")`                     | `Bead(atoms=tuple(sub.atoms), type="CH3")`              |
| B2 | `bead.atomistic`（属性）                                | `bead.get("atoms")`（dict key）                         |
| B3 | `cg.def_bead(atomistic=sub, x=…, y=…, z=…)`           | `cg.def_bead(atoms=tuple(sub.atoms), x=…, y=…, z=…)`    |
| B4 | `cg.from_atomistic(ato, mask)`                        | 用户自己写一个函数（见 §5.2）                            |
| B5 | `cg.to_atomistic()`                                   | 用户在 builder 层按 CG 模型重建                          |
| B6 | `Bead.__init__(data_dict, atomistic=None, **attrs)`   | `Bead(**attrs)`                                         |
| B7 | bond inference 永远开启                                 | 用户在投影函数里写明                                     |
| B8 | bead 位置由 `from_atomistic` 自动写入                   | 用户在投影函数里显式 `def_bead(x=..., y=..., z=...)`     |

迁移步骤：

1. 凡 `Bead(atomistic=X)` → `Bead(atoms=tuple(X.atoms))`，或如不需要原子映射，直接 `Bead(type=…)`。
2. 凡 `bead.atomistic` 读取 → `bead.get("atoms", ())`。
3. 凡 `cg.from_atomistic(ato, mask)` → 复制 §5.2 的代码片段，或改写为自己版本。
4. 凡 `cg.to_atomistic()` → 删除调用，按需在 builder 层重建。
5. 凡 `bead.x / bead.y / bead.z` 读取 → 不变（仍走 dict）；如需从 atom 派生，用户自己写一行 numpy。

## 7. Testing Strategy

### 7.1 删除（`tests/test_core/test_cg.py`）

整组移除：

- `test_bead_with_atomistic_mapping`（第 48–55 行）
- `TestToAtomisticConversion`（第 320–358 行）
- `test_convert_atomistic_with_mask`、`test_convert_creates_multiple_beads`、`test_bead_position_is_center_of_mass`、`test_infer_cgbonds_from_atomistic_bonds`、`test_mask_length_validation`（第 364–442 行）
- `TestRoundTripConversion`（第 445–469 行）

### 7.2 新增 / 改写

保留并扩充以下测试，全部对应 Atomistic 的同名测试逻辑：

- `test_bead_is_empty_entity`：`Bead()` 可创建，无强制字段，`bead.data == {}`。
- `test_bead_dict_interface`：`Bead(type="A", x=1.0)["type"] == "A"`、`bead["x"] == 1.0`。
- `test_bead_deepcopy_copies_dict_only`：`copy.deepcopy(Bead(atoms=(a, b)))["atoms"][0] is a`（refs 共享）。
- `test_def_bead_registers_in_entities`：`cg.def_bead(...) in cg.beads`。
- `test_def_cgbond_registers_in_links`。
- `test_cgbond_endpoints`：`bond.ibead is a`、`bond.jbead is b`。
- `test_beads_of_disjoint`：`cg.beads_of(atom) == (bead,)`。
- `test_beads_of_overlap`：同一 atom 在两 bead，`len(cg.beads_of(a)) == 2`。
- `test_beads_of_unknown_returns_empty`：`cg.beads_of(unknown) == ()`。
- `test_beads_of_skips_beads_without_atoms_key`：mixed bead，部分有 `atoms` 部分没。
- `test_move_translates_bead_xyz`：与 `test_atomistic.py::test_move` 平行。
- `test_merge`、`test_select`、`test_replicate`：与 Atomistic 平行。

### 7.3 不新增的测试模块

不创建 `tests/test_core/test_ops_cg.py` 或 `tests/test_op/test_cg.py`——没有对应实现。

### 7.4 覆盖率目标

| Module                    | 覆盖率目标 |
| ------------------------- | ---------- |
| `src/molpy/core/cg.py`    | ≥ 95%      |

### 7.5 全量回归

- `pytest tests/ -m "not external" -v`：无回归。
- `grep -rn "bead\.atomistic\|to_atomistic\|from_atomistic" src/ tests/ docs/`：除 CHANGELOG 外应为空。

## 8. Performance Considerations

新 `core/cg.py` 不含任何重型算法，性能特征即 `Struct` / `Entity` 自身：

| 操作              | 复杂度                   |
| ----------------- | ------------------------ |
| `def_bead`        | O(1)                     |
| `def_cgbond`      | O(1)                     |
| `beads_of(atom)`  | O(N_beads × ⟨\|atoms\|⟩) |
| `merge` / `copy`  | 同 Atomistic              |

`beads_of` 不缓存：缓存失效逻辑会污染 `def_bead` / `add_bead` / `remove_entity` / `merge` 多处，工程债不划算。需要快查的用户在自己代码里建 `dict[id(atom), list[Bead]]`。

旧 `from_atomistic` 中的 O(N×M) bond 推断问题随该方法删除而消失；用户写新版时若用 `dict` 存 atom→idx 映射即可达到 O(N+M)（见 §5.2 注释）。

## 9. Open Questions

| #  | Question                                                              | 当前默认                                                  |
| -- | --------------------------------------------------------------------- | --------------------------------------------------------- |
| Q1 | `bead["atoms"]` 这个约定 key 名要不要写进 `core/cg.py` 的常量？        | 不写。docstring 描述即可；硬编码在 `beads_of` 一处。      |
| Q2 | 用户写 `my_coarsegrain` 模板函数的反复劳动会不会成为负担？             | 暂不解决。等出现 ≥3 处重复后起新 spec 加 op helper。       |
| Q3 | 反向 builder（CG → AA via fragment library）何时立项？                  | 独立 spec，依赖 fragment library 设计先行。               |
| Q4 | CG angles / dihedrals 是否进 link 体系？                                | 本 spec 不涉及；后续视 CG 力场需求评估。                  |
| ~~Q5~~ | ~~`CoarseGrain.to_frame()` I/O 入口何时补？~~ | **已实现**：与 `Atomistic.to_frame` 对称的 `beads` / `cgbonds` 双 block 形式；endpoints 索引字段命名为 `ibead` / `jbead`。 |

## 10. Critical Files

| 路径                                  | 作用                              |
| ------------------------------------- | --------------------------------- |
| `src/molpy/core/cg.py`                | 整个模块重写（约 150 行）          |
| `tests/test_core/test_cg.py`          | 测试删除 + 改写（向 Atomistic 对齐） |
| `docs/api/core.md`                    | 描述更新                          |
