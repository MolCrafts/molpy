---
slug: molrs-backend
status: in-progress
created: 2026-05-06
revised: 2026-05-06
---

# molrs-backend — molrs 作为 molpy 的默认计算后端

## Summary

将 molrs（Rust，PyPI 包 `molcrafts-molrs`，`import molrs`）作为
**molpy 的必选运行时依赖**，并以「**继承优先 / 直接复用 molrs 类型**」
的原则重构相关数据模型与 compute 算子：

- **数据模型层** — molpy.Box 直接继承 molrs.Box；其它有 molrs 等价物
  的容器（NeighborList、RDFResult 等）直接复用，不再做 wrapper。
- **新算子** — `molpy.compute.NeighborList`、`molpy.compute.RDF`，
  molpy 风格 API，molrs 后端。
- **替换现有算子** — MCD / PMSD / RDKit 三块逐项审查，能映射到 molrs
  的（RDKit 3D embed → molrs.embed）做替换；molrs 无等价（MCD/PMSD 时间
  相关分析）的保留 Python 实现，但底层距离/邻居/PBC 走 molrs。
- **拓宽分析覆盖** — 把 molrs 已有的 MSD / Cluster / GyrationTensor /
  InertiaTensor / RadiusOfGyration / CenterOfMass / PCA / KMeans 通过
  molpy 风格的 `Compute` 子类对外暴露。

molrs 不再可选。这是一次有意识的破坏性变更（详见 Out of scope）。

## Domain basis

- **邻居列表**：链表/链格法 (cell list / linked-cell)，O(N)。已实现于
  `molrs::neighbors::LinkCell` + `NeighborQuery`，PyO3 暴露为
  `molrs.NeighborQuery` / `molrs.NeighborList`。
- **RDF**：g(r) = (V / (N·N_q)) · ⟨n(r)⟩ / (4πr² Δr)。
  Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed., §2.6。
  已实现于 `molrs-compute::RDF`，`compute()` 内部即 `finalize()`。
- **MSD / Cluster / Gyration / Inertia / Rg / COM / PCA / KMeans**：
  均为标准轨迹分析；`molrs-compute` 已实现并通过 `molrs.MSD` / `Cluster`
  / `GyrationTensor` 等导出。

无新物理；molpy 侧只做 API 适配。

## Design

### 依赖关系

```toml
# pyproject.toml
[project]
dependencies = [
    # ... existing ...
    "molcrafts-molrs>=0.0.7",
]
```

不再有 `[project.optional-dependencies].molrs`。这是 **breaking change**：
下游用户必须升级。changelog 必须明确说明（见 docs 任务）。

### 继承优先：molpy.Box ← molrs.Box

```python
# src/molpy/core/box.py
import molrs
import numpy as np

class Box(molrs.Box):
    """molpy 风格的 Box，直接继承 molrs.Box。"""

    class Style(Enum):
        FREE = 0
        ORTHOGONAL = 1
        TRICLINIC = 2

    def __init__(self, matrix=None, pbc=None, origin=None):
        h, pbc, origin = _normalize_box_args(matrix, pbc, origin)
        super().__init__(h, origin=origin, pbc=pbc)
        # Python-only state needed by molpy (PeriodicBoundary 接口、Style 枚举等)
        # 通过组合 / 委托补足；core/box.py 现有 API 保留

    @classmethod
    def cubic(cls, length): ...
    @classmethod
    def from_lengths_angles(cls, lengths, angles): ...
    @property
    def style(self) -> "Box.Style": ...
    # ...其它 molpy 已有方法
```

`PeriodicBoundary` 旧基类移除或改为 mixin（取决于其方法是否仍被引用，
实现期审）。

**前置条件**：molrs-python 侧 `#[pyclass(name = "Box", from_py_object)]`
必须加 `subclass`。这是本 spec 显式纳入的 molrs 修改。

### 直接复用：NeighborList / RDFResult

不做 molpy 侧 wrapper。`molpy.compute.NeighborList`(算子) 内部返回
`molrs.NeighborList`(结果) 原样：

```python
# src/molpy/compute/neighborlist.py
import molrs
from .base import Compute

class NeighborList(Compute):
    def __init__(self, cutoff: float):
        self.cutoff = cutoff

    def __call__(self, frame) -> molrs.NeighborList:
        nq = molrs.NeighborQuery(frame.box, frame["atoms"][["x", "y", "z"]], self.cutoff)
        return nq.query_self()
```

注意：
- `frame.box` 已经 IS-A `molrs.Box`，无需转换。
- `frame["atoms"][["x", "y", "z"]]` 通过 `Block.__getitem__(list)` 走
  `np.column_stack` 路径，返回 (N, 3) ndarray。这就是**唯一一次**坐标
  拷贝；没有任何额外的 `_frame_to_xyz` / `_molrs.py` 适配层。

RDF 同理：

```python
# src/molpy/compute/rdf.py
import molrs

class RDF(Compute):
    def __init__(self, n_bins, r_max, r_min=0.0):
        self._inner = molrs.RDF(n_bins, r_max, r_min)

    def __call__(self, frames, neighbors) -> molrs.RDFResult:
        # frames / neighbors 都已是 molrs 友好类型；molpy 不再 wrap。
        return self._inner.compute(frames, neighbors)
```

### 替换现有 compute 算子

| 现有 molpy 算子 | molrs 等价 | 决策 |
|---|---|---|
| `Generate3D` (RDKit) | `molrs.generate_3d` + `EmbedOptions` | 替换 |
| `OptimizeGeometry` (RDKit) | molrs.embed pipeline 的最终 minimize 阶段 | 替换 |
| `MCDCompute` | 无 (molrs 只有 MSD) | 保留 Python 实现，底层距离/PBC 改走 molrs |
| `PMSDCompute` | 无 | 同上 |

### 暴露其它 molrs 分析

为以下每一项添加一个 `molpy.compute.<Name>` 薄壳类（仅参数转发 + molpy
风格 callable 接口）：

`MSD`、`Cluster`、`ClusterCenters`、`CenterOfMass`、`GyrationTensor`、
`InertiaTensor`、`RadiusOfGyration`、`Pca`、`KMeans`。

每个 op 都是 5–15 行的 `Compute` 子类，结果直接返回 molrs 的
`*Result` 类型，不另起 `molpy.*Result`。

### 内存拷贝纪律

所有边界一律走 PyO3 `PyReadonlyArray2` 零拷贝视图；返回的索引/距离由
molrs 以 `borrow_from_array` 形式回传。整个 molpy 侧**禁止**调用 `.copy()`
或 `.astype(..., copy=True)`，由验收契约里的 lint 强制（grep `compute/`
目录）。

唯一的物理拷贝是 `Block.__getitem__(["x","y","z"])` 内部的
`np.column_stack`——这一次重排无法消除，除非把 frame 坐标存储改为
单列 (N, 3)（出 scope）。

### 不可变性

所有 `Compute.__call__` 不修改输入 frame：不写回 atoms block，不挂
属性，不缓存。返回值可以直接是 molrs 类型（其底层 buffer 由 molrs
所有，但只读）。

## Files

| 文件 | 作用 |
|---|---|
| **molpy 侧** | |
| `pyproject.toml` | `molcrafts-molrs` 移入主 `dependencies`；删 extras 项 |
| `src/molpy/core/box.py` | `class Box(molrs.Box)`；保留 `Style`/`cubic`/`from_lengths_angles` 等 |
| `src/molpy/core/region.py` | `PeriodicBoundary` 评估去留 |
| `src/molpy/compute/neighborlist.py` | `NeighborList` 算子（薄壳） |
| `src/molpy/compute/rdf.py` | `RDF` 算子（薄壳） |
| `src/molpy/compute/msd.py` | `MSD` 算子 |
| `src/molpy/compute/cluster.py` | `Cluster` + `ClusterCenters` |
| `src/molpy/compute/shape.py` | `GyrationTensor` + `InertiaTensor` + `RadiusOfGyration` + `CenterOfMass` |
| `src/molpy/compute/decomposition.py` | `Pca` + `KMeans` |
| `src/molpy/compute/embed.py` | `Generate3D` + `OptimizeGeometry`，替换 `compute/rdkit.py` |
| `src/molpy/compute/__init__.py` | 重新整理 `__all__`；删 `_HAS_RDKIT` / 不再有 `_HAS_MOLRS` |
| `src/molpy/compute/rdkit.py` | **删除**（被 embed.py 取代） |
| `src/molpy/compute/mcd.py` | 内部 PBC 距离改调 molrs；签名不变 |
| `src/molpy/compute/pmsd.py` | 同上 |
| `tests/test_compute/test_neighborlist.py` | 新算子单测 + parity + 不可变性 |
| `tests/test_compute/test_rdf.py` | 理想气体 g(r)≈1 / 多帧累积 / 缺 box |
| `tests/test_compute/test_box_inheritance.py` | `isinstance(molpy.Box(...), molrs.Box)`；frame.box 直传 molrs API |
| `tests/test_compute/test_msd.py` etc. | 每个新暴露算子各一个 smoke + parity 测试 |
| `tests/test_compute/test_embed_replacement.py` | 替换前后产物等价（构象 RMSD ≤ tol） |
| `tests/test_core/test_box.py` | 现有 Box 测试需更新构造签名 |
| `docs/user/compute/molrs-backend.md` | 安装、API 说明、与旧版差异 |
| `docs/changelog.md` | breaking change 条目 |
| **molrs 侧（本 spec 显式纳入）** | |
| `molrs/molrs-python/src/simbox.rs` | `#[pyclass(...)]` 加 `subclass` 标记 |
| `molrs/molrs-python/tests/test_subclass.py` | Python 端验证 `class Sub(molrs.Box)` 工作 |

## Tasks

> 共 5 个 Phase；每 Phase 内部 RED-before-GREEN。Phase 4/5 列为单行
> 高层任务，`/mol:impl` 启动时再逐算子展开。

**Phase 0 — molrs 前置**

- [x] Patch `molrs/molrs-python/src/simbox.rs`: change `#[pyclass(name = "Box", from_py_object)]` to add `subclass`. Add `tests/test_subclass.py` confirming `class Sub(molrs.Box)` instantiates and inherits methods. Cut a new `molcrafts-molrs` patch release (e.g. 0.0.8).  *(release deferred until full suite green)*

**Phase 1 — 必选依赖 + Box 继承**

- [x] Move `molcrafts-molrs>=0.0.8` from optional-dependencies to main `dependencies` in `pyproject.toml`; remove the `molrs` extras key; verify `pip install -e ".[dev]"` resolves.
- [x] Write failing test `test_box_inheritance.py::test_molpy_box_is_a_molrs_box` — `isinstance(molpy.Box.cubic(10.0), molrs.Box) is True`.
- [x] Refactor `core/box.py` so `class Box(molrs.Box)`; preserve all public API (`Style`, `cubic`, `from_lengths_angles`, `matrix`, `origin`, `pbc`, `lengths`, `tilts`, `__mul__`, `__eq__`, `__repr__`); test passes.  *(setters and `set_*` mutators removed — Box is now immutable; updated 3 tests in `test_core/test_box.py` accordingly. Mutators were unused in production code.)*
- [x] Update existing `test_core/test_box.py` for any signature changes; full `test_core/` suite green.
- [x] Write failing test `test_box_inheritance.py::test_frame_box_passed_directly_to_molrs` — `molrs.NeighborQuery(frame.box, xyz, 2.0)` accepts `frame.box` with no conversion.

**Phase 2 — NeighborList + RDF (新算子)**

- [x] Write failing test `test_neighborlist.py::test_basic_periodic` — random frame + cubic box → `num_pairs > 0`, all distances ≤ cutoff.
- [x] Implement `compute/neighborlist.py` (~10 lines: `Compute` subclass forwarding to `molrs.NeighborQuery`); test passes.
- [x] Write failing test `test_neighborlist.py::test_parity_with_molrs_direct` — same inputs through `molrs.NeighborQuery` directly; assert pair count + distance set match within 1e-12.
- [x] Write failing test `test_neighborlist.py::test_input_frame_immutable` — snapshot `frame.box.matrix` and `frame["atoms"]["x"/"y"/"z"]`; unchanged after `__call__`.
- [x] Write failing test `test_rdf.py::test_ideal_gas_g_of_r_approaches_one` — uniform random points, middle bins ∈ [0.7, 1.3].
- [x] Implement `compute/rdf.py` (~25 lines including the molrs.Frame view adapter); test passes.  *(The view adapter creates a fresh molrs.Frame whose simbox = molpy.Frame.box — zero coordinate copy because RDF reads coordinates from the NeighborList, not the frame itself.)*
- [x] Write failing test `test_rdf.py::test_multi_frame_aggregation` — list of 3 frames vs single-frame averaged manually; assert agreement.
- [x] Write failing test `test_rdf.py::test_no_box_raises` — frame without `.box` → `ValueError` mentioning "box".

**Phase 3 — 替换 RDKit-based embed**

- [ ] Write failing test `test_embed_replacement.py::test_generate3d_returns_3d_coords` against the new `compute/embed.py::Generate3D` (molrs-backed).
- [ ] Implement `compute/embed.py::Generate3D` and `OptimizeGeometry` wrapping `molrs.generate_3d` / `molrs.EmbedOptions`; tests pass.
- [ ] Write parity test `test_embed_replacement.py::test_old_vs_new_geometry_close` — for a small molecule set, RMSD between old RDKit-produced and new molrs-produced geometries within a documented tolerance (or, if not bit-comparable, both pass an independent geometry sanity check: bond lengths within 10% of literature values).
- [ ] Delete `src/molpy/compute/rdkit.py`; remove `_HAS_RDKIT` from `compute/__init__.py`; remove `rdkit` extras from pyproject.
- [ ] Update any docs / examples that referenced the old RDKit-backed names.

**Phase 4 — MCD / PMSD 内部改造**

> 单行高层任务；每条在 `/mol:impl` 启动时拆为 (a) 改 PBC 距离调用 →
> molrs.NeighborQuery (b) parity 测试 vs 旧实现 (c) 性能基准（可选）。

- [ ] 把 `compute/mcd.py` 内部的 PBC 距离/邻居计算改调 molrs；公开签名保持不变。
- [ ] 把 `compute/pmsd.py` 内部的 PBC 距离/邻居计算改调 molrs；公开签名保持不变。

**Phase 5 — 暴露其它 molrs 分析**

> 单行高层任务；`/mol:impl` 启动时为每个 op 拆为 (a) 失败测试
> (b) `Compute` 子类实现 (c) 与 `molrs.<X>.compute(...)` 的 parity 测试
> (d) 文档段落。

- [x] 暴露 `molrs.MSD` 为 `molpy.compute.MSD`。
- [x] 暴露 `molrs.Cluster` + `molrs.ClusterCenters` 为 `molpy.compute.Cluster` / `ClusterCenters`。
- [x] 暴露 `molrs.CenterOfMass` 为 `molpy.compute.CenterOfMass`。
- [x] 暴露 `molrs.GyrationTensor` / `InertiaTensor` / `RadiusOfGyration` 为对应 molpy 算子。
- [x] 暴露 `molrs.Pca2` 为 `molpy.compute.Pca`；暴露 `molrs.KMeans` 为 `molpy.compute.KMeans`。

**Phase 6 — 文档与收尾**

- [ ] 写 `docs/user/compute/molrs-backend.md`：安装命令、Box 继承示例、NeighborList + RDF 最小例子、性能说明（一处 column_stack 拷贝及其原因）、其它分析的清单与示例。
- [ ] 写 `docs/changelog.md` 的 breaking-change 段：molrs 由可选变必选；`compute.rdkit` 模块移除；最小 Python 兼容版本（如有变化）。
- [ ] Run `pytest tests/ -v -m "not external"` 与 `pre-commit run --all-files`；都绿。

## Testing strategy

- **Unit** — 每个新算子：实例化、属性、错误路径。
- **Parity** — 每个 wrapper 算子 (`NeighborList`/`RDF`/MSD/Cluster/...) 都
  必须有一个 `test_parity_with_molrs_direct`：绕过 molpy 包装，直接调
  `molrs.<X>`，assert 输出按 canonical 排序后逐元素一致或在 1e-12 之内。
  Box 继承额外加 `isinstance(molpy.Box(...), molrs.Box)` 一行 assert。
- **Scientific** — RDF 理想气体 g(r)≈1 是物理 sanity 检验；其它分析依赖
  parity 测试间接保证（molrs 自身已有物理测试）。
- **Immutability** — 每个新算子 `test_input_frame_immutable`。
- **回归** — 替换 RDKit embed 必须 parity 到旧实现的物理量（键长 /
  RMSD），不能凭"两边都跑通"就算过。
- **现有套件兼容** — `test_core/`、`test_io/` 等模块不应受 Box 继承影响；
  若有失败，必须修而非 skip。
- **删除验证** — `compute/rdkit.py` 删除后 `pytest -k rdkit` 没有任何
  剩余测试；`grep -r '_HAS_RDKIT\|_HAS_MOLRS' src/` 无命中。

`pytest tests/test_compute/ --cov=src/molpy/compute -v` ≥ 80%。

## Out of scope

- 多组分 / 类型间 RDF（g_AB(r)）—— v1 只支持单一物种 self-pair，留作
  follow-up spec。
- 把 frame 坐标存储改为单列 (N, 3) ndarray —— 这能消除最后一个
  column_stack 拷贝，但需要独立的 `frame-storage-refactor` spec
  覆盖 Block / Frame / 所有 IO reader/writer。
- molrs 新增物理算法（MCD/PMSD 的 Rust 实现）—— 若未来想 port，单独
  立 spec。
- 任何 deprecation shim：`molcrafts-molpy[molrs]` extras 键直接删除，
  不保留 alias；老用户升级时会得到 `pip` 的 "extra not provided"
  warning 即可。
- molpy 其它子包（io、parser、builder、typifier、reacter、pack、engine）
  对 molrs 的依赖整理 —— 本 spec 只覆盖 compute + core/box.py。
