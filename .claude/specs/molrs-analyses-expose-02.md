---
slug: molrs-analyses-expose-02
status: code-complete
created: 2026-05-28
revised: 2026-05-28
---

# molrs-analyses-expose-02 — 暴露剩余 molrs.compute 分析算子

## Summary

`molrs-backend` 的 Phase 5 已把 MSD / Cluster / ClusterCenters / CenterOfMass /
GyrationTensor / InertiaTensor / RadiusOfGyration / Pca / KMeans 暴露为
`molpy.compute` 算子。molrs 还有一批已实现、但 molpy 尚未暴露的分析。本 spec
延续同一「**继承优先 / 直接复用 molrs 类型**」原则，把它们补齐为 molpy 风格的
薄壳 `Compute` 子类：

- **键取向序参量** — `order.Steinhardt`、`Hexatic`、`Nematic`、`SolidLiquid`
- **密度场** — `density.LocalDensity`、`density.GaussianDensity`
- **衍射** — `diffraction.StaticStructureFactorDebye`（静态结构因子 S(k)）
- **局部环境** — `environment.BondOrder`
- **势平均力** — `pmft.PMFTXY`
- **团簇属性** — `cluster.ClusterProperties`

每个算子是 5–15 行 `Compute` 子类，仅做参数转发 + molpy 风格 callable 接口，
**直接返回 molrs 的原生结果**（tuple / ndarray / dict / `*Result`，视各算子而定），
不另起 `molpy.*Result`。无新物理；molpy 侧只做 API 适配。

## Domain basis

均为标准凝聚态 / 软物质轨迹分析，物理实现位于 `molrs-compute`，molpy 仅适配：

- **Steinhardt $q_\ell$ / $w_\ell$** — 键取向序参量；Steinhardt, Nelson & Ronchetti,
  *Phys. Rev. B* **28**, 784 (1983)。
- **Hexatic $\psi_k$** — 二维 k 重键取向序；Nelson & Halperin, *Phys. Rev. B*
  **19**, 2457 (1979)。
- **Nematic Q 张量 / director** — 液晶取向序；de Gennes & Prost,
  *The Physics of Liquid Crystals*, 2nd ed. (1993)。
- **Solid-liquid 判别** — 基于 $q_\ell$ 键关联的固液分类；ten Wolde, Ruiz-Montero
  & Frenkel, *J. Chem. Phys.* **104**, 9932 (1996)。
- **Static structure factor (Debye)** — $S(k)=\frac{1}{N}\big\langle\sum_{i,j}
  \frac{\sin(kr_{ij})}{kr_{ij}}\big\rangle$；Debye 散射方程。
- **PMFT** — potential of mean force and torque；van Anders et al.,
  *ACS Nano* **8**, 931 (2014)。

molpy 不重新推导任何公式；正确性由 parity 测试间接保证（molrs 自身已有物理测试）。

## Design

### 调用约定：复用 RDF 的双输入 `__call__` 先例

多数算子需要邻居列表（`compute(frames, nlists)`），与已有 `compute.RDF`
（`__call__(self, frames, neighbors)`）完全同构。沿用 RDF 的写法：`_compute`
抛 `NotImplementedError`，真正实现放在 `__call__`，按 molrs 原生签名转发。

各算子的 molrs 原生签名（构造 + 调用）：

| molpy 算子 | molrs 构造 | molrs 调用 |
|---|---|---|
| `Steinhardt` | `(l, average=False, wl=False, wl_normalize=False)` | `(frames, nlists)` |
| `Hexatic` | `(k)` | `(frames, nlists)` |
| `Nematic` | `()` | `(frames, directors)` |
| `SolidLiquid` | `(l, q_threshold=0.7, n_threshold=6)` | `(frames, nlists)` |
| `LocalDensity` | `(r_max, diameter=0.0)` | `(frames, nlists)` |
| `GaussianDensity` | `(nx, ny, nz, sigma)` | `(frames)` |
| `StaticStructureFactorDebye` | `(k_values)` | `(frames)` |
| `BondOrder` | `(n_theta, n_phi)` | `(frames, nlists)` |
| `PMFTXY` | `(x_max, y_max, n_x, n_y)` | `(frames, nlists, orientations=None)` |
| `ClusterProperties` | `()` | `(frames, clusters)` |

### 薄壳示例（与 Phase 5 / RDF 一致）

```python
# src/molpy/compute/order.py
import molrs
from .base import Compute

class Steinhardt(Compute):
    def __init__(self, l, average=False, wl=False, wl_normalize=False):
        super().__init__(l=l, average=average, wl=wl, wl_normalize=wl_normalize)
        self._inner = molrs.compute.order.Steinhardt(l, average, wl, wl_normalize)

    def __call__(self, frames, nlists):
        return self._inner.compute(frames, nlists)

    def _compute(self, input):  # pragma: no cover — two-input op, use __call__
        raise NotImplementedError("Steinhardt takes (frames, nlists)")
```

### 内存拷贝纪律 / 不可变性

延续 `molrs-backend`：边界零额外拷贝（禁止 `.copy()` / `copy=True`，由验收 grep
强制），算子不修改输入 frame（不写回 block、不挂属性）。`frames` 已经
IS-A `molrs.Frame`，`nlists` 已是 molrs `NeighborList`，无转换层。

## Files

| 文件 | 作用 |
|---|---|
| `src/molpy/compute/order.py` | `Steinhardt` + `Hexatic` + `Nematic` + `SolidLiquid` |
| `src/molpy/compute/density.py` | `LocalDensity` + `GaussianDensity` |
| `src/molpy/compute/diffraction.py` | `StaticStructureFactorDebye` |
| `src/molpy/compute/environment.py` | `BondOrder` |
| `src/molpy/compute/pmft.py` | `PMFTXY` |
| `src/molpy/compute/cluster.py` | 追加 `ClusterProperties`（已有 `Cluster`/`ClusterCenters`） |
| `src/molpy/compute/__init__.py` | 导入 + `__all__` 追加 10 个名字 |
| `tests/test_compute/test_order.py` | 4 个序参量：smoke + parity + immutability |
| `tests/test_compute/test_density.py` | `LocalDensity` / `GaussianDensity`：smoke + parity + immutability |
| `tests/test_compute/test_diffraction.py` | `StaticStructureFactorDebye`：smoke + parity |
| `tests/test_compute/test_environment.py` | `BondOrder`：smoke + parity |
| `tests/test_compute/test_pmft.py` | `PMFTXY`：smoke + parity |
| `tests/test_compute/test_cluster.py` | 追加 `ClusterProperties` parity |
| `docs/developer/molrs-backend.md` | 分析目录表追加新算子 |

## Tasks

> 每模块 RED-before-GREEN：先写失败的 smoke/parity 测试，再实现薄壳，确认绿。
> 所有算子复用 RDF 的双输入 `__call__` 先例。

**Phase 1 — order（键取向序参量）**

- [x] Write failing `test_order.py`：4 个算子各一个 smoke + `test_*_parity_with_molrs_direct` + `test_order_input_frame_immutable`。（含 `parity_helpers.py` 共享工具：`random_periodic_frame` / `assert_nested_equal` / `frame_coords_snapshot`。）
- [x] Implement `compute/order.py`：`Steinhardt`（`l` 为 Sequence）/ `Hexatic` / `Nematic`（`(frames, directors)`，directors 为 (N,3)）/ `SolidLiquid` 薄壳；7 passed。

**Phase 2 — density（密度场）**

- [x] Write failing `test_density.py`：`LocalDensity`（`(frames, nlists)`）+ `GaussianDensity`（`(frames)`）的 smoke + parity + immutability。
- [x] Implement `compute/density.py`：两个薄壳；5 passed。

**Phase 3 — diffraction / environment / pmft**

- [x] Write failing `test_diffraction.py`：`StaticStructureFactorDebye(k_values)`，`(frames)` 调用，理想气体 S(k)→1 大 k 极限物理 sanity + parity + immutability。
- [x] Implement `compute/diffraction.py`；测试转绿。
- [x] Write failing `test_environment.py`：`BondOrder(n_theta, n_phi)`，`(frames, nlists)`，smoke + parity + immutability。
- [x] Implement `compute/environment.py`；测试转绿。
- [x] Write failing `test_pmft.py`：`PMFTXY(x_max, y_max, n_x, n_y)`，`(frames, nlists, orientations=None)`，smoke + parity + immutability。
- [x] Implement `compute/pmft.py`；3 文件共 12 passed（含 S(k) sanity）。

**Phase 4 — cluster properties**

- [x] Write failing `test_cluster.py`：先用 `Cluster` 得 clusters，再 `ClusterProperties()(frame, [clusters])`；smoke + parity + immutability。（clusters 需包成 Sequence。）
- [x] Implement `compute/cluster.py::ClusterProperties` 薄壳；4 passed。

**Phase 5 — 整合与文档**

- [x] `compute/__init__.py`：导入并把 10 个名字加入 `__all__`；import 全部通过。
- [x] `docs/developer/molrs-backend.md` 分析目录表追加新算子（按调用约定分组 + 最小调用示例：Steinhardt / S(k)）。
- [x] Run `pytest tests/ -m "not external"`（1897 passed / 1 xfailed）、`ty check`、`ruff`；都绿。Hygiene cleanup（`/mol:simplify`）：2 处 dead `_compute` forwarding 改为 raise（与兄弟模块一致）；1 处 `_inner`/`_impl` 跨文件命名漂移交 `/mol:note`+`/mol:refactor`。

## Testing strategy

- **Unit / smoke** — 每个算子实例化 + 在小随机周期体系上跑通，断言输出形状/类型。
- **Parity** — 每个算子一个 `test_*_parity_with_molrs_direct`：相同输入下 molpy 包装
  与直调 `molrs.compute.<sub>.<X>` 逐元素一致（浮点 1e-12；含随机性的算子用
  seed-pin 或确定性输入）。
- **Scientific sanity** — `StaticStructureFactorDebye` 理想气体大 k 极限 S(k)→1；
  其余依赖 parity 间接保证（molrs 自带物理测试）。
- **Immutability** — 每个算子 `test_input_frame_immutable`：快照 frame 坐标，
  `__call__` 后不变。
- **零拷贝纪律** — `! grep -rE '\.copy\(\)|copy=True'`（沿用 molrs-backend 的
  config-dict 例外）覆盖新文件。

`pytest tests/test_compute/ --cov=src/molpy/compute -v` 维持 ≥ 80%。

## Out of scope

- 类型间 / 多组分 RDF（`g_AB(r)`）—— 单列入 follow-up（见 molrs-backend Out of scope）。
- 为这些算子设计 molpy 侧的 director / orientation 提取 helper（`Nematic` 的
  `directors`、`PMFTXY` 的 `orientations` 由调用方提供；自动推导留作 follow-up）。
- 把结果包成 molpy 富类型（`molpy.*Result`）—— 本 spec 坚持直接返回 molrs 原生结果。
- molrs 端任何新算法实现 —— 本 spec 只暴露已实现的。
