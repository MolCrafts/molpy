---
title: Crosslink 1/3 — Crosslinker(ABC) + DeterministicCrosslinker（消费 molrs 引擎）
status: done
created: 2026-07-05
depends_on: "molrs: reaction-smarts-01-python-matcher, reaction-smarts-02-smirks-applier"
---

# Crosslink deterministic — 消费 molrs 的 SMARTS/SMIRKS 引擎

> molpy 交联层**只做交联编排**——匹配、反应应用、图编辑全在 molrs（Rust）。本 spec 落地
> `Crosslinker(ABC)` 基类（`apply(graph)→新 graph`，immutable，仿 `VirtualSiteBuilder`）+
> `DeterministicCrosslinker`（穷尽 / `spacing` 均匀 / 显式 `pairs`，无 RNG）。SMARTS 匹配用
> `molrs.SmartsPattern`、SMIRKS 应用用 `molrs.Reaction`、距离用 `molrs.NeighborList`、
> 主链排序用 `molrs.Atomistic.topo_distances`。**molpy 侧无 SMARTS 引擎、无自造 MatchSet/Block。**
> 架构见 `.claude/notes/crosslinking-syntax-design.md` (v6)。

## Summary

`Crosslinker(ABC)` 是 house-style 结构变换类（对照 `VirtualSiteBuilder`）：构造器收一条 **Daylight
reaction SMARTS** 字符串 + `cutoff`，一个 `apply(graph) -> MolGraph` 动词——**copy 一次工作副本、
经 molrs 匹配 LHS 组分、循环 `select` 配对、经 `molrs.Reaction.apply` 就地改图、返回新图**，入参不动。
唯一抽象 hook 是 `select`（配对策略 = mode 差异）；无 `@classmethod`。

`DeterministicCrosslinker(Crosslinker)` 实现 `select` 为**穷尽/确定**，三种确定性选点：穷尽（默认 100%）
/ 规则部分（`spacing=K`，沿链每 K 个匹配位点取 1 → 均匀交联点）/ 显式（`pairs`）。**无随机、无 seed、
无转化率提前停**——"100% 就是确定性、可跳步"。

## Domain basis

### molrs 提供引擎，molpy 只编排（依赖 molrs 链 reaction-smarts-01/02）

| 交联步骤 | molpy 侧 | molrs 侧（Rust，本 spec 消费） |
|----------|----------|-------------------------------|
| 反应语言 | 传一条 reaction SMARTS 字符串 | `molrs.Reaction(smirks)` 解析 `LHS>>RHS`、编译映射号 diff |
| 找位点 | 循环调用 | `reaction.reactant_patterns[i].find_matches_mapped(mol)` → `list[dict[map:handle]]` |
| 成键原子对 | 距离判据用 | `reaction.forming_bonds` → `[(map_a,map_b)]` |
| 改图 | 循环调用 | `reaction.apply(mol, binding)` 就地成键/删原子/加原子/刷拓扑 |
| 距离 | 候选过滤 | `molrs.NeighborList` / PBC |
| 主链排序 | `spacing` 用 | `mol.topo_distances(source)` BFS |
| copy | immutable | `mol.copy()`（molrs Clone，reaction-smarts-01） |

**molpy 侧不写 SMARTS 引擎、不写 SMIRKS 应用、不建 MatchSet/Block**——匹配结果就是 molrs 返回的
`list[dict[map:handle]]`，molpy 直接在其上配对。

### 配对是拓扑 + 组合，几何仅作过滤（xyz 可选）

- **给 `cutoff` 且有坐标**：只在成键原子对（`reaction.forming_bonds`）距离 ≤ `cutoff` 时配对，用
  `molrs.NeighborList`/PBC。这是"先 packing / 手动摆网络 → 按邻近交联"的工作流。
- **无 `cutoff`（或无坐标）**：拓扑配对（跨组分组合，受"每位点一次" + 排除约束）。xyz 非必选。
- **`cutoff` 给了但无坐标** → fail-fast `ValueError`。

### 均匀交联点：拓扑规则 vs 几何规则

- **拓扑规则**（交联点沿链均匀）：`spacing=K` 用 `topo_distances` 排序、每 K 个取 1。纯拓扑、不用坐标。
- **几何规则**（空间整齐 mesh）：交联器不排列链。要么先用 builder/placer 排成规则阵列再按 `cutoff` 连，
  要么用 `pairs=`（晶格索引）定规则拓扑 + 后置 `molpy.optimize` 松弛。见 03-builder。

## Design

### 1. `Crosslinker(ABC)`（house-style，仿 VirtualSiteBuilder）

```python
import molrs

class Crosslinker(ABC):
    """离线交联器。apply(graph) 返回同子类新图；入参不改。引擎在 molrs。"""
    def __init__(self, reaction: str, *, cutoff: float | None = None):
        self._reaction = molrs.Reaction(reaction)          # molrs 解析 + 编译
        self._cutoff = cutoff

    def apply(self, graph: MolGraph) -> MolGraph:
        work = graph.copy()                                # immutable：molrs Clone
        occ = [p.find_matches_mapped(work)                 # 每个 LHS 组分 → list[{map:handle}]
               for p in self._reaction.reactant_patterns]
        cands = self._candidate_pairs(work, occ)           # 跨组分 + cutoff（molrs 邻居表）
        for binding in self.select(work, cands):           # ← 子类 hook（mode）
            self._reaction.apply(work, binding)            # molrs 就地改图
        return work

    @abstractmethod
    def select(self, graph: MolGraph, candidates: list["Candidate"]) -> Iterator[dict[int, int]]:
        """产出 {映射号: handle} 绑定。mode 唯一差异；cutoff 已在候选里过滤。"""
```

基类共享 helper：

| helper | 作用 |
|--------|------|
| `_candidate_pairs(graph, occ)` | 跨组分枚举 occurrence 对；成键原子取 `reaction.forming_bonds`；给 `cutoff`+坐标则 `molrs.NeighborList` 只留 ≤cutoff，否则全组合 |
| `_binding(candidate)` | 合并两 occurrence 的 `{映射号:handle}` |
| `_same_molecule(a, b)` | 连通分量/`mol_id`（拓扑，不看坐标） |
| `_regular_sites(graph, occ, k)` | 每条链按 `mol.topo_distances` 主链排序、stride K → 均匀交联点子集（线性精确，支链 best-effort） |
| `_distance(graph, a, b)` | molrs PBC 距离；`cutoff` 无坐标 → `ValueError` |

`Candidate` = frozen dataclass `(occ_a, occ_b, comp_a, comp_b, distance)`。

### 2. `DeterministicCrosslinker(Crosslinker)`

```python
class DeterministicCrosslinker(Crosslinker):
    """确定性交联，无 RNG。穷尽 / spacing 规则部分 / 显式 pairs。"""
    def __init__(self, reaction, *, cutoff=None, spacing: int | None = None,
                 pairs=None, exclude_same_molecule=False, exclude_same_match=False):
        super().__init__(reaction, cutoff=cutoff)
        self._spacing, self._pairs = spacing, pairs
        self._exclude_same_molecule = exclude_same_molecule
        self._exclude_same_match = exclude_same_match

    def select(self, graph, candidates):
        # 0. spacing：先把候选限制到 _regular_sites 的规则子集（均匀交联点）
        # 1. pairs 显式：只产出这些（校验合规）
        # 2. 否则：candidates 确定性排序（(distance, occ_a, occ_b)），依次取、跳过已消耗/同分子/自配对、
        #          yield binding、标记消耗，直到用尽（100%）
```

- 确定性排序（无 `Math.random`/无 seed）；同输入 → 同输出。
- 每位点一次 → functionality 自然涌现（无显式参数）。
- 自反应（A×A）：LHS 同 pattern×2 不同映射号，`exclude_same_match` 防 occ_a==occ_b。

### 3. immutable

`apply` 是 immutable 边界：`work = graph.copy()` 一次，循环里 `molrs.Reaction.apply(work, …)` 就地改，
返回 `work`，入参不动。`copy()` 返回 `Self`，`Atomistic→Atomistic`、`CoarseGrain→CoarseGrain` 保持。

### 4. 新增类型（均 molpy 侧基础设施，非化学概念）

| 类型 | 性质 |
|------|------|
| `Crosslinker` | 交联器基类（ABC，plain class） |
| `DeterministicCrosslinker` | 确定性子类 |
| `Candidate` | 候选配对（frozen dataclass） |

## Files to create or modify

### 新建

- `src/molpy/builder/crosslink/__init__.py` — 导出 `Crosslinker`、`DeterministicCrosslinker`
- `src/molpy/builder/crosslink/_crosslinker.py` — `Crosslinker(ABC)` + `Candidate` + 共享 helper（调 molrs）
- `src/molpy/builder/crosslink/_deterministic.py` — `DeterministicCrosslinker`
- `tests/test_builder/test_crosslink/__init__.py`
- `tests/test_builder/test_crosslink/test_crosslinker_base.py` — immutable、molrs 匹配、cutoff（molrs 邻居表）、无坐标回退、cutoff-无坐标报错
- `tests/test_builder/test_crosslink/test_deterministic.py` — 穷尽 100%、spacing 均匀、确定性复现、显式 pairs、自反应

### 修改

- `pyproject.toml` — pin 提供 `SmartsPattern`/`Reaction` 的 molrs 版本（reaction-smarts-01/02 落地后）
- `src/molpy/builder/__init__.py` — 导出 `crosslink` 子模块

## Tasks

- [x] **T1**: `Crosslinker(ABC)`（`_crosslinker.py`）—— `__init__(reaction, *, cutoff)` 建 `molrs.Reaction`、`apply(graph)→新 graph`（copy + molrs 匹配 + 循环 select + molrs.apply + 返回）、抽象 `select`
- [x] **T2**: 基类 helper —— `_candidate_pairs`（`reaction.forming_bonds` + `molrs.NeighborQuery`/全组合）、`_binding`、`_same_molecule`、`_regular_sites`（`topo_distances`+stride）、`_pair_distance`（无坐标+cutoff→ValueError）
- [x] **T3**: `DeterministicCrosslinker`（`_deterministic.py`）—— `select` 穷尽 / `spacing` / `pairs`；确定性排序、每位点一次、`exclude_*`；无 seed/conversion
- [x] **T4**: 导出 + pin molrs 版本（molrs 0.6.0 已提供引擎，pin 已匹配）+ 测试
- [x] **T5**: 质量闸 —— ruff/ty/pytest 全绿

## Testing strategy

- **immutable** — `out = DeterministicCrosslinker(rxn, cutoff=5).apply(g)`：`g` 计数/拓扑不变；`out` 新对象、同子类、含新键。
- **molrs 匹配** — 断言 `reaction.reactant_patterns[*].find_matches_mapped(work)` 被调用（molpy 不自匹配）。
- **cutoff 过滤** — 成键原子对部分在 5Å 内/外：只连 ≤5Å；走 `molrs.NeighborList`（非手搓 O(N²)）。
- **无坐标回退 / fail-fast** — 无坐标不给 cutoff：拓扑全组合确定性配对；给 cutoff 无坐标 → `ValueError`。
- **穷尽 100%** — 2 组分各 N 位点全在 cutoff 内：反应到用尽，消耗 = 理论最大；无 conversion 参数。
- **spacing 均匀** — 线性链每单体一位点、共 M：`spacing=5` 交联点落在第 0,5,10,… 位点，数 ≈ M/5；纯拓扑（无坐标也可选点）。
- **确定性复现** — 同输入两次产物逐键一致。
- **显式 pairs / exclude / 自反应** — `pairs=[(0,2),(1,0)]` 只成两键；`exclude_same_molecule` 无同分子对；A×A+`exclude_same_match` 无自配对。

## Out of scope

- **molrs 引擎**（SMARTS 匹配、SMIRKS 应用、图编辑、原子映射语法） — molrs 链 reaction-smarts-01/02
- **随机配对 / conversion / seed / max_per_molecule** — 02-random
- **PortMatcher / PEO 配方 / builder 收敛** — 03-builder
- **超长键松弛** — 已有 `molpy.optimize`（用户后置）
- **CoarseGrain SMARTS 匹配** — molrs Engine A 针对 Atomistic；CG 匹配 follow-up（molrs 或 molpy label matcher）
