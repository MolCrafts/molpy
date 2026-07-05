# Offline Crosslinking Architecture — Design Proposal v6

Status: draft | Author: Claude Code | Date: 2026-07-05 (supersedes v5)

> **What this is:** building a **pre-crosslinked network offline** as an **immutable graph
> transformation** — `MolGraph` in, a **new** `MolGraph` (same subclass) out. The reaction is
> a **Daylight reaction SMARTS** string. Two modes: **deterministic** and **random**.
> Selection is by distance when the graph has coordinates (pack/place-then-crosslink), else
> topological — **xyz optional**. Stretched bonds are relaxed by the **existing**
> `molpy.optimize`.
>
> **v6 — the molrs/molpy split (per review "共用组件都在molrs，只有 crosslinking 在 molpy"):**
> the **shared engine lives in molrs (Rust)**; **only the crosslinking orchestration is molpy**.
> molrs and molpy move together (two co-authored spec chains).
>
> | Repo | Owns |
> |------|------|
> | **molrs** (Rust + PyO3) | SMARTS engine (atom maps `[C:1]` — *already there*), reusable matcher exposed to Python, **reaction SMARTS / SMIRKS applier** (parse `>>`, atom-map diff → graph edit), graph-edit + `generate_topology` primitives |
> | **molpy** (`builder/crosslink/`) | `Crosslinker(ABC)` + `DeterministicCrosslinker` + `RandomCrosslinker`, pair selection (distance/spacing/conversion/seed), `PortMatcher`, PEO recipes |
>
> **Key discovery:** molrs's canonical SMARTS engine (`molrs::SmartsPattern`,
> `core/chem/smarts/`) **already parses Daylight atom maps** (`QueryAtom.map_label`,
> correct "ignored-in-molecule-SMARTS" semantics) and is production-grade (the OPLS typifier
> drives it). So no grammar work is needed — molrs just needs a PyO3 wrapper + a SMIRKS
> applier. The molpy-side SMARTS engine / grammar / `MatchSet` from v1–v5 are **gone**.

---

## 0. Principles

```
交联 = 纯图重写         MolGraph → 新 MolGraph（同子类），入参不变（immutable）
引擎在 molrs           SMARTS 匹配 + SMIRKS 应用 + 图编辑全在 Rust；molpy 只编排
反应语言 = Daylight     一条 reaction SMARTS "reactants>>products"，SMARTS 之上零自造语法
配对 = 距离优先，可退化   有坐标→按 cutoff 距离（molrs 邻居表）；无坐标→拓扑。xyz 非必选
两种模式               deterministic（穷尽/spacing/显式）| random（到转化率，seed）
松弛 = 复用现成          超长键松弛用已有 molpy.optimize，不进交联 feature
OOP，house-style        plain class + __init__ 配置 + apply(graph)→新 graph，无 classmethod
```

---

## 1. Two worlds (settled)

| | Simulation-time (exists) | Offline pre-crosslink (this) |
|---|---|---|
| When | during MD, engine-triggered | at build time, once |
| Where | `molpy.reacter/bond_react` → LAMMPS `fix bond/react` | new: molrs engine + `molpy.builder/crosslink` |
| Output | a reaction *template* for the engine | a finished crosslinked `MolGraph` |

Distinct features; offline crosslinking outputs a finished structure, never a LAMMPS template.

---

## 2. The reaction language — Daylight SMARTS/SMIRKS, in molrs

The single user-facing notation is a **Daylight reaction SMARTS** (Daylight Theory Manual —
SMARTS + SMIRKS), **implemented in molrs**.

- **Match side (Daylight SMARTS)** — `molrs::SmartsPattern` already implements the atomic /
  bond primitives, logic, recursion, **and atom maps `[expr:n]`** (parsed to
  `QueryAtom.map_label`, emitted as `AtomPrimitive::Any` so the map adds no match constraint
  — correct Daylight "ignored in molecule SMARTS"). molrs work = expose it to Python
  (`reaction-smarts-01`).
- **Reaction side (SMIRKS transform)** — `reactants >> products`; edit derived from the
  atom-map diff (Daylight SMIRKS: pairwise maps preserved, unmapped LHS deleted, unmapped RHS
  added, bond diff → form/break/order). Greenfield in molrs (`reaction-smarts-02`), on top of
  the existing matcher + `add_bond`/`remove_bond`/`add_atom`/`remove_atom`/`generate_topology`.
- **Reaction SMARTS, not strict SMIRKS** — SMARTS queries allowed on reacting atoms
  (`[N;H2:1]`), so functional groups are queryable. No strict mode.

---

## 3. Architecture — molrs engine, molpy orchestration

```
  molrs (Rust + PyO3)                                       molpy (builder/crosslink/)
  ─────────────────────                                     ──────────────────────────
  SmartsPattern (atom-map matcher)  ──find_matches_mapped──▶  DeterministicCrosslinker ─┐
  Reaction (SMIRKS parse + apply)   ──reactant_patterns────▶  RandomCrosslinker         ├─ Crosslinker(ABC)
    · forming_bonds  (distance)     ──forming_bonds────────▶  PortMatcher (选点前端)      │   apply(graph)->新 graph
    · apply(mol, binding)  (edit)   ◀─apply(work, binding)──                             │   copy→match(molrs)→
  Atomistic: copy/remove/topo_dist  ◀─copy/topo_distances──                             ┘   select→apply(molrs)→return
  NeighborList (PBC 距离)            ◀─neighbors within cutoff─
                                                                  (post) LBFGS(SoftPotential()).run(frame)  ← 松弛
```

molpy holds **no** SMARTS engine, **no** SMIRKS applier, **no** `MatchSet`/`Block` — matches
are just molrs's `list[dict[map:handle]]`, paired in plain Python. Two modes = two subclasses
overriding one `select` hook (mirroring `DrudeBuilder`/`Tip4pBuilder` over `VirtualSiteBuilder`).

---

## 4. molpy crosslinker (consumes molrs)

```python
import molrs

class Crosslinker(ABC):                                 # house-style, 仿 VirtualSiteBuilder
    def __init__(self, reaction: str, *, cutoff: float | None = None):
        self._reaction = molrs.Reaction(reaction)       # molrs 解析+编译
        self._cutoff = cutoff
    def apply(self, graph: MolGraph) -> MolGraph:
        work = graph.copy()                             # immutable (molrs Clone)
        occ = [p.find_matches_mapped(work)              # molrs 匹配 → list[{map:handle}]
               for p in self._reaction.reactant_patterns]
        for binding in self.select(work, self._candidate_pairs(work, occ)):
            self._reaction.apply(work, binding)         # molrs SMIRKS 就地改图
        return work
    @abstractmethod
    def select(self, graph, candidates): ...            # mode 唯一差异（距离/spacing/随机 在这里）

class DeterministicCrosslinker(Crosslinker):
    # 穷尽(默认 100%) | spacing=K 均匀交联点(topo_distances) | 显式 pairs；无 RNG
    def __init__(self, reaction, *, cutoff=None, spacing=None, pairs=None,
                 exclude_same_molecule=False, exclude_same_match=False): ...

class RandomCrosslinker(Crosslinker):
    # RandomState(seed) 打乱候选 → 消耗到 conversion；cutoff/exclude_*/max_per_molecule
    def __init__(self, reaction, *, conversion=1.0, seed=None, cutoff=None,
                 exclude_same_molecule=False, exclude_same_match=False, max_per_molecule=None): ...
```

- **cutoff**: 有坐标 → `molrs.NeighborList` 只配 ≤cutoff 的成键原子对（`reaction.forming_bonds`）；
  无坐标 → 拓扑全组合；cutoff 但无坐标 → `ValueError`（xyz 可选）。
- **spacing**: `mol.topo_distances` 沿主链排序、每 K 个取 1 → 均匀交联点（纯拓扑）。
- **immutable**: `work = graph.copy()` 一次，循环就地改，返回 `work`。

---

## 5. User API + PEO gel example

```python
import molpy as mp
from molpy.builder import polymer
from molpy.builder.crosslink import DeterministicCrosslinker, RandomCrosslinker, crosslink_gel

# 均匀网络：deterministic + spacing 均匀交联点 + 邻近连
peg = polymer("{[<][<]CCO[>][>]}", degree=100, count=200)          # builder 建链（含 3D）
gel = crosslink_gel(                                                # crosslink → LBFGS 松弛 → 新 Atomistic
    peg,
    DeterministicCrosslinker(
        "[C:1]=[C:2].[C:3]=[C:4] >> [C:1][C:2][C:3][C:4]",          # Daylight reaction SMARTS（molrs 解析）
        spacing=10, cutoff=6.0,
    ),
)                                                                   # relax 默认走 SoftPotential（无 FF）；传 ff= 用 ForceFieldPotential
mp.io.write(gel, "peo_gel_uniform.data", format="lammps")

# 随机网络：到 70% 转化（同样自动 LBFGS 松弛超长键）
gel_r = crosslink_gel(peg, RandomCrosslinker(rxn, conversion=0.7, seed=42, cutoff=6.0))
```

规则网络提示：`spacing` 给规则**分布**（拓扑）；规则**几何** mesh 需先排列（builder/placer + `cutoff`）
或显式 `pairs`（晶格索引）+ 松弛。

---

## 6. Spec chains (co-moving)

**molrs** (`/Users/roykid/work/molcrafts/molrs/.claude/specs/`):

| Spec | Scope |
|------|-------|
| `reaction-smarts-01-python-matcher` | expose `SmartsPattern` (atom-map matcher, already in Rust) to Python + `PyAtomistic` `remove_atom`/`remove_bond`/`set_bond_order`/`copy` |
| `reaction-smarts-02-smirks-applier` | Daylight reaction SMARTS: parse `>>` + compile atom-map `Transform` + apply to one occurrence; expose `Reaction` to Python |

**molpy** (`builder/crosslink/`, consumes molrs):

| Spec | Scope |
|------|-------|
| `crosslink-01-deterministic` | `Crosslinker(ABC)` + `DeterministicCrosslinker` (穷尽/spacing/pairs); cutoff via molrs `NeighborList` |
| `crosslink-02-random` | `RandomCrosslinker` (conversion, seed) — only overrides `select` |
| `crosslink-03-builder` | `PortMatcher` (ports→occurrence, molrs-shaped) + PEO recipes + builder convergence analysis |

Minimize is not a spec — it's the existing `molpy.optimize`.

---

## 7. What we deliberately do NOT do

- **Put the engine in molpy.** SMARTS matching, SMIRKS applying, and graph editing are molrs
  (shared). molpy only orchestrates crosslinking. (No molpy SMARTS engine, no `smarts.lark`
  grammar work — molrs already has atom maps; no molpy `MatchSet`/`Block`.)
- **Invent syntax on top of SMARTS.** Captures are Daylight atom maps `[C:1]`; the edit is a
  Daylight reaction SMARTS `>>`. Both in molrs.
- **Deviate from Daylight.** Atom/bond primitives, atom maps, and the `>>` transform follow
  the Daylight Theory Manual. Permissive reaction SMARTS (queries on reacting atoms), no strict
  SMIRKS mode.
- **Require coordinates.** Distance is used when present; topological fallback otherwise.
- **Build a bespoke relaxer.** Stretched-bond relaxation is the existing `molpy.optimize`.
- **Compile to LAMMPS `fix bond/react`** — that's the simulation-time feature.
- **Rewrite the working `PolymerBuilder`** — `crosslink-03` only bridges (`PortMatcher`) +
  analyzes; the `Connector`→molrs-engine migration is a gated follow-up.
- **Mutate the input, add `Atomistic.annotations`, or introduce chemical types.**

---

## 8. Builder convergence — outcome (crosslink-03 T5, delivered)

**Conclusion (analysis only — `PolymerBuilder` is not touched):** chain assembly is a
*degenerate crosslink*. `PolymerBuilder._build_from_graph` (`builder/polymer/core.py`) is a thin
driver around two-body reactions where **match** = `atom["port"]` markers, **pair** =
`Connector.select_ports` (adjacent CGSmiles nodes only — no distance / limit / RNG), **edit** =
one `Reacter.run` per side. That is exactly a `Crosslinker` restricted to sequential,
consume-each-port-once pairing.

**What shipped as the convergence bridge:**

- `PortMatcher` (`builder/crosslink/_port_matcher.py`) reads the same `atom["port"]` markers the
  builder uses and emits **molrs-shaped** occurrences (`list[dict[map_number, handle]]`),
  identical to `SmartsPattern.find_matches_mapped`. So "sites marked at modelling time" and
  "sites found now by SMARTS" are two front-ends over **one** molrs edit engine.
- `Crosslinker._match_occurrences` is the seam: override it (as `_PortCrosslinker` in the tests
  does) to feed port occurrences into the unchanged `apply` → `molrs.Reaction.apply` path.

**Follow-up (separate, gated spec — must be zero regression):** migrate `Connector`'s bonding
step onto `molrs.Reaction` (chain-internal bonds and inter-chain crosslinks share one edit
path); keep placement / sequence generation in the builder. Not done here because the builder
is working, tested, and higher-risk to change than to bridge.
