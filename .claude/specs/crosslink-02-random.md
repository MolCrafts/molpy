---
title: Crosslink 2/3 — RandomCrosslinker（随机到目标转化率，seed 可复现）
status: done
created: 2026-07-05
depends_on: crosslink-01-deterministic
---

# Crosslink random — 随机交联

> 第二个 `Crosslinker` 子类 `RandomCrosslinker`：在 `01` 基类之上**只重写 `select`**——用带 seed 的
> 随机打乱候选、逐个消耗到**目标转化率** `conversion`，遵守 `cutoff`/`exclude_same_molecule`/
> `exclude_same_match`/`max_per_molecule`。Flory–Stockmayer 式随机成网，`seed` 可复现。
> 匹配/距离/改图全继承基类（→ molrs），不重造。架构见 `.claude/notes/crosslinking-syntax-design.md` (v6)。

## Summary

`RandomCrosslinker(Crosslinker)` 只实现 `select`（基类 `apply` 的 copy / molrs 匹配 / molrs 应用 / 返回
全继承）。`select`：取基类候选（已按 `cutoff` 过滤）→ `numpy.random.RandomState(seed)` 打乱 → 逐个消耗
（跳过任一端已消耗/同分子/自配对/超 `max_per_molecule`）→ 到 `conversion` 或无候选时停。配置进
`__init__`；无 classmethod。

## Domain basis

- **随机成网**（区别于 01 穷尽）：反应位点随机偶联到某转化率，得非理想网络（悬挂端、成环、链段不均）。
- **转化率定义**：`conversion` = 已反应位点 / 限制反应物总位点（A×B 取较小；A×A 取半）。达到即停；整数步长
  → 实测 ±1 步。`conversion=1.0` 无可配对时**提前停不死循环**（退化为穷尽随机序）。
- **复用 01 基类**：`_candidate_pairs`（molrs 邻居表 cutoff）、`_binding`、`_same_molecule`、消耗跟踪——全在基类；
  本 spec **只重写 `select`**。匹配/应用仍是 molrs。

## Design

```python
import numpy

class RandomCrosslinker(Crosslinker):
    """随机交联到目标转化率，seed 可复现。"""
    def __init__(self, reaction: str, *, conversion: float = 1.0, seed: int | None = None,
                 cutoff: float | None = None,
                 exclude_same_molecule: bool = False, exclude_same_match: bool = False,
                 max_per_molecule: tuple[str, int] | int | None = None):
        super().__init__(reaction, cutoff=cutoff)
        self._conversion, self._seed = conversion, seed
        self._exclude_same_molecule = exclude_same_molecule
        self._exclude_same_match = exclude_same_match
        self._max_per_molecule = max_per_molecule

    def select(self, graph, candidates):
        rng = numpy.random.RandomState(self._seed)          # seed=None → 非确定
        order = rng.permutation(len(candidates))
        target = self._target_reactions(candidates)         # conversion × 限制反应物位点
        consumed, per_mol, n = set(), {}, 0
        for k in order:
            c = candidates[k]
            if n >= target: break
            if self._skip(c, consumed, per_mol): continue
            yield self._binding(c)
            self._mark(c, consumed, per_mol); n += 1
```

- `_skip`：任一端已消耗；`exclude_same_molecule` 且同分子；`exclude_same_match` 且 occ_a==occ_b；某分子已达 `max_per_molecule`。
- **RNG**：`numpy.random.RandomState(seed)`（molpy 运行时）；同 seed + 同输入 → 同产物。
- immutable / copy-once / molrs 匹配应用 / 返回新图——全继承 `01` 基类。

## Files to create or modify

### 新建

- `src/molpy/builder/crosslink/_random.py` — `RandomCrosslinker`
- `tests/test_builder/test_crosslink/test_random.py` — 转化率精度、seed 复现、cutoff、exclude_*、max_per_molecule、conversion=1 不死循环、CG 无坐标

### 修改

- `src/molpy/builder/crosslink/__init__.py` — 导出 `RandomCrosslinker`

## Tasks

- [x] **T1**: `RandomCrosslinker`（`_random.py`）—— `__init__(reaction, *, conversion, seed, cutoff, exclude_*, max_per_molecule)`，继承基类
- [x] **T2**: `select` —— `RandomState(seed)` 打乱 + 消耗跟踪 + `_skip` + 到 `conversion` 停；`conversion=1.0` 无候选提前停不死循环（循环受候选列表长度约束）
- [x] **T3**: `_target_reactions` —— conversion × 限制反应物位点（A×B 取较小、A×A 取半）
- [x] **T4**: 导出 + 测试
- [x] **T5**: 质量闸 —— ruff/ty/pytest 全绿

## Testing strategy

- **转化率精度** — `conversion=0.5`：消耗位点 / 总位点 ≈ 0.5（±1 步）。
- **seed 复现** — 同 seed 两次产物逐键一致；不同 seed 一般不同。
- **cutoff** — 只在 cutoff 内随机配对（继承基类 molrs 邻居表）。
- **exclude / max_per_molecule** — `exclude_same_molecule` 无同分子对；A×A+`exclude_same_match` 无自配对；`max_per_molecule=2` 任一分子 ≤2。
- **conversion=1.0 不死循环** — 无更多可配对时提前停。
- **CG 无坐标** — bead 网络不给 cutoff：随机拓扑配对到转化率，`CoarseGrain→CoarseGrain`（前提：molrs 支持 CG 匹配；否则记 follow-up）。
- **immutable（继承）** — 入参不变、返回新图。

## Out of scope

- **molrs 引擎** — molrs 链 reaction-smarts-01/02
- **穷尽/显式/spacing 配对** — 01-deterministic
- **PortMatcher / PEO 配方 / builder 收敛** — 03-builder
- **反应动力学/速率模型** — `conversion` 只管拓扑转化率
