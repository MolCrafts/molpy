---
title: Crosslink 3/3 — PortMatcher 桥接 + PEO 凝胶配方 + polymer builder 收敛
status: done
created: 2026-07-05
depends_on: crosslink-02-random
---

# Crosslink builder — 与 polymer builder 整合

> 三件事：`PortMatcher` 让"建模时标好的位点"（BigSMILES `<`/`>`/`$` → `atom["port"]`）产出与 molrs
> SMARTS 匹配**同形**的 occurrence（`list[dict[map:handle]]`），供 `Crosslinker` 消费；端到端 **PEO
> 凝胶配方**（build → crosslink → `molpy.optimize` minimize → export）；回答 polymer builder 架构——
> **链装配 ≈ 退化交联**，ports 与 SMARTS 是同一 molrs 引擎上的两个"选点前端"。
> 架构见 `.claude/notes/crosslinking-syntax-design.md` (v6)。

## Summary

1. **`PortMatcher`**：读 `atom["port"]` 标记（来自 BigSMILES `<`/`>`/`$` 描述符或 `[*:n]` atom-class
   port）产出 occurrence 列表 `list[dict[int, int]]`（映射号→handle），与 `molrs.SmartsPattern.
   find_matches_mapped` **同形**——于是"建模标好的位点"和"molrs SMARTS 现找的位点"合流，被同一
   `Crosslinker` 消费（**匹配前端可换、molrs 编辑引擎唯一**）。
2. **PEO 凝胶配方**：polymer builder 建 PEG → `DeterministicCrosslinker(spacing=…)` 均匀网络 /
   `RandomCrosslinker(conversion=…)` 随机网络 → `molpy.optimize` 松弛 → 导出。可运行示例 + 回归测试。
3. **polymer builder 收敛（分析 + 建议，不重写）**：`PolymerBuilder._build_from_graph` =
   "match=port / pair=相邻单体（`Connector`）/ edit=一次 `Reacter.run`"的硬编码特例 = 退化交联；
   建议 bonding 步统一走 molrs 引擎、placement/sequence 留 builder，实际迁移列 follow-up（须零回归）。

## Domain basis

### 链装配 = 退化交联（已核 builder 调查）

`PolymerBuilder`（`builder/polymer/core.py:195`）是绕 `Reacter` 两体反应的薄驱动循环：match=`atom["port"]`
标记、pair=`Connector.select_ports`（只配 CGSmiles 图相邻单体，无距离/限量/RNG）、edit=每边一次
`Reacter.run`、化学=`ReactionPresets` 三元组。所以线性链 = 顺序配对、每端消耗一次的**退化交联**。

### 两个选点前端，一个引擎

| 选点前端 | 何时 | 产出 |
|----------|------|------|
| `molrs.SmartsPattern`（molrs 链） | 导入/任意结构，按化学现找 | `list[dict[map:handle]]` |
| `PortMatcher`（本 spec） | 建模就标好（BigSMILES `<`/`>`/`$`、`[*:n]`） | `list[dict[map:handle]]` |

同形 → 下游 `Crosslinker` 与 molrs `Reaction.apply` 消费任一前端。**这才是"重设计 builder 架构"的落点：
选点可换、编辑（molrs）唯一。**

## Design

### 1. `PortMatcher`（house-style，产出与 molrs 同形）

```python
class PortMatcher:
    """把 atom["port"] 读成 occurrence（映射号 1 = 带 port 的原子）。与 molrs 匹配结果同形。"""
    def __init__(self, *, port_key: str = "port"):
        self._port_key = port_key
    def find_matches_mapped(self, graph: MolGraph, port_name: str) -> list[dict[int, int]]:
        # 返回所有 atom[port_key]==port_name 的原子，每个 → {1: handle}（可选 anchor 邻居 → {2: ...}）
```

- 与 `molrs.SmartsPattern.find_matches_mapped(mol)` 同签名形态（`list[dict[int,int]]`），可作为
  `Crosslinker` 的替代选点前端（构造时注入或子类覆写"取 occurrence"这一步）。
- port 来自 BigSMILES 描述符（`parser/smiles/converter.py:457 descriptor_to_port_name`）或 `[*:n]`。纯拓扑，无坐标也可。
- port-based 交联的 reaction 仍是一条 reaction SMARTS（写在 port 原子及 anchor 的局部化学上），molrs 应用。

### 2. PEO 凝胶配方（端到端，可运行 + 回归）

```python
import molpy as mp
from molpy.builder import polymer
from molpy.builder.crosslink import DeterministicCrosslinker, RandomCrosslinker, crosslink_gel

peg = polymer("{[<][<]CCO[>][>]}", degree=100, count=200)          # builder 建链（含 3D）
# 均匀网络：spacing 均匀交联点 + 邻近连；crosslink_gel 内部 LBFGS 松弛
gel = crosslink_gel(
    peg,
    DeterministicCrosslinker(
        "[C:1]=[C:2].[C:3]=[C:4] >> [C:1][C:2][C:3][C:4]", spacing=10, cutoff=6.0,
    ),
)                                                                  # relax 默认 SoftPotential；传 ff= 走 ForceFieldPotential
mp.io.write(gel, "peo_gel_uniform.data", format="lammps")
# 随机网络：到 70% 转化（同样自动 LBFGS 松弛）
gel_r = crosslink_gel(peg, RandomCrosslinker(rxn, conversion=0.7, seed=42, cutoff=6.0))
```

recipe 只**组合**已有件（builder + 01/02 + molrs 引擎 + optimize）；不新增引擎。

### 3. polymer builder 收敛（建议，非本 spec 重写）

目标：placement/sequence 留 builder（builder 特有），**bonding 步收敛到 molrs 引擎**（`molrs.Reaction`）：
`Connector` 选点 = `PortMatcher` + 相邻配对；`ReactionPresets` 三元组 ↔ 一条 reaction SMARTS；链内成键
与链间交联走同一 molrs 编辑路径。**本 spec 不动 `PolymerBuilder`**（能跑、有测试，改它有回归风险）；只交付
`PortMatcher` + recipe，等价关系写成文档+测试证据；`Connector` 迁移列 **follow-up spec**（须零回归）。

### 4. 新增类型

| 类型 | 性质 |
|------|------|
| `PortMatcher` | 读 `atom["port"]` 的选点前端（plain class，无 classmethod） |

## Files to create or modify

### 新建

- `src/molpy/builder/crosslink/_port_matcher.py` — `PortMatcher`
- `src/molpy/builder/crosslink/recipes.py` — PEO 凝胶薄封装（可选）
- `tests/test_builder/test_crosslink/test_port_matcher.py` — port→occurrence（BigSMILES 描述符/`[*:n]`）；与 molrs 匹配同形、可被 Crosslinker 消费
- `tests/test_builder/test_crosslink/test_peo_recipe.py` — 端到端 build→crosslink(spacing/random)→minimize→write

### 修改

- `src/molpy/builder/crosslink/__init__.py` — 导出 `PortMatcher`（及 recipe 封装如有）

## Tasks

- [x] **T1**: `PortMatcher`（`_port_matcher.py`）—— `__init__(*, port_key="port")` + `.find_matches_mapped(graph, port_name)`，映射号 1 = port 原子；与 molrs 匹配同形
- [x] **T2**: 验证 `PortMatcher` 产出可作为 `Crosslinker` 的选点前端（覆写 `_match_occurrences` 取 occurrence 步）
- [x] **T3**: PEO 均匀网络 recipe + 测试（`spacing`+`cutoff`+write；`minimize` 作为注入式 `relax` 回调——molpy 无 `optimize.minimize`，FF 松弛留调用方）
- [x] **T4**: PEO 随机网络 recipe + 测试（`conversion`+`seed`+write；同上，`relax` 可注入）
- [x] **T5**: 文档化 builder↔交联等价（链装配=退化交联，见 notes §8）；`Connector` 迁移列 follow-up（不动 builder）
- [x] **T6**: 导出 + 质量闸；`pytest tests/test_builder/ -m "not external"` 无回归（277 passed）

## Testing strategy

- **PortMatcher** — BigSMILES 建带 `<`/`>` port 的结构：`PortMatcher().find_matches_mapped(g, ">")` 命中所有 `>` port 原子，`{1:handle}` 解引用正确；与 molrs 匹配结果同形（`list[dict[int,int]]`）。
- **可消费** — `Crosslinker` 用 `PortMatcher` 选点正常交联。
- **PEO 均匀 / 随机** — 线性 PEG × N：`spacing`/`conversion` 产网、可 `minimize`、可 `io.write` LAMMPS。
- **builder 无回归** — `pytest tests/test_builder/ -m "not external"` 前后全绿（未改 `PolymerBuilder`）。

## Out of scope

- **molrs 引擎** — molrs 链 reaction-smarts-01/02
- **重写 `PolymerBuilder`/`Connector`**（bonding 迁 molrs 引擎） — follow-up spec，须零回归
- **3D placement / sequence 生成** — 留 builder
- **超长键松弛算法** — 已有 `molpy.optimize`
- **交联网络分析 / 交联剂库** — `compute` / `data` 层，另立
