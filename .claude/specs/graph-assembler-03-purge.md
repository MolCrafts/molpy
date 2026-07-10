---
title: 删除 reacter/ —— 反应语义统一到 molrs.Reaction
status: code-complete
created: 2026-07-10
depends_on: "graph-assembler-02-kernel"
---

# 一次性删除,不留兼容层

> chain graph-assembler 3/4。**不依赖任何 molrs 新 API**。
> 02 之后 `reacter/` 已经没有调用者了 —— 本 spec 只做删除、迁移、和一次
> **breaking public API** 的记账。

## Summary

02 把 `PolymerBuilder` / `Crosslinker` 收敛到 `GraphAssembler` + `molrs.Reaction` 之后,
`molpy/reacter/`(2682 行)与 `builder/polymer/connectors.py` + `presets.py`(362 行)成为死代码。

但**它们不是纯死代码**:

- `molpy.reacter` 是 `molpy/__init__.py` 里的公开子包(第 29/47/113 行)。删它是 **breaking change**。
- `reacter/bond_react.py` 的 `BondReactTemplate` 被 `io/data/lammps_bond_react.py:write_bond_react_map`
  消费 —— 那是一个**用户可见的导出格式**,不能一起删。它的替身是 02 的 `RegionPatch`。
- 约 **143 个 `tests/test_reacter/` 测试** + `test_polymer_connectors.py` + `test_polymer_presets.py`
  + `tests/test_io/test_data/test_lammps_bond_react.py` + `tests/test_core/test_affected_region.py`
  会红。
- **6 个文档页 / notebook** 教的是 `Reacter` / `Connector` / `port` / `<` `>`。

旧 spec 只算了"净删除 ≈ 3100 行",没有计入这 ~2000 行测试与 6 篇文档。那不是记账误差,那是**这个
spec 的主要工作量**。

## Domain basis

### 1. LAMMPS `bond_react` 的 pre/post 模板就是序列化的 `RegionPatch`

`fix bond/react` 需要一对 pre/post 分子模板 + 一张类型映射。这正是一个受影响区域在编辑前后的
两个快照。`BondReactTemplate` 手写了这套结构;`RegionPatch` 已经携带 `types` + `terms` +
canonical order。

所以迁移是**同构替换**,不是重写:`LammpsBondReactWriter(patch_before, patch_after)`。
`io/data/lammps_bond_react.py` 里的 `collect_type_maps` / `apply_type_maps` /
`write_bond_react_map` 三个自由函数并入一个 writer 类,和 `io/` 其余 writer 对齐。

golden 文件不变 —— 这是本 spec 里唯一"输出必须逐字节相同"的地方,也是删除 588 行
`reacter/bond_react.py` 的安全网。

### 2. 硬编码化学必须随 `reacter/` 一起走

- `reacter/selectors.py` 的 `select_hydroxyl_group` / `select_dehydration_*` 把**具体反应**写进库。
- `builder/polymer/presets.py` 的 `"dehydration"` / `"condensation"` 同理。
- `builder/polymer/core.py:568-597` `_cleanup_stale_ports` 写死 `element == "O"` 与 `element == "H"`
  —— "去掉没有氢的氧上的 port 标记" 是缩聚反应的专有清理逻辑,泄漏进了通用 builder。

02 之后这三处都不再被调用(反应语义在 SMARTS 里),但**必须显式确认删除**,否则它们会以"工具函数"
的名义活下来。

### 3. `Connector` 是单次使用的门面

`PolymerBuilder.__init__` 接受 `connector` **或** `reacter`,二选一,并在收到 `reacter` 时
`Connector(reacter=reacter)` 把它包起来。`Connector` 的全部职责是 `select_ports` + `connect`,
在 02 之后分别是 `select()` 和 `molrs.Reaction.apply`。一个只被一个类构造、只被一个类调用的
抽象,不是抽象。

### 4. 工作流拼装属于文档,不属于库(铁律 4)

`builder/crosslink/recipes.py` 的 `crosslink_gel()` 与 `write_lammps()` 是**自由函数**,
且它们的全部内容是把 `Crosslinker` + `LBFGS` + writer 三个已有的类按顺序调一遍
——它自己的 docstring 就写着"compose existing pieces, add no engine"。

这正是铁律 4 最后一句要排除的东西:一个只在教程里出现一次的调用序列,不该占一个公开
函数名。它搬进 `docs/user-guide/04_crosslinking.md`,以叙述 + 代码块的形式存在。
`molpy.builder.crosslink.__init__` 的 `crosslink_gel` / `write_lammps` 导出一并移除。

### 5. `placer.py` 的"缺元素就当碳"(铁律 1 + 铁律 5)

```python
# builder/polymer/placer.py:124-125
left_symbol  = left_anchor.get("symbol", "C")
right_symbol = right_anchor.get("symbol", "C")
```

一个原子没有 `symbol` 列,几何放置就**默默按碳的共价半径**摆位。两条铁律同时踩:
字段带默认值(1)、缺前置条件返回中性值(5)。而且它错得很安静 —— 键长偏了 0.1 Å 不会有人发现。

改成读 `fields.SYMBOL`,缺列 `KeyError`。同一文件里 `_get_port_direction` 随 `port` 一起删除。

## Files

**删除**

| 文件 | 行数 | 理由 |
|---|---|---|
| `reacter/base.py` | 753 | `Reacter` / `ReactionResult` / 三处 `hasattr` 嗅探 / 全图 fallback |
| `reacter/selectors.py` | 458 | 14 个 selector;硬编码化学 |
| `reacter/utils.py` | 367 | bond former(键级写在 SMARTS 的 RHS) |
| `reacter/topology_detector.py` | 516 | `generate_topology`(molrs)已有 |
| `reacter/bond_react.py` | 588 | pre/post 模板 → `RegionPatch` |
| `reacter/__init__.py` | 135 | — |
| `builder/polymer/connectors.py` | 188 | `Connector` / `port_role` / `ports_compatible` |
| `builder/polymer/presets.py` | 174 | `"dehydration"` / `"condensation"` |
| `builder/crosslink/recipes.py` | 73 | `crosslink_gel` / `write_lammps` —— 自由函数 + 工作流门面 |

> `builder/polymer/core.py`(含 `_cleanup_stale_ports`)、`builder/crosslink/_crosslinker.py`
> `_deterministic.py` `_random.py`、`builder/polymer/placer.py` 已由 **02** 删除/迁入
> `builder/assembly/`。本 spec 只清理它们的**残余引用**(`recipes.py` 从 `_crosslinker` 导入
> `Crosslinker`,`connectors.py` / `presets.py` 从 `reacter` 导入)。

**改**

- `src/molpy/__init__.py` — 删 `reacter` 子包导出(3 处)
- `src/molpy/io/data/lammps_bond_react.py` — `LammpsBondReactWriter` 消费 `RegionPatch`;
  `collect_type_maps` / `apply_type_maps` / `write_bond_react_map` 三个自由函数并入类
  (铁律 4);golden 输出逐字节不变
- `src/molpy/builder/polymer/__init__.py` — 删 `Connector` / preset 导出
- `src/molpy/builder/crosslink/__init__.py` — 删 `crosslink_gel` / `write_lammps` 导出;
  docstring 里的 `site_field` 措辞改为 `fields.SITE`
- `src/molpy/core/entity.py:547`、`src/molpy/core/atomistic.py:718` — docstring 里的 "reacter" 措辞

**测试迁移**

- 删 `tests/test_reacter/`(143 测试)。其中**仍然有效的行为断言**(键级、离去基团、拓扑生成)
  移入 `tests/test_builder/test_assembler.py`,用 `molrs.Reaction` 重写;**测试实现细节的**
  (`ReactionResult.entity_maps`、`TopologyDetector` 内部)直接删,commit body 逐类说明
- 删 `tests/test_builder/test_polymer_connectors.py`、`test_polymer_presets.py`
- 改 `tests/test_io/test_data/test_lammps_bond_react.py` — 喂 `RegionPatch`,golden 不变
- 改 `tests/test_core/test_affected_region.py` — 生产者换成 `GraphAssembler`
- 改 `tests/test_docs/test_doc_examples.py` — **它会 `exec` `docs/api/reacter.md` 的每个
  python 块**,还断言 `ReactionPresets.register` 是公开扩展点、并对
  `src/molpy/reacter/base.py` 跑 doctest。三处随本 spec 一起改

**文档与示例迁移**

- `docs/api/reacter.md`(2.8K,**代码块被执行**)→ 重写为 `docs/api/assembly.md`
- `examples/02_build_polymer.py`、`examples/03_polymer_topology.py` — 用 `GraphAssembler` 重写
  (`test_doc_examples.py` 会跑它们的 `main()`)
- `docs/developer/extending-typifiers.md` — **铁律 6**:扩展示例里的
  `molrs.SmartsPattern("[C:1][O:2]")` 改为 `mp.SmartsPattern(...)`;
  `T` 的类型说明 `molrs.Atomistic` → `molpy.Atomistic`(二者本就是同一个类)。
  写扩展的人也是用户,不该被要求知道 molrs
- `docs/developer/molrs-backend.md` — **唯一**可以正面讨论 molrs 的文档,不动

**文档迁移**

`docs/user-guide/02_polymer_stepwise.md(+.ipynb)`、`03_polymer_topology`、`04_crosslinking`、
`05_polydisperse_systems`、`06_typifier.ipynb`、`docs/developer/architecture-overview.md`、
`docs/changelog.md`。

`Reacter` → `molrs.Reaction`;`Connector` / `port` / `<` `>` `$` → `site` 列 + `%site` 谓词;
`PolymerBuilder(connector=...)` → `PolymerBuilder(library, reaction, typifier=...)`。

> 文风按 `.claude/notes/docs-style.md`:先讲为什么,再给代码;不列 API。
> notebook 用 `python scripts/render_notebooks.py` 重渲染,`nbstripout` 保证不提交输出。

## Tasks

- [x] T1 `LammpsBondReactWriter(RegionPatch, RegionPatch)`;三个自由函数并入类;
      golden 逐字节回归;删 `reacter/bond_react.py`
- [x] T2 `tests/test_reacter/` 分流:有效行为断言 → `tests/test_builder/test_assembler.py`,
      实现细节断言 → 删(commit body 逐类说明)
- [x] T3 删 `reacter/` 整包 + `connectors.py` + `presets.py` + `crosslink/recipes.py`
      (`polymer/core.py`、`crosslink/_*.py`、`placer.py` 已在 02 处理)
- [x] T4 `molpy/__init__.py` 去掉 `reacter` 公开导出;`docs/changelog.md` 记 **breaking**
- [ ] T5 6 篇文档 / notebook 改写 + `scripts/render_notebooks.py` 重渲染;
      `crosslink_gel` 的调用序列以叙述形式落进 `04_crosslinking.md`
- [x] T6 `.claude/notes/architecture.md` 与 `CLAUDE.md` 的包表去掉 `reacter`
- [ ] T7 minor 版本号 bump(experimental stage,但公开子包消失需要显式记账)
- [ ] T8 全仓库铁律扫描:`^def ` 在 `builder/` `typifier/` `io/data/lammps_bond_react.py`
      计数为 0;字段字符串字面量清零
- [ ] T9 **铁律 5**:`placer.py:124-125` 的 `get("symbol","C")` → `fields.SYMBOL` + `KeyError`
      (symbol 是**身份**);共价半径之和作初始键长**保留**,改为具名常量 + docstring
      注明"初始猜测,由几何优化收敛"(**数值初值**,合法例外)。
      `virtualsite.py` 的 `get("charge",0.0)` / `get("x",0.0)` 同治;
      `polymer/system.py` 的 `callable(getattr(distribution,"sample_dp",None))` 能力嗅探
      → `isinstance` + 构造时 `TypeError`
- [x] T10 `docs/api/reacter.md` → `docs/api/assembly.md`;`tests/test_docs/test_doc_examples.py`
      的三处断言(`_exec_blocks("reacter.md")`、`ReactionPresets.register` 公开面、
      `reacter/base.py` doctest)一并改;`examples/02_build_polymer.py` 与
      `examples/03_polymer_topology.py` 用 `GraphAssembler` 重写
- [ ] T11 新建 `docs/user-guide/02_assembly.md`(由 `.claude/notes/assembly-guide-draft.md` 定稿),
      并入 `zensical.toml` nav,替换 02/03/04 三页

## Testing

- `pytest tests/ -m "not external"` 全绿
- `write_bond_react_map` 的 golden 文件**逐字节相同**(唯一的字节级判据)
- `import molpy; molpy.reacter` → `AttributeError`(不是留一个 deprecated shim)
- 反硬编码 grep(见 acceptance)
- `pytest --cov=src/molpy tests/` 覆盖率不低于删除前(分母变小,分子不能变小得更快)

## Out of scope

- `reach` 由 molrs 编译期导出、无界谓词构造期报错 → 04
