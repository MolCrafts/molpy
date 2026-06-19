---
title: 基于相似性的力场缺失成键参数估计器 ParameterEstimator
status: superseded
created: 2026-06-18
superseded_on: 2026-06-18
superseded_by: molrs ff/typifier (Rust 整体下沉)
---

> ⚠️ **SUPERSEDED (2026-06-18)** — 决策改为把 OPLS/GAFF/CL&P 分型**连同**估计器整体下沉到 molrs(Rust)，molpy 退成薄调用层。本 spec 的 molpy-Python 实现方案作废；**算法/Domain basis（parmchk2 级联、CGenFF 惩罚、GAFF 经验公式、单位对齐、溯源约定、验收容差）整体转用于 molrs 仓库的 Rust 版 spec**。集成锚点不再是 atomistic.py 的 Python typifier，而是 molrs `ff/typifier/{opls,gaff}` 的 Rust no-match 分支。新 spec 落在 `/Users/roykid/work/molcrafts/molrs/.claude/specs/`。

# 基于相似性的力场缺失成键参数估计器 ParameterEstimator

> 锚定在 `src/molpy/typifier/atomistic.py` 的三个生产 typifier（`ForceFieldBondTypifier` / `ForceFieldAngleTypifier` / `ForceFieldDihedralTypifier`）的 `best_X is None and not self.strict` 兜底分支。不改 strict 路径，不建并行类层级。

## Summary

为 molpy 增加一个独立的、按需注入的力场参数估计器 `ParameterEstimator`：当 OPLS-AA / GAFF / CL&P 的成键参数（bond / angle / dihedral）在精确匹配与 class 匹配下均缺失时，用 parmchk2 式的类比级联（精确类型 → 等价类型 → 对应类型 → 通配符）、加性惩罚（CGenFF 式内原子加权），以及 GAFF 经验兜底公式（键力常数 Badger 型反幂律、角 θ₀ 取共享中心原子已有角均值、角 K_θ GAFF 经验式、二面角回退通用项或近零势垒）来补齐参数，完全不做从头量化计算。估计器以可选属性 `self._estimator` 组合注入到上述三个生产 typifier，默认关闭（opt-in）；当用户开启且 `strict=False` 时，原本静默返回无参数项的兜底分支会改为给出一个带溯源（`estimated` / `estimate_method` / `estimate_penalty` / `estimate_analog`）与置信度分级（penalty <10 可靠 / 10–50 谨慎 / >50 差）的估计参数；精确匹配始终优先，`strict=True` 行为完全不变。

## Domain basis

类比 / 相似性补参数是力场领域成熟标准做法（parmchk2、CGenFF、MATCH、OpenFF、MMFF94 均用）。成立条件：必须报告溯源 + penalty；相似性须尊重元素 / 杂化 / 周期；取最近类比**直接复制**（不取平均）；二面角回退到通用项或近零势垒，**绝不伪造势垒**。

1. **parmchk2 类比级联** — Wang et al., *J. Mol. Graph. Model.* 2006, 25:247–260, DOI:10.1016/j.jmgm.2005.12.005。每个缺失项按层级搜索类比：(1) 精确 type 匹配 → (2) 等价原子类型（共振 / 几何孪生，penalty 0）→ (3) 对应原子类型（按属性差异累加 penalty）→ (4) 通配符。最低总 penalty 胜出，直接复制其参数。
2. **加性惩罚 + 内原子加权** — CGenFF: Vanommeslaeghe & MacKerell, *J. Chem. Inf. Model.* 2012, 52:3144–3154 (DOI:10.1021/ci300363c) 及 52:3155–3168 (DOI:10.1021/ci3003649)。`penalty = Σ_i w_pos(i)·subst_penalty(t_i, p_i)`。内原子（角中心、二面角两个内原子、improper 中心）权重 ×10。阈值带（CGenFF 发布值）：penalty **<10 可靠 / 10–50 谨慎使用建议人工核验 / >50 差需优化**。
3. **GAFF 经验兜底公式** — Wang et al., *J. Comput. Chem.* 2004, 25:1157–1174, DOI:10.1002/jcc.20035（基于元素，力场无关）：
   - **键力常数**：Badger 型反幂律（力常数随键长的反幂关系，按元素对标定常数）。exact 形式与元素对常数表从 GAFF 原文 / parmchk2 源码逐字转录，由经验公式单测钉死。
   - **角 θ₀**：取共享中心原子 B 的已有角 θ₀(A-B-A) 与 θ₀(C-B-C) 的**均值**。
   - **角 K_θ**：GAFF 角力常数经验式（含常数 143.9、按元素的 Z 因子与 C 因子、依赖 θ₀ 与键长）。exact 形式与 Z/C 表从 GAFF 原文逐字转录，单测钉死。
   - **二面角**：优先回退到最通用的已有扭转项（通配端 `X-b-c-X`，按两个中心原子键控）；若无则给**近零势垒 + 高 penalty**，绝不伪造刚性势垒；多重周期项须**整组复制**（所有 periodicity / k / phase）。
4. **单位对齐（correctness 约束）**：oplsaa.xml 与 gaff.xml 单位系不同——OPLS：键长 nm / 键 k kJ·mol⁻¹·nm⁻² / 角 θ₀ rad / 角 k kJ·mol⁻¹·rad⁻² / 扭转 RB c0–c5 kJ·mol⁻¹；GAFF：键长 Å / 键 k kcal·mol⁻¹·Å⁻² / 角 θ₀ rad / 角 k kcal·mol⁻¹·rad⁻² / 扭转 periodic k kcal·mol⁻¹ phase rad。**最近类比复制天然单位安全**（逐字复制同表兄弟项的值，单位 / Style 一致）。经验公式天生产出 AMBER 单位（Å、kcal），写入前必须**采样目标 FF 同类已有项探测单位并换算**。这既是"优先类比复制、经验仅作最后兜底"的质量选择，也是 OPLS 目标的正确性要求。
5. **关键陷阱**：① 不伪造本应近零的二面角；② 区分"FF 故意省略二面角靠 1-4 补"与"真缺失需估"——v1 保守：仅在该中心键无任何扭转项时才估计；③ 多重周期二面角整组复制；④ 不同力场等价类型表不可互串；⑤ 经验角 K_θ 单位 / 约定用回归测试钉死。

## Design

**1. 模块归属与命名**

新建独立模块 `src/molpy/typifier/estimator.py`，公开类 `ParameterEstimator`（PascalCase，**无 `Typifier` 后缀**——它不是完整 pipeline typifier，只是缺参兜底补全器）。内部 helper 一律前导下划线，对齐 atomistic.py 现有 `_end_score` / `_sequence_score` 风格：`_analogy_score`、`_substitution_penalty`、`_empirical_bond_k`、`_empirical_angle_theta0`、`_empirical_angle_k`、`_generic_dihedral` 等。构造 `ParameterEstimator(ff: ForceField)`，等价 / 对应替换表与元素对 / Z·C 常数表在 `__init__` 一次性从 `data/forcefield/*.json` 加载并缓存（镜像 gaff.py:72 的资产加载范式）。`BondType` / `AngleType` / `DihedralType` / `AtomType` 由 `molpy.core.forcefield`（molrs re-export）取得。

**2. 调用时机：lazy、按需、组合注入**

估计器以可选属性 `self._estimator: ParameterEstimator | None`（默认 `None`，opt-in）组合注入三个生产 typifier，镜像 `ForceFieldTypifier._init_typifiers` (atomistic.py:563) 组合 sub-typifier 的方式，**不建并行类层级**。三个 typifier `__init__` 增加可选形参（如 `estimator: ParameterEstimator | None = None`）存为 `self._estimator`；`ForceFieldTypifier` 增加 `estimator` 透传给 `_init_typifiers`。集成钩子在每个 typifier 的兜底分支（atomistic.py:193-194 / 251-252 / 317-318），即 `best_X is None and not self.strict`，在原 `return term` 之前插入：

```python
if self._estimator is not None:
    term = self._estimator.estimate_bond(   # / estimate_angle / estimate_dihedral
        term, self._bond_table, self._type_to_class, self._class_to_layer
    )
return term
```

加性改动：精确 / class 匹配（`best_X is not None`）始终优先且路径不变；`strict=True` 维持现有 `raise ValueError`，估计器**完全不介入** strict 路径；`self._estimator is None`（默认）时行为与现状逐字节一致。保持 mutate-in-place + return term 契约。

**3. 复用既有打分基元（不 rederive、不从 ff.get_types 重建表）**

相似度 / 惩罚构建在现有 `_end_score`（atomistic.py:57，特异性 3/1/0/None）与 `_sequence_score`（atomistic.py:74，已含 forward+reversed 端对端对称求和）之上，继承端对端对称性。候选表直接复用各 typifier 已构建的 `self._bond_table`（L152）/ `_angle_table`（L211）/ `_dihedral_table`（L269），形如 `[((name_i, name_j[, ...]), TypeObj)]`，连同 `_type_to_class`、`_class_to_layer` 一并作为 `estimate_*` 入参传入。`_build_type_class_layer`（atomistic.py:25）的结果由 typifier 持有，估计器不重建。`core/utils.py:15 get_nearest_type` 与本特性无关，不使用。

**4. 参数发射形式与溯源约定**

估计出的项照 atomistic.py:189-190 的形式写回：`term.data["type"] = TypeObj.name; term.data.update(**TypeObj.params.kwargs)`——估计器必须产出与目标 Style 相同 `params.kwargs` 形状的字典。溯源为新建约定（bonded-term 无既有先例；唯一先例是 atom 级 matcher.py:312 的 `source` 字段如 `"oplsaa:CT"`，命名对齐之）。估计器在 `term.data` 额外写入：`estimated=True`、`estimate_method ∈ {analogy, empirical, generic-wildcard}`、`estimate_penalty=<float>`、`estimate_analog=<源 TypeObj.name 或 None>`。类比复制路径时 `estimate_analog` = 被复制兄弟项名；经验 / 通用通配路径时为 `None`。

**5. 生命周期 / 所有权**

`ParameterEstimator` 不拥有 typifier 状态，只读取传入的候选表与映射 + 自身缓存的常数 / 等价表。错误处理 fail-fast `ValueError`，与 atomistic.py:195 一致；not-strict 路径下估计器最终失败（连经验兜底都无法产出，如二面角既无通用项又无法判定近零适用）时一致降级——v1 约定：返回原无参项（与 `self._estimator is None` 同效果），不另开错误通道。新公开符号：`ParameterEstimator`（从 `typifier/__init__.py` 导出）。新数据资产：`gaff_equiv.json`、`bond_empirical.json`、`angle_empirical.json`。

## Files to create or modify

- `src/molpy/typifier/estimator.py` (new) — `ParameterEstimator` 类 + `estimate_bond` / `estimate_angle` / `estimate_dihedral` + 私有 helper（类比级联、加性惩罚、GAFF 经验公式、通用二面角回退、单位探测换算）。
- `src/molpy/typifier/atomistic.py` — 三个 typifier `__init__` 增加可选 `estimator` 形参存为 `self._estimator`（默认 `None`）；在 L193-194 / L251-252 / L317-318 的 `not self.strict` 分支 `return term` 前插入估计器调用；`ForceFieldTypifier.__init__` + `_init_typifiers` 透传 `estimator`。
- `src/molpy/typifier/__init__.py` — 导出 `ParameterEstimator`，加入 `__all__`。
- `src/molpy/data/forcefield/gaff_equiv.json` (new) — GAFF 等价 / 对应原子类型替换表 + 取代惩罚值。
- `src/molpy/data/forcefield/bond_empirical.json` (new) — Badger 型键力常数元素对常数表（从 GAFF 原文 / parmchk2 逐字转录）。
- `src/molpy/data/forcefield/angle_empirical.json` (new) — 角 K_θ GAFF 经验式 Z / C 因子表 + 常数 143.9（从 GAFF 原文逐字转录）。
- `tests/test_typifier/test_estimator.py` (new) — 经验公式钉死、留一法还原、strict 回归、单位对齐、溯源 / penalty 分级、二面角不伪造、组合注入。
- `tests/test_typifier/test_estimator_external.py` (new) — parmchk2 金标准交叉验证（`@pytest.mark.external`）。

## Tasks

- [ ] Write failing tests for GAFF empirical formulas (bond k Badger, angle θ₀ mean, angle K_θ Eq.5) against transcribed reference values (tests/test_typifier/test_estimator.py)
- [ ] Transcribe GAFF/parmchk2 constants into data/forcefield/bond_empirical.json, angle_empirical.json, gaff_equiv.json and implement _empirical_bond_k / _empirical_angle_theta0 / _empirical_angle_k in src/molpy/typifier/estimator.py
- [ ] Write failing tests for analogy cascade + additive penalty (estimate_bond/angle/dihedral leave-one-out recovery + penalty tiers + source fields) (tests/test_typifier/test_estimator.py)
- [ ] Implement ParameterEstimator analogy cascade (_analogy_score on _sequence_score/_end_score, _substitution_penalty inner-atom weighting, nearest-analog copy) and estimate_bond/estimate_angle/estimate_dihedral with source-field emission in src/molpy/typifier/estimator.py
- [ ] Write failing tests for dihedral generic fallback + near-zero barrier + multi-periodicity group copy + unit alignment for OPLS target (tests/test_typifier/test_estimator.py)
- [ ] Implement _generic_dihedral wildcard fallback, near-zero-barrier-with-high-penalty path, multi-periodicity group copy, and target-FF unit-probe/convert in src/molpy/typifier/estimator.py
- [ ] Write failing tests for opt-in composition injection + strict=True non-interference regression (tests/test_typifier/test_estimator.py)
- [ ] Wire self._estimator into ForceFieldBondTypifier/AngleTypifier/DihedralTypifier not-strict fallback branches and ForceFieldTypifier passthrough in src/molpy/typifier/atomistic.py; export ParameterEstimator from src/molpy/typifier/__init__.py
- [ ] Add Google-style docstrings with units for ParameterEstimator public methods and document the term.data source convention (estimated/estimate_method/estimate_penalty/estimate_analog)
- [ ] Write parmchk2 gold-standard cross-validation test marked @pytest.mark.external (tests/test_typifier/test_estimator_external.py)
- [ ] Run full check + test suite

## Testing strategy

- **Happy path / unit** — 经验公式（Badger 键 k、角 θ₀ 均值、Eq.5 K_θ）对已知元素对 / 角输入产出钉死数值（GAFF 原文示例值，容差内）。
- **Happy path / unit** — 留一法（leave-one-out）：从 oplsaa.xml（或 gaff.xml）删除一个已知扭转 / 键 / 角（如 PEO 醚相关 X-CT-OS-CT 一类），估计器补出参数在容差内还原真值。容差：bond r₀ ≈ 0.02 Å、angle θ₀ ≈ 3°、力常数 ≈ 10%。
- **Edge / unit** — strict=True 回归：缺参时仍 `raise ValueError`，估计器不介入；`self._estimator is None`（默认 opt-in 关闭）时行为与现状一致。
- **Edge / unit** — 单位对齐：对 OPLS（nm/kJ）目标，经验兜底产出参数单位与同表既有项一致（不是裸 kcal/Å）。
- **Edge / unit** — 溯源：估计项带 `estimated` / `estimate_method` / `estimate_penalty` / `estimate_analog`；penalty 分级阈值 <10 / 10–50 / >50 正确标注。
- **Edge / unit** — 二面角不伪造：缺扭转且无通用项时给近零势垒 + 高 penalty，不产出非零刚性势垒；多重周期项整组复制。
- **Domain validation / scientific (external)** — parmchk2 金标准交叉验证（`@pytest.mark.external`，AmberTools25 conda env）：对一个含 GAFF 缺失项的分子，molpy 估计 vs parmchk2 frcmod 在容差内一致。
- **Build smoke** — `ruff format --check`、`ruff check`、`ty check`、`pytest -m "not external"` 全绿。

## Out of scope

- **OPLS / CL&P 的等价类型表 curate** → v2。v1 仅编码 GAFF 等价 / 对应替换表 + 元素对 / Z·C 常数；但估计器对 OPLS / CL&P 仍可走"元素 / 杂化相似 + GAFF 经验兜底"通用路径（不依赖专属等价表）。
- **improper 估计** v1 优先级最低、可选；本 spec 不包含 improper 兜底注入（`ForceFieldImproperTypifier` 若存在不改）。
- **OpenFF 式 AM1-Wiberg 键级插值**（molrs 暂无键级）→ 不做。
- **从头量化计算** / 任何 QM 拟合 → 明确排除，估计器纯经验 / 类比。
- **eager 全量 pass**：不做开机全表补全；仅在三个生产 typifier 无匹配兜底分支按需调用。
- **wrapper 层 `Parmchk2Wrapper`（wrapper/prepgen.py）import** → 排除（违反分层）；纯 Python 复刻算法，parmchk2 仅作行为参考与外部金标准验证。
- **改动** `typifier/bond.py` / `angle.py` / `dihedral.py` / `base.py`（未导出、不在生产路径的旧 FF-agnostic typifier）→ 排除。
