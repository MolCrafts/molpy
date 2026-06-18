---
slug: ff-parameter-estimator
criteria:
  - id: ac-001
    summary: ParameterEstimator importable from molpy.typifier, no Typifier suffix
    type: code
    pass_when: |
      `from molpy.typifier import ParameterEstimator` succeeds, the class name
      is exactly "ParameterEstimator", and "ParameterEstimator" appears in
      molpy.typifier.__all__.
    status: pending
  - id: ac-002
    summary: estimator data assets resolve via get_forcefield_path
    type: code
    pass_when: |
      get_forcefield_path("gaff_equiv.json"),
      get_forcefield_path("bond_empirical.json") and
      get_forcefield_path("angle_empirical.json") each return a path to an
      existing file under src/molpy/data/forcefield/.
    status: pending
  - id: ac-003
    summary: empirical bond k (Badger) matches transcribed GAFF reference
    type: scientific
    evaluator_hint: "marker: estimator"
    pass_when: |
      _empirical_bond_k for at least two element pairs with known GAFF
      reference bond k (transcribed into the test) reproduces the reference
      force constant within rtol 1e-3 (formula pinned, not data-fit tolerance).
    status: pending
  - id: ac-004
    summary: empirical angle theta0 is mean of shared-center existing angles
    type: scientific
    evaluator_hint: "marker: estimator"
    pass_when: |
      Given existing angles A-B-A (θ1) and C-B-C (θ2) sharing center B,
      _empirical_angle_theta0 for A-B-C returns (θ1+θ2)/2 within rtol 1e-6.
    status: pending
  - id: ac-005
    summary: empirical angle K_theta (GAFF Eq.5) matches transcribed reference
    type: scientific
    evaluator_hint: "marker: estimator"
    pass_when: |
      _empirical_angle_k for at least one angle with known GAFF reference K_θ
      (using transcribed 143.9 constant + Z/C factors) reproduces the reference
      within rtol 1e-3.
    status: pending
  - id: ac-006
    summary: leave-one-out analogy recovers a deleted bonded term within tolerance
    type: scientific
    evaluator_hint: "marker: estimator; fixture: deleted X-CT-OS-CT-class term"
    pass_when: |
      After removing one known bond/angle/dihedral from the loaded force field,
      ParameterEstimator recovers parameters within bond r0 atol 0.02 Å,
      angle θ0 atol 3°, and force constants rtol 0.10 versus the removed truth.
    status: pending
  - id: ac-007
    summary: analogy path copies nearest analog params verbatim, not averaged
    type: code
    evaluator_hint: "marker: estimator"
    pass_when: |
      When an exact-equivalence analog exists, the estimated term.data params
      equal that analog TypeObj.params.kwargs value-for-value (no averaging),
      and estimate_analog equals that analog's name.
    status: pending
  - id: ac-008
    summary: source-provenance keys written on every estimated term
    type: code
    evaluator_hint: "marker: estimator"
    pass_when: |
      Every term returned via an estimation path has term.data["estimated"] is
      True, estimate_method in {"analogy","empirical","generic-wildcard"},
      estimate_penalty a float, and estimate_analog a str or None.
    status: pending
  - id: ac-009
    summary: penalty tier classification <10 / 10-50 / >50 correct
    type: code
    evaluator_hint: "marker: estimator"
    pass_when: |
      For constructed substitution cases with known total penalty, the tier
      label (reliable <10 / use-with-caution 10–50 / poor >50) matches the
      CGenFF threshold bands exactly at the boundary values.
    status: pending
  - id: ac-010
    summary: strict=True path unchanged; estimator does not intervene
    type: code
    evaluator_hint: "marker: estimator"
    pass_when: |
      With strict=True a missing bond/angle/dihedral still raises ValueError
      even when an estimator is attached, and with self._estimator is None the
      not-strict fallback returns the term with no estimate_* keys added.
    status: pending
  - id: ac-011
    summary: empirical OPLS-target output uses target-FF units (nm/kJ), not kcal/Å
    type: scientific
    evaluator_hint: "marker: estimator"
    pass_when: |
      When estimating an empirical term for an OPLS (nm/kJ) target force field,
      the written params are in the target FF's units (consistent in magnitude
      with sibling existing terms in the same table), not raw AMBER kcal/Å.
    status: pending
  - id: ac-012
    summary: dihedral never fabricates a rigid barrier; multi-periodicity copied as a group
    type: scientific
    evaluator_hint: "marker: estimator"
    pass_when: |
      For a dihedral with no analog and no generic wildcard term, the estimate
      is a near-zero barrier (|k| ≤ a small epsilon defined in the test) with a
      high penalty; when a multi-periodicity generic term exists, all of its
      periodicity/k/phase entries are copied as one group.
    status: pending
  - id: ac-013
    summary: opt-in composition injection wired into all three typifiers + passthrough
    type: code
    evaluator_hint: "marker: estimator"
    pass_when: |
      ForceFieldBondTypifier/AngleTypifier/DihedralTypifier accept an optional
      estimator stored as self._estimator (default None), invoke it only in the
      `best_X is None and not self.strict` branch, and ForceFieldTypifier
      passes its estimator argument through to the three sub-typifiers.
    status: pending
  - id: ac-014
    summary: parmchk2 gold-standard cross-validation within tolerance
    type: scientific
    evaluator_hint: "marker: external; AmberTools25 parmchk2 frcmod"
    pass_when: |
      For a molecule with GAFF missing terms, molpy estimated parameters agree
      with the parmchk2 frcmod values within bond r0 atol 0.02 Å, angle θ0
      atol 3°, and force constants rtol 0.10. Marked @pytest.mark.external.
    status: pending
  - id: ac-015
    summary: lint, type check, and local suite clean
    type: runtime
    pass_when: |
      `ruff format --check src tests`, `ruff check src tests`,
      `ty check src/molpy/`, and `pytest tests/ -m "not external" -v`
      all exit 0.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002 / ac-013**: wiring + 归属 + 调用时机契约——独立模块、可达数据资产、opt-in 组合注入到三个生产兜底分支并经 `ForceFieldTypifier` 透传，精确匹配优先、估计器只在无匹配 not-strict 分支触发。
- **ac-003 / ac-004 / ac-005**: GAFF 经验公式正确性，常数从原文逐字转录、由单测钉死（公式 pin，非拟合容差）。
- **ac-006**: 留一法端到端还原——验证类比级联整体在真实 FF 表上能在物理容差内复原已知项。
- **ac-007 / ac-008 / ac-009**: 方法学约束的可信度——最近类比直接复制（不平均）、溯源四键齐全、CGenFF penalty 分级正确。
- **ac-010**: strict 路径零回归 + 默认关闭零行为变化（用户最关心的不介入保证）。
- **ac-011**: 单位正确性约束（OPLS 目标必须探测同表单位换算，不写裸 AMBER 单位）。
- **ac-012**: 二面角"绝不伪造势垒" + 多重周期整组复制的硬约束。
- **ac-014**: 外部金标准科学验证；需 AmberTools25 conda env，`@pytest.mark.external`。ac-006 与 ac-014 依赖测试期转录 / 准备参考真值（删除项的 ground truth、parmchk2 frcmod），这是这两条可评估的前置条件。
- **ac-015**: 仓库质量闸。
