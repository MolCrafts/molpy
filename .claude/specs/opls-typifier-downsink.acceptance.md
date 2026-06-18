---
slug: opls-typifier-downsink
criteria:
  - id: ac-001
    summary: OplsTypifier still importable; delegates to molrs
    type: code
    pass_when: |
      `from molpy.typifier import OplsTypifier` succeeds and the implementation
      delegates to molrs.OplsTypifier (no molpy-side Python SMARTS typing /
      _sequence_score logic remains in the call path).
    status: pending
  - id: ac-002
    summary: molrs pinned to a version exposing OplsTypifier
    type: code
    pass_when: |
      pyproject.toml pins molcrafts-molrs to a version whose `molrs` exposes
      `OplsTypifier`, and `python -c "import molrs; molrs.OplsTypifier"` works in
      the dev environment.
    status: pending
  - id: ac-003
    summary: removed Python typification symbols have zero references
    type: code
    pass_when: |
      grep across src/molpy finds no remaining references to the deleted
      symbols (_OplsAtomTypifier, ForceFieldBondTypifier/AngleTypifier/
      DihedralTypifier, _sequence_score, _end_score, _build_type_class_layer).
    status: pending
  - id: ac-004
    summary: public API consumers unaffected (delegate contract)
    type: code
    pass_when: |
      Typifying a known molecule via molpy OplsTypifier returns the same public
      shape (typed Frame / labeled Atomistic) as before the rewire; builder/io
      paths using OplsTypifier still pass.
    status: pending
  - id: ac-005
    summary: parity gate honored before deletion
    type: code
    evaluator_hint: "manual: confirm molrs opls-typifier-03-parity green"
    pass_when: |
      No molpy Python OPLS typification code is deleted until molrs
      opls-typifier-03-parity acceptance is verified (per-atom 100% + params in
      tolerance). This is a sequencing precondition for ac-001/ac-003.
    status: pending
  - id: ac-006
    summary: lint, type check, and local suite clean
    type: runtime
    pass_when: |
      `ruff format --check src tests`, `ruff check src tests`,
      `ty check src/molpy/`, and `pytest tests/ -m "not external" -v`
      all exit 0.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002**: molpy `OplsTypifier` 退成 molrs 薄壳委托 + molrs 依赖 pin（释放链路对齐 `feedback_release_order_and_pinning`）。
- **ac-003**: 删除符号零残留引用（grep-clean）。
- **ac-004**: 公开 API 契约稳定，下游（builder/io）不回归。
- **ac-005**: 删除前 molrs `opls-typifier-03-parity` 必须绿（顺序前置门）。
- **ac-006**: molpy 质量闸（ruff/ty/pytest not-external）。
