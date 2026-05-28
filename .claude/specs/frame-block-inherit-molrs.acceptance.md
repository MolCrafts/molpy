---
slug: frame-block-inherit-molrs
spec: frame-block-inherit-molrs.md
created: 2026-05-28
---

# Acceptance — frame-block-inherit-molrs

Binding "done" contract. `/mol:impl` must satisfy every criterion below
before deleting the spec.

## Criteria

```yaml
- id: prereq-molrs-subclass-landed
  type: code
  summary: molrs Frame / Block already subclassable
  evaluator_hint: ""
  pass_when: |
    python -c "
    import molrs
    class _F(molrs.Frame): pass
    class _B(molrs.Block): pass
    _F(); _B()
    "
  status: verified
  last_checked: 2026-05-28

- id: molpy-block-is-a-molrs-block
  type: code
  summary: molpy.Block inherits from molrs.Block
  evaluator_hint: ""
  pass_when: |
    pytest tests/test_core/test_block_inheritance.py::test_molpy_block_is_a_molrs_block -v
  status: verified
  last_checked: 2026-05-28

- id: molpy-frame-is-a-molrs-frame
  type: code
  summary: molpy.Frame inherits from molrs.Frame
  evaluator_hint: ""
  pass_when: |
    pytest tests/test_core/test_frame_inheritance.py::test_molpy_frame_is_a_molrs_frame -v
  status: verified
  last_checked: 2026-05-28

- id: frame-directly-accepted-by-molrs
  type: code
  summary: molpy.Frame accepted by molrs APIs without conversion
  evaluator_hint: ""
  pass_when: |
    pytest tests/test_core/test_frame_inheritance.py::test_frame_directly_accepted_by_molrs_api -v
  status: verified
  last_checked: 2026-05-28

- id: to_molrs-deleted
  type: code
  summary: no .to_molrs() or self._inner references remain
  evaluator_hint: ""
  pass_when: |
    ! grep -rnE '\.to_molrs|self\._inner' src/molpy/
  status: verified
  last_checked: 2026-05-28

- id: rdf-bridge-removed
  type: code
  summary: compute/rdf.py no longer constructs an interim molrs.Frame
  evaluator_hint: ""
  pass_when: |
    ! grep -n 'mf = molrs.Frame()' src/molpy/compute/rdf.py
  status: verified
  last_checked: 2026-05-28

- id: python-only-state-isolated
  type: code
  summary: metadata survives a roundtrip through a molrs API call
  evaluator_hint: ""
  pass_when: |
    pytest tests/test_core/test_frame_inheritance.py::test_metadata_survives_molrs_roundtrip -v
  status: verified
  last_checked: 2026-05-28

- id: reader-upgrade-applied
  type: code
  summary: io/ experimental readers return molpy.Frame (verified via box round-trip)
  evaluator_hint: ""
  pass_when: |
    pytest tests/test_io/test_data/test_xyz.py -v -k "test_extended_xyz_format and experimental"
    pytest tests/test_compute/test_gro_experimental.py -v
  status: verified
  last_checked: 2026-05-28

- id: full-suite-green
  type: code
  summary: full local test suite passes
  evaluator_hint: ""
  pass_when: |
    pytest tests/ -m "not external" -v
  status: verified
  last_checked: 2026-05-28

- id: lint-clean
  type: code
  summary: ruff format + ruff check + ty all clean
  evaluator_hint: ""
  pass_when: |
    ruff format --check src tests
    ruff check src tests
    ty check src/molpy/
  status: verified
  last_checked: 2026-05-28
---
