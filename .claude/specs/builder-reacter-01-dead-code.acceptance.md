---
slug: builder-reacter-01-dead-code
criteria:
  - id: ac-001
    summary: Exported PolymerBuilder is the core implementation
    type: code
    pass_when: |
      python -c "from molpy.builder.polymer import PolymerBuilder; from molpy.builder.polymer import core; assert PolymerBuilder is core.PolymerBuilder"
      exits 0.
    status: verified
    last_checked: 2026-06-10
  - id: ac-002
    summary: GeometryError derives from AssemblyError (single-source)
    type: code
    pass_when: |
      python -c "from molpy.builder.polymer.errors import GeometryError, AssemblyError; assert issubclass(GeometryError, AssemblyError)"
      exits 0.
    status: verified
    last_checked: 2026-06-10
  - id: ac-003
    summary: PolymerBuildResult defined in core.py, no polymer_builder back-import
    type: code
    pass_when: |
      ripgrep "from .polymer_builder" src/molpy/builder/polymer/core.py returns no
      matches AND ripgrep "class PolymerBuildResult" src/molpy/builder/polymer/core.py
      returns exactly one match.
    status: verified
    last_checked: 2026-06-10
  - id: ac-004
    summary: All targeted dead files are removed from disk
    type: code
    evaluator_hint: "test -e on each deleted path"
    pass_when: |
      None of these paths exist: src/molpy/builder/polymer/polymer_builder.py,
      growth_kernel.py, stochastic_generator.py, stochastic.py, selectors.py,
      residue_manager.py, ambertools/polymer_amber.py (all under
      src/molpy/builder/polymer/).
    status: verified
    last_checked: 2026-06-10
  - id: ac-005
    summary: No dangling references to deleted symbols in src tree
    type: code
    pass_when: |
      ripgrep -t py "growth_kernel|stochastic_generator|GrowthKernel|ProbabilityTableKernel|StochasticChainGenerator|process_port_markers"
      over src/ returns no matches.
    status: verified
    last_checked: 2026-06-10
  - id: ac-006
    summary: builder/__init__.py false docstring claims removed
    type: code
    pass_when: |
      ripgrep "notebooks/reacter_polymerbuilder_integration|removed in favour"
      src/molpy/builder/__init__.py returns no matches.
    status: verified
    last_checked: 2026-06-10
  - id: ac-007
    summary: Package smoke imports succeed with no broken exports
    type: runtime
    pass_when: |
      python -c "import molpy.builder; import molpy.builder.polymer; import molpy.reacter"
      exits 0.
    status: verified
    last_checked: 2026-06-10
  - id: ac-008
    summary: Regression tests for identity + exception single-source pass
    type: runtime
    evaluator_hint: "pytest node: tests/test_builder/test_polymer_core.py"
    pass_when: |
      pytest tests/test_builder/test_polymer_core.py -v exits 0, including the new
      export-identity and connectors-raise-catchable-as-AssemblyError tests.
    status: verified
    last_checked: 2026-06-10
  - id: ac-009
    summary: Full lint, type, and local test suite green
    type: runtime
    pass_when: |
      ruff format --check src tests && ruff check src tests && ty check src/molpy/
      && pytest tests/ -m "not external" -v all exit 0.
    status: verified
    last_checked: 2026-06-10
---

# Acceptance criteria

- ac-001 — Binds the consolidation: the public `PolymerBuilder` symbol must be the
  identical object as `core.PolymerBuilder`, proving the stale fork is gone and the
  re-export repoints to the live class.
- ac-002 — Binds the exception single-source fix; catches the original divergence
  bug (`GeometryError(Exception)`) that let `except AssemblyError` silently miss.
- ac-003 — Binds the `PolymerBuildResult` move and removal of the `core.py:112`
  back-import.
- ac-004 — Binds the seven file deletions concretely.
- ac-005 — Binds grep-no-references for the dead stochastic subsystem and dead
  selector helper.
- ac-006 — Binds removal of the two false docstring claims in `builder/__init__.py`.
- ac-007 — Binds no-broken-import smoke across builder + reacter.
- ac-008 — Binds the RED-first regression tests landing green.
- ac-009 — Binds the zero-behavior-change promise: full lint/type/test suite green.
