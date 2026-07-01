---
slug: mmff-native-01-frame-optimizer
criteria:
  - id: ac-001
    summary: Optimizer.run accepts a molrs.Frame and returns OptimizationResult[Frame]
    type: code
    pass_when: |
      Optimizer.run in src/molpy/optimize/base.py has signature
      run(self, frame: molrs.Frame, fmax=0.01, steps=1000, *, inplace=True)
      and OptimizationResult exposes a `frame: molrs.Frame` field (no `structure`
      field); ty check src/molpy/ passes.
    status: pending
  - id: ac-002
    summary: Bridge reads/writes Frame coordinate columns, not structure.atoms
    type: code
    pass_when: |
      get_positions/set_positions in base.py operate on frame["atoms"]["x"|"y"|"z"]
      and get_energy_and_forces calls self.potential.calc_energy(frame) /
      calc_forces(frame) directly; no occurrence of `.to_frame()` or `.atoms`
      remains in src/molpy/optimize/base.py or lbfgs.py.
    status: pending
  - id: ac-003
    summary: LBFGS step mutates the Frame in place with unchanged algorithm
    type: code
    pass_when: |
      LBFGS.step(self, frame) updates the Frame's coordinate columns in place and
      the two-loop recursion (_lbfgs_direction), step-size control
      (_compute_step_size), curvature condition (sy > 1e-10), and memory trimming
      are byte-identical in logic to the pre-refactor lbfgs.py.
    status: pending
  - id: ac-004
    summary: No factory symbol added to the optimize package
    type: code
    pass_when: |
      src/molpy/optimize/__init__.py __all__ contains exactly
      ["Optimizer", "OptimizationResult", "LBFGS", "ForceFieldPotential"] and no
      factory/helper constructor function is added anywhere under
      src/molpy/optimize/.
    status: pending
  - id: ac-005
    summary: inplace flag controls Frame mutation correctly
    type: runtime
    pass_when: |
      In tests/test_optimize/test_optimizer_frame.py, run(frame, inplace=True)
      returns result.frame is frame with mutated coordinate columns, and
      run(frame, inplace=False) leaves the input Frame's coordinates unchanged
      while result.frame is a distinct Frame; the test passes under
      pytest -m "not external".
    status: pending
  - id: ac-006
    summary: TIP3P water regression parity on optimized geometry
    type: scientific
    pass_when: |
      tests/test_optimize/test_tip3p_water.py passes a Frame via struct.to_frame(),
      reads positions from result.frame, and the optimized O-H bond lengths are
      within 0.01 nm of 0.09572 nm and the H-O-H angle within 0.15 rad of
      1.82421813418 rad, with result.converged or nsteps == 500 — identical
      assertions to the pre-refactor test.
    status: pending
  - id: ac-007
    summary: Changed public API carries Google-style docstrings with units
    type: docs
    pass_when: |
      run, LBFGS.step, and OptimizationResult have Google-style docstrings naming
      the molrs.Frame parameter/field and the energy/force units, and the
      base.py / lbfgs.py runnable examples call opt.run(frame, ...).
    status: pending
  - id: ac-008
    summary: Full quality gate green
    type: runtime
    pass_when: |
      ruff format src tests (clean), ruff check src tests, ty check src/molpy/,
      and pytest tests/ -m "not external" all exit 0.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002 / ac-003 (code)** — the structural refactor: Frame-typed
  signatures, Frame coordinate bridge, removal of the `to_frame()`/`.atoms`
  writeback path, and an unchanged L-BFGS core.
- **ac-004 (code)** — guards the "no factory" design rule.
- **ac-005 (runtime)** — new behavioural test pinning in-place vs copy Frame semantics.
- **ac-006 (scientific)** — behaviour-preservation: the migrated TIP3P test reproduces
  the published geometry within the original tolerances.
- **ac-007 (docs)** — docstring/example migration to the Frame API.
- **ac-008 (runtime)** — the full format/lint/type/test gate.
