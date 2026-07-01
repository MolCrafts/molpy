---
title: Frame-native geometry optimizer
status: approved
created: 2026-06-16
---

# Frame-native geometry optimizer

## Summary
The geometry optimizer in `molpy.optimize` is refactored to operate directly on a
`molrs.Frame` — the universal coordinate-plus-topology container — instead of an
Atomistic-style structure that it internally converts via `.to_frame()` and writes
back through `.atoms`. After this change, `Optimizer.run(frame, ...)` reads and
writes the Frame's coordinate columns in place and returns an
`OptimizationResult[molrs.Frame]` carrying the optimized Frame. This is a strictly
behaviour-preserving refactor: the L-BFGS algorithm, step-size control, and
convergence results are unchanged, and `ForceFieldPotential` (already Frame-based)
is untouched. The result is a unified stack where both the potential and the
optimizer consume the same `Frame` object, honouring the design rule "the
optimizer's input is always a Frame."

## Domain basis
No new physics is introduced. The minimised objective is the molrs force field's
potential energy evaluated through `ForceFieldPotential`; forces are `F = -∇E`. The
L-BFGS quasi-Newton update (two-loop recursion with limited memory and a maxstep
displacement cap) is preserved bit-for-bit from the current implementation in
`src/molpy/optimize/lbfgs.py`. The regression target is the existing TIP3P water
case (GROMACS nm/kJ units): O-H equilibrium bond length 0.09572 nm and H-O-H angle
104.52° (1.82421813418 rad), per the TIP3P parameterisation in the existing test
fixture. Equilibrium sanity: at the minimum `F(r_eq) ≈ 0` and the geometry recovers
the published TIP3P bond length and angle. The acceptance bar is parity with the
prior structure-based optimisation, not re-derivation of these values.

## Design
- `Optimizer.run` signature changes from `run(self, structure: S, ...)` to
  `run(self, frame: molrs.Frame, fmax=0.01, steps=1000, *, inplace=True) ->
  OptimizationResult[molrs.Frame]`. `inplace=False` operates on a Frame copy
  (frame-level copy of coordinate state) so the caller's Frame is untouched;
  `inplace=True` mutates the passed Frame's coordinate columns directly.
- The generic `S = TypeVar("S")` over arbitrary structures is replaced: the
  optimizer and `OptimizationResult` are specialised to `molrs.Frame`. The
  `Generic[S]` machinery is dropped (no remaining structure abstraction to be
  generic over).
- Bridge methods operate on the Frame's `"atoms"` block coordinate columns:
  - `get_positions(frame)` → `(N, 3)` array via
    `np.column_stack([frame["atoms"]["x"], ["y"], ["z"]])`.
  - `set_positions(frame, positions)` → writes `frame["atoms"]["x"|"y"|"z"]` from
    the `(N, 3)` array (reshaping a flat vector first).
  - `get_energy_and_forces(frame)` → calls `self.potential.calc_energy(frame)` and
    `self.potential.calc_forces(frame)` directly; the `structure.to_frame()`
    conversion is removed.
  - `get_energy(frame)` / `get_forces(frame)` delegate to the pair method.
- `LBFGS.step(frame)` runs one L-BFGS step that updates the Frame's coordinate
  columns in place. The math (`_lbfgs_direction`, `_compute_step_size`, history
  bookkeeping, curvature condition) is unchanged; only the state container changes
  from structure to Frame. `_reset_state(id(frame))` keys LBFGS history on the
  Frame's identity.
- `OptimizationResult` field `structure: S` becomes `frame: molrs.Frame`; the
  other fields (`energy`, `fmax`, `nsteps`, `converged`, `reason`) are unchanged.
- `PotentialLike` protocol is unchanged (already `calc_energy`/`calc_forces`).
- No factory functions are added; callers instantiate `LBFGS(potential, ...)`
  directly and pass a Frame. Callers holding an Atomistic call `atomistic.to_frame()`
  themselves before `run`.
- The `entity_type` constructor parameter is retained for signature compatibility
  but is no longer used to select positions (the Frame's atom block is the single
  source of coordinates); keep it to avoid widening the diff and breaking the
  public constructor.

## Files to create or modify
- `src/molpy/optimize/base.py` — re-type `Optimizer`/`OptimizationResult` to
  `molrs.Frame`; rewrite `get_positions`/`set_positions`/`get_energy_and_forces`
  to read/write Frame coordinate columns; change `run` signature and copy path;
  update docstrings and the `>>> result = opt.run(...)` example.
- `src/molpy/optimize/lbfgs.py` — re-type `step`/class to Frame; keep algorithm
  math unchanged; update docstrings and example to the Frame API.
- `tests/test_optimize/test_tip3p_water.py` — pass a Frame (`struct.to_frame()`)
  into `run`; read optimised coordinates from the returned Frame; keep the physics
  assertions (bond length, angle, convergence) identical.

## Tasks
- [ ] Write failing tests for Frame-native `run` and bridge methods in `tests/test_optimize/test_optimizer_frame.py` (new): assert `run` accepts a `molrs.Frame`, mutates its coordinate columns in place when `inplace=True`, leaves the input Frame untouched when `inplace=False`, and returns an `OptimizationResult` whose `frame` is a `molrs.Frame`.
- [ ] Migrate `tests/test_optimize/test_tip3p_water.py` to build a Frame via `struct.to_frame()`, call `opt.run(frame, ...)`, and read optimised positions from the returned `result.frame` while keeping bond-length/angle/convergence assertions identical.
- [ ] Implement Frame coordinate bridge `get_positions`/`set_positions` in `src/molpy/optimize/base.py` against the `"atoms"` block `x`/`y`/`z` columns.
- [ ] Implement Frame-based `get_energy_and_forces`/`get_energy`/`get_forces` in `src/molpy/optimize/base.py` calling the potential on the Frame directly (remove the `structure.to_frame()` path).
- [ ] Implement Frame-typed `run` and `OptimizationResult.frame` in `src/molpy/optimize/base.py`, including the `inplace=False` Frame-copy path; drop the `Generic[S]` structure abstraction.
- [ ] Implement Frame-typed `LBFGS.step` in `src/molpy/optimize/lbfgs.py` updating the Frame's coordinates in place, keeping the L-BFGS math unchanged.
- [ ] Add Google-style docstrings with units to the changed `run`/`step`/`OptimizationResult` and update the `base.py`/`lbfgs.py` examples to the Frame API per docs.style.
- [ ] Verify regression parity on the TIP3P water case: optimised bond length/angle and convergence match the prior structure-based results within the test tolerances.
- [ ] Run full check + test suite (ruff format, ruff check, ty, pytest -m "not external").

## Testing strategy
- Happy path: `run(frame, inplace=True)` on the TIP3P water Frame converges (or
  reaches max steps) and the optimised geometry recovers O-H ≈ 0.09572 nm and
  H-O-H ≈ 1.824 rad within the existing tolerances, read from the returned Frame.
- In-place semantics: with `inplace=True` the input Frame's coordinate columns are
  mutated and `result.frame` is the same Frame; with `inplace=False` the input
  Frame is unchanged and `result.frame` is a distinct optimised Frame.
- Bridge round-trip: `set_positions(frame, get_positions(frame))` is a no-op on the
  Frame's coordinate columns (read/write symmetry, Nx3 ↔ flat).
- Return type: `run` returns `OptimizationResult` with a `molrs.Frame` `frame`
  field; `converged`/`nsteps`/`fmax`/`energy` populated.
- Domain validation (regression parity): the optimised energy and final fmax on the
  TIP3P case match the prior structure-based optimisation to tight tolerance, and at
  the recovered minimum `fmax` is small (forces ≈ 0), confirming behaviour
  preservation rather than a new optimiser.
- Full gate: `ruff format`, `ruff check`, `ty check src/molpy/`, and
  `pytest tests/ -m "not external"` all green.

## Out of scope
- MMFF typifier / native MMFF force field (spec `mmff-native-02-typifier-ff`, which
  depends on this).
- Any change to `ForceFieldPotential` — it stays Frame-based as-is.
- Any Atomistic-input convenience wrapper or factory for `run`; callers convert via
  `atomistic.to_frame()` themselves.
- New optimiser algorithms or changes to the L-BFGS math / step-size control.
- Box/PBC handling in the optimiser (coordinates only; topology and box pass
  through the Frame unchanged).
