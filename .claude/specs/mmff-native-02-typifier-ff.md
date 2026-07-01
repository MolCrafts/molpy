---
title: Native RDKit-free MMFF94 path via public MMFFTypifier + ForceField classes
status: approved
created: 2026-06-16
---

# Native RDKit-free MMFF94 path via public MMFFTypifier + ForceField classes

## Summary

This spec adds a native, RDKit-free MMFF94/MMFF94s path to molpy by exposing two
public classes that wrap the molrs MMFF backend: a new `MMFFTypifier` (sibling of
`OplsTypifier`/`GaffTypifier` in `molpy.typifier`) and the MMFF94 force field
surfaced through molpy's existing public `ForceField` class. A user typifies an
`Atomistic` molecule into an assembly-complete `molrs.Frame`, retrieves the MMFF
`ForceField`, and runs geometry optimization with spec 01's Frame-based
`molpy.optimize.LBFGS` + `ForceFieldPotential` — no MMFF-specific optimizer and no
factory functions anywhere. The existing RDKit `OptimizeGeometry(forcefield="MMFF94")`
adapter stays untouched as the external peer used for validation parity. This is
wiring of existing molrs MMFF + optimizer machinery; it introduces no new
force-field math.

## Domain basis

MMFF94 is the Merck Molecular Force Field as published in Halgren, T. A.,
*J. Comput. Chem.* **17**, 490-519 (1996), DOI 10.1002/(SICI)1096-987X(199604)17:5/6<490::AID-JCC1>3.0.CO;2-P
(cited in the `molrs.MMFFTypifier` docstring). The static "s" variant is Halgren,
T. A., *J. Comput. Chem.* **20**, 720-729 (1999), DOI 10.1002/(SICI)1096-987X(199905)20:7<720::AID-JCC7>3.0.CO;2-X.
The molrs MMFF implementation is RDKit-validated; this spec adds no equations and
defines no new parameters. Units follow molrs MMFF: energy in kcal/mol, length in
angstrom. Validation is twofold: (1) energy must be non-increasing along the LBFGS
trajectory (a minimizer invariant), and (2) native molrs-MMFF optimized energy and
geometry must agree with RDKit MMFF94 (`OptimizeGeometry(forcefield="MMFF94")`)
within tolerance on a small molecule set.

## Design

Depends on `mmff-native-01-frame-optimizer` (provides `molpy.optimize.LBFGS.run(frame, fmax, steps, *, inplace=True)`
and `ForceFieldPotential(ff).calc_energy/calc_forces(frame)` via `ff.to_potentials(frame)`).
This spec must not re-implement or alter that optimizer — it consumes it.

New public class `molpy.typifier.MMFFTypifier` (`src/molpy/typifier/mmff.py`),
a sibling of `OplsTypifier`/`GaffTypifier`/`ClpTypifier`, wrapping `molrs.MMFFTypifier`:

- `__init__(self, variant: str = "MMFF94")` — accepts `"MMFF94"` or `"MMFF94s"`.
  Unknown variant raises `ValueError` fail-fast (precedent message style:
  "MMFF parameters may not be available for this molecule"). `import molrs` at
  module top — molrs is the mandatory backend, NOT an optional `try/except` adapter.
- `typify(self, mol: Atomistic) -> molrs.Frame` — returns an **assembly-complete**
  (`to_potentials`-ready) frame: the stretch-bend `r0_ij` merge (`merge_stbn_r0`)
  MUST already be applied so the spec-01 path `ForceFieldPotential -> ff.to_potentials(frame)`
  does not raise `ValueError('mmff_stbn: missing "r0_ij" column ...')`. An
  MMFF-untypeable molecule raises `ValueError`.
- `forcefield(self) -> ForceField` — returns the MMFF94 force field exposed through
  molpy's public `ForceField` class (re-exported at `molpy.ForceField`, defined in
  `molpy.core.forcefield`).
- Direct instantiation only; no `optimize_mmff`, no `build_*`, no module-level
  factory function.

Stretch-bend assembly resolution (the critical blocker): plain `molrs.MMFFTypifier.typify`
yields a frame on which `to_potentials` raises because `merge_stbn_r0` is not run,
whereas `molrs.MMFFTypifier.build` / `molrs.build_mmff_potentials` run it internally.

- **Primary (preferred): cross-repo molrs fix** — make `molrs.MMFFTypifier.typify`
  itself emit a `to_potentials`-ready frame (run `merge_stbn_r0` during typify).
  This is existing molrs assembly logic, not new FF math. Tracked as an explicit
  cross-repo prerequisite task below.
- **Fallback (in molpy, if the molrs fix has not landed): molpy's
  `MMFFTypifier.typify` triggers the merge before returning** — e.g. by routing
  through the assembly that `build`/`build_mmff_potentials` perform so the returned
  frame carries `r0_ij`. The chosen behavior is: ship the fallback in molpy so the
  spec is self-contained, and remove it once the primary molrs fix lands.

Geometry optimization uses **no new class and no factory**. The documented usage is:

    typifier = MMFFTypifier(variant="MMFF94")
    frame = typifier.typify(mol)                 # assembly-complete molrs.Frame
    ff = typifier.forcefield()                   # molpy public ForceField
    result = LBFGS(ForceFieldPotential(ff)).run(frame, fmax=..., steps=...)

This is spec 01's Frame-based optimizer verbatim. The RDKit `OptimizeGeometry`
adapter in `adapter/rdkit.py` is the external peer for parity testing and is
structurally mirrored in nothing (no factory, no shared base) and left untouched.

## Files to create or modify

- `src/molpy/typifier/mmff.py` (new) — `MMFFTypifier` public class wrapping `molrs.MMFFTypifier`.
- `src/molpy/typifier/__init__.py` — export `MMFFTypifier` alongside `OplsTypifier`/`GaffTypifier`/`ClpTypifier`.
- `tests/test_typifier/test_mmff.py` (new) — typify + forcefield + variant + error tests.
- `tests/test_optimize/test_mmff_optimization.py` (new) — Frame-based LBFGS run + RDKit parity (external).
- `docs/` MMFF usage note (new) — native MMFF path documentation (no factory; classes + methods).
- molrs (cross-repo, prerequisite): `MMFFTypifier.typify` runs `merge_stbn_r0` so its frame is `to_potentials`-ready.

## Tasks

- [ ] (cross-repo prerequisite) Land molrs fix so `molrs.MMFFTypifier.typify` yields a `to_potentials`-ready frame (run `merge_stbn_r0`); until merged, molpy uses the fallback below
- [ ] Write failing tests for MMFFTypifier in tests/test_typifier/test_mmff.py (typify returns to_potentials-ready molrs.Frame; forcefield() returns molpy ForceField; unknown variant + untypeable molecule raise ValueError)
- [ ] Implement MMFFTypifier in src/molpy/typifier/mmff.py (variant init, typify with merge_stbn_r0 fallback, forcefield(), top-level import molrs, fail-fast ValueError; no factory)
- [ ] Export MMFFTypifier from src/molpy/typifier/__init__.py as sibling of OplsTypifier/GaffTypifier/ClpTypifier
- [ ] Add Google-style docstrings per docs.style with kcal/mol + angstrom units and Halgren 1996/1999 DOIs on MMFFTypifier and its methods
- [ ] Write failing test for Frame-based LBFGS MMFF run in tests/test_optimize/test_mmff_optimization.py (energy non-increasing along trajectory using spec-01 LBFGS + ForceFieldPotential)
- [ ] Write failing @pytest.mark.external test asserting molrs-MMFF vs RDKit OptimizeGeometry("MMFF94") energy + geometry parity within tolerance on a small molecule set
- [ ] Add docs MMFF usage note showing MMFFTypifier().typify -> forcefield() -> LBFGS(ForceFieldPotential(ff)).run(frame); no factory
- [ ] Run full check + test suite

## Testing strategy

- Happy path: `MMFFTypifier(variant="MMFF94").typify(mol)` returns a `molrs.Frame`
  that `ff.to_potentials(frame)` accepts without raising (i.e. carries the merged
  `r0_ij` column); `forcefield()` returns a molpy `ForceField`.
- Edge cases: unknown `variant` raises `ValueError`; MMFF-untypeable molecule raises
  `ValueError` with the precedent message; `"MMFF94s"` variant typifies successfully.
- Domain validation — invariant: along an `LBFGS().run(frame)` trajectory the
  potential energy is monotonically non-increasing (runtime check on a distorted
  small molecule). Tolerance: each step's energy <= previous step's energy within a
  small numerical slack (e.g. 1e-6 kcal/mol).
- Domain validation — parity (`@pytest.mark.external`): for each molecule in a small
  set, native molrs-MMFF optimized final energy matches RDKit
  `OptimizeGeometry(forcefield="MMFF94")` within an energy tolerance and the optimized
  geometry matches within an RMSD tolerance (in angstrom). Marked external because it
  requires RDKit.
- All non-external tests must run under `pytest tests/ -m "not external"`.

## Out of scope

- New force-field math, MMFF parameter tables, or any change to molrs MMFF physics
  (this is wiring only; molrs MMFF is the validated source of truth).
- Any factory / builder function (`optimize_mmff`, `build_mmff`, etc.) — explicitly
  forbidden; classes + direct instantiation only.
- A new MMFF-specific optimizer — spec 01's `LBFGS` Frame path is reused unchanged.
- Changes to `adapter/rdkit.py` `OptimizeGeometry` — it stays as the untouched
  external peer.
- UFF or other RDKit force fields beyond MMFF94/MMFF94s.
