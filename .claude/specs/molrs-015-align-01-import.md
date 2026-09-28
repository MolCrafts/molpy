---
title: "molrs 0.15 alignment 01 — unblock import: canonical keys, force-field model, dead formatter layer"
status: code-complete
created: 2026-09-27
---

# molrs 0.15 alignment 01 — unblock import: canonical keys, force-field model, dead formatter layer

## Summary

After this link, `import molpy` succeeds against molrs 0.15: the local editable molrs, branch feat/cgsmiles, at or after 481ce765. The link fixes the three eager import-time breaks and deletes the molpy force-field layers that nothing calls:
- The retired keys `Q0` and `SITE` leave `molpy.core.fields`.
- `molpy.core.forcefield` becomes a pure identity re-export of `molrs.ff`. The 23 molpy style subclasses, the `AtomisticForcefield` alias and the `molpy.potential` package are deleted. Force fields are built with `ForceField.def_style(category, name, params)`, and `PotentialCompiler` is re-exported beside `Potentials`.
- The `ForceFieldFormatter` parameter-formatter registry, `LammpsForceFieldFormatter` and the CL&Pol `_format_pair_*` functions are deleted (operator ruling R4). Nothing in `src/` calls them. The native `FieldFormatter` family is untouched.
- `io/forcefield/_rb_opls.py` is deleted with its test. It duplicates the native RB→OPLS conversion in `read_forcefield_xml`, and nothing calls it.

This link records the chain's molrs preconditions (see Design) and checks P0–P3. It is the first link and depends on no other link. Some things still break until later links fix them:
- `molpy.builder` and `molpy.typifier` still fail to import (the deferred builder link and 06).
- Moltemplate and the LAMMPS FF writers still fail at runtime (the deferred moltemplate and LAMMPS links).

## Design

### Preconditions (one list for the whole chain)

All molrs references in this chain mean the local editable molrs, branch feat/cgsmiles, at or after 481ce765.

- **P0 — the law-16 exception is recorded.** `.claude/notes/law.md` § VII, "Recorded exceptions → molrs 0.15 co-development pin (2026-09-27)", holds the operator's ruling R1. The operator recorded it before this link. No link in this chain writes it, and the deferred close link ac-004 only checks that it is there.
- **P1** — `molrs.ff` file I/O accepts `str | os.PathLike`. This covers the LAMMPS, XML, GROMACS and AMBER force-field `read_*` / `write_*` functions.
- **P2** — the native LAMMPS FF writer (`write_lammps_forcefield`, `write_lammps_forcefield_str`) raises for a pair style whose coefficients it cannot emit (`thole`, `coul/tt`).
- **P3 — one home for LAMMPS coefficient → native parameters** (landed in 481ce765). `molrs.ff.lammps_coeff_params(category, style, values, units="real") -> dict[str, float]`, where `values` are the raw coefficient tokens (`Sequence[str]`). It is the same conversion the native LAMMPS FF reader uses:
  - `k = 2K` for harmonic bond, angle and improper; degrees → radians;
  - dihedral `opls` (k1..k4), `harmonic` (k, sign, periodicity), `fourier` (m K n d …), `charmm` (k, periodicity, phase, w) and `multi/harmonic` (a1..a5);
  - every pair style spelled `lj/cut…` (epsilon, sigma).

  It raises `ValueError` ("<category> <style>: unsupported LAMMPS <category> style `<style>`") for any other kernel, such as bond `morse` or improper `cvff`. It also raises for a missing or non-numeric token and for unknown `units`. It ignores tokens beyond a kernel's arity. That native gap is open in molrs `.claude/notes/notes.md` (2026-09-27) and is cross-referenced in the deferred close link.
- **P4 — the inverse, native parameters → LAMMPS coefficients** (landed in molrs 0762fc9c; needed only by the deferred moltemplate link). `molrs.ff.lammps_coeff_values(category, style, params, units="real") -> list[float]` is taken from the native LAMMPS FF writer's per-type coefficient rendering (`K = k/2`, radians → degrees) and covers the same kernels as P3. `lammps_coeff_params(c, s, [repr(v) for v in lammps_coeff_values(c, s, p)]) == p` holds for every supported kernel. Link 03's first task checks it. It is not checked here, because links 01–02 do not need it.

Task 1 checks P0–P3 with a one-off script that runs from a scratch directory outside the repository, not in the suite. If any check fails, the implementer stops and reports. There is no molpy-side workaround. Emitting Thole and coul/tt coefficients is a molrs ask, recorded in the deferred close link.

### `src/molpy/core/fields.py`
- Delete `Q0` (`:70`) and `SITE` (`:97`), together with the "assembly field" docstring text (`:15-20`) and block (`:88-97`). molrs 0.15 retired both keys.
- Delete the whole `ForceFieldFormatter` section (`:99-239`):
  - `_BASE_STYLE_CATEGORIES`, `_STYLE_IDENTITY_CACHE`, `_style_class_identity`;
  - class `ForceFieldFormatter`: the `_param_formatters` registry, `register_param_formatter`, `_resolve_formatter`, `format_params`;
  - the now-unused `Callable` import.
- `__all__` (`:242-252`) loses `ForceFieldFormatter`, `Q0` and `SITE`.
- The module docstring says that canonical names and the `FieldFormatter` family are native re-exports and that molpy adds nothing here.

### `src/molpy/core/forcefield.py`
Becomes identity re-exports of `molrs.ff` only:
- `ForceField` and `Parameters`;
- `Style`, `AtomStyle`, `BondStyle`, `AngleStyle`, `DihedralStyle`, `ImproperStyle`, `PairStyle`;
- `Type`, `AtomType`, `BondType`, `AngleType`, `DihedralType`, `ImproperType`, `PairType`;
- `PotentialCompiler`.

It has no classes and no aliases. The docstring says energy and forces come from `PotentialCompiler(ff).compile(frame)`.

### `src/molpy/__init__.py` and `src/molpy/core/__init__.py`
- Drop every `*Style` subclass name from the imports and from `__all__`. That includes the four styles molrs deleted (`BondHarmonicStyle`, `AngleHarmonicStyle`, `DihedralOPLSStyle`, `PairCoulLongStyle`) and `AtomisticForcefield`.
- The root gains `PotentialCompiler` in the `core.forcefield` band, listed in `__all__` next to `Potentials`.

### Delete `src/molpy/potential/`
Six `__init__.py` files. The package was neither lazy nor in `__all__`. `Potentials` stays at `mp.Potentials` and `mp.md.Potentials`. The task removes the whole directory, including its ignored `__pycache__/` folders. A leftover directory would still import as an empty namespace package.

### `src/molpy/io/forcefield/lammps.py`
- Delete `_format_pair_thole` and `_format_pair_coul_tt` (`:17-30`), `LammpsForceFieldFormatter` (`:33-49`) and the imports at `:13-14`. With them go the point-of-use defaults (`alpha` → `0.0`, `a_thole` → `2.6`, …) that architecture.md § 5 forbids.
- `LAMMPSForceFieldWriter` stays until the deferred LAMMPS link deletes it, and the module with it.
- The module docstring no longer claims parameter formatters.

### Delete `src/molpy/io/forcefield/_rb_opls.py` and `tests/test_potential/test_dihedral_opls_rb_conversion.py`
The test is deleted, not moved. Its "used by the XML/LAMMPS I/O" docstring is false, and the native reader owns RB→OPLS. The task removes `tests/test_potential/` as a whole directory.

### `tests/test_engine/test_openmm.py:9, :37-40`
The `empty_forcefield` fixture imports `AtomisticForcefield`, which this link deletes. It becomes `ForceField("test")`, imported from `molpy.core.forcefield`. The alias was the same class, so the tests are unchanged.

### `src/molpy/optimize/__init__.py:5`
The docstring becomes `PotentialCompiler(forcefield).compile(frame)`.

### Pin (law 16)
- `pyproject.toml:34` reads `molcrafts-molrs>=0.15.0,<0.16` (introduced in 60a3455). This link does not edit it.
- molrs v0.15.0 is not published. The operator ruled (R1, 2026-09-27: "use local editable version first!") that this chain is developed against the local editable molrs. That exception is P0, already in § VII. molpy is not tagged or released before molrs v0.15.0 is tagged and published.

### Reuse decision

- **reuse `molrs.ff.PotentialCompiler`** — identity re-export on the root via `core/forcefield.py`.
- **reuse `molrs.ff.ForceField.def_style`** — replaces the 23 subclasses, which are deleted, not generalized.
- **reuse `molrs.ff.Potentials`** — already at `mp.Potentials` and `mp.md.Potentials`; `molpy.potential` is deleted.
- **reuse the native RB→OPLS conversion in `read_forcefield_xml`** — `_rb_opls.py` is deleted.
- **reuse the native `FieldFormatter` family** — unchanged.
- **delete `ForceFieldFormatter`, `LammpsForceFieldFormatter` and `_format_pair_*`** — an extension point with no caller (law 3). Emitting Thole and coul/tt coefficients belongs to molrs (P2, plus the the deferred close link ask).
- **delete alias `AtomisticForcefield`** (`core/forcefield.py:43`); its one test user is ported.
- **new: none.**

### Laws

- **Law 1:** one force-field model, owned by molrs.
- **Law 3:** no extension point without a caller.
- **Law 7:** no convenience classes over `def_style`.
- **Law 9:** the LAMMPS coefficient convention has one home (P3, P4).
- **Law 13:** identity re-exports only.
- **Law 14:** the `alpha=0.0` guessed default leaves with its function.
- **Law 16:** the pin exception is P0, recorded in § VII by the operator.
- **Law 17:** tests of deleted and re-exported code are deleted, including the native-parity key test.

## Files to create or modify

- src/molpy/core/fields.py
- src/molpy/core/forcefield.py
- src/molpy/core/__init__.py
- src/molpy/__init__.py
- src/molpy/optimize/__init__.py
- src/molpy/io/forcefield/lammps.py
- src/molpy/io/forcefield/_rb_opls.py (delete)
- src/molpy/potential/__init__.py (delete)
- src/molpy/potential/bond/__init__.py (delete)
- src/molpy/potential/angle/__init__.py (delete)
- src/molpy/potential/dihedral/__init__.py (delete)
- src/molpy/potential/improper/__init__.py (delete)
- src/molpy/potential/pair/__init__.py (delete)
- tests/test_core/test_fields.py
- tests/test_engine/test_openmm.py
- tests/test_io/test_forcefield/test_clpol_formatters.py (delete)
- tests/test_potential/test_dihedral_opls_rb_conversion.py (delete)
- tests/test_potential/test_pair/test_thole.py (delete)
- tests/test_potential/test_pair/test_tang_toennies.py (delete)
- tests/test_core/test_forcefield.py (delete)
- tests/test_core/test_forcefield_editing.py (delete)
- tests/test_md/test_driver.py (delete)

## Tasks

- [x] Verify preconditions P0–P3 with a one-off script run from a scratch directory outside the repo (not a suite test):
  - P0: `.claude/notes/law.md` has "### molrs 0.15 co-development pin (2026-09-27)" under "## Recorded exceptions";
  - P1: `molrs.ff.read_lammps_forcefield(pathlib.Path(p))` and `molrs.ff.write_lammps_forcefield(pathlib.Path(q), ff, frame)` accept a `Path`;
  - P2: `molrs.ff.write_lammps_forcefield_str(ff, frame)` raises for a force field that holds one `pair thole` type used by the frame;
  - P3: `molrs.ff.lammps_coeff_params("bond", "harmonic", ["268", "1.529"]) == {"k": 536.0, "r0": 1.529}`; `("dihedral", "charmm", ["-0.5", "1", "-180", "0.0"])` gives k −0.5, periodicity 1, phase `math.radians(-180)`, w 0.0; `("pair", "lj/cut", ["0.066", "3.5"])` gives epsilon 0.066 and sigma 3.5; `("improper", "cvff", ["2.5", "-1", "2"])` and `("bond", "morse", ["1", "2", "3"])` raise `ValueError`.

  Stop and report if any check fails.
- [x] Delete `Q0`, `SITE`, the assembly-field text and the whole `ForceFieldFormatter` section (with its `__all__` entries) from src/molpy/core/fields.py, and rewrite the module docstring. In tests/test_core/test_fields.py, delete:
  - `TestForceFieldFormatter`;
  - `test_site_is_a_molpy_owned_column`;
  - `test_every_molrs_key_is_a_named_string_constant` (a native-parity check, law 17);
  - the `ForceFieldFormatter`, `BondHarmonicStyle` and `BondStyle` imports.
- [x] Delete `_format_pair_thole`, `_format_pair_coul_tt`, `LammpsForceFieldFormatter` and their imports from src/molpy/io/forcefield/lammps.py. Delete tests/test_io/test_forcefield/test_clpol_formatters.py.
- [x] Delete src/molpy/io/forcefield/_rb_opls.py and tests/test_potential/test_dihedral_opls_rb_conversion.py.
- [x] Rewrite src/molpy/core/forcefield.py as the `molrs.ff` identity re-export list, including `PotentialCompiler`. Delete the 23 subclasses and `AtomisticForcefield`. Port the `empty_forcefield` fixture in tests/test_engine/test_openmm.py (`:9`, `:37-40`) to `ForceField("test")` from `molpy.core.forcefield`.
- [x] Update the re-export bands in src/molpy/__init__.py and src/molpy/core/__init__.py: drop the style names and `AtomisticForcefield`, and add `PotentialCompiler` next to `Potentials`. Fix the docstring in src/molpy/optimize/__init__.py.
- [x] Delete src/molpy/potential/ as a whole directory: `git rm` the six `__init__.py` files, then remove the directory and its `__pycache__/` folders. Delete the tests of deleted or re-exported code, then remove tests/test_potential/ as a whole directory:
  - tests/test_potential/test_pair/test_thole.py;
  - tests/test_potential/test_pair/test_tang_toennies.py;
  - tests/test_core/test_forcefield.py;
  - tests/test_core/test_forcefield_editing.py;
  - tests/test_md/test_driver.py.
- [x] Verify that `uv run --extra dev python -c "import molpy"` exits 0 and that the pyproject.toml pin line is unchanged.
- [x] Run tests/test_core/test_fields.py and tests/test_engine/test_openmm.py via test_single. The full check and suite are deferred to the deferred close link by operator ruling.

## Testing strategy

- **No new unit tests.** This link deletes code and changes no molpy-owned behaviour. Acceptance checks the import. The remaining `tests/test_core/test_fields.py::TestFieldFormatter` stays green; it is routed in the deferred close link as a test of a native re-export.
- **tests/test_engine/test_openmm.py** keeps its subject (`OpenMMEngine`). Only its fixture's force-field constructor changes, from the deleted alias to the class it aliased.
- **P0–P3** are one-off evaluator checks, not suite tests (law 17: no native-parity check in the suite). P4 is checked the same way by the deferred moltemplate link.
- **No molpy tests for re-exports** (testing.md:19-21). That covers `core.forcefield`, `potential` and `md`, so their old tests are deleted.
- **No domain validation:** no physics change. **No regression example:** law 17 forbids `regressions/`.

## Out of scope

- `LAMMPSForceFieldWriter` and the LAMMPS writers, and moltemplate (deferred links; see the notes entry written by link 03).
- `tests/test_md/{test_units,test_integrators,test_neighbors,test_maxwell}.py` and `TestFieldFormatter`, which test pure re-exports. They are routed as follow-ups in the deferred close link; this change does not break them.
- Emitting Thole and coul/tt coefficients (molrs ask, the deferred close link).
- Checking P4 (done by the deferred moltemplate link).
