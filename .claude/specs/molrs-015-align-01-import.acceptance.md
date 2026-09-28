---
slug: molrs-015-align-01-import
criteria:
  - id: ac-001
    summary: import molpy succeeds against molrs feat/cgsmiles at or after 481ce765
    type: runtime
    pass_when: |
      `uv run --extra dev python -c "import molpy"` exits 0 from the repo root, with the editable molrs at or after 481ce765.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-002
    summary: preconditions P0 (§ VII entry), P1 (PathLike), P2 (thole raises), P3 (coeff params) hold
    type: runtime
    pass_when: |
      A one-off script run outside the repo shows:
      P0: .claude/notes/law.md has "### molrs 0.15 co-development pin (2026-09-27)" under "## Recorded exceptions";
      P1: `molrs.ff.read_lammps_forcefield` and `molrs.ff.write_lammps_forcefield` accept a `pathlib.Path` without `TypeError`;
      P2: `molrs.ff.write_lammps_forcefield_str(ff, frame)` raises for a force field with one `pair thole` type used by the frame;
      P3: `molrs.ff.lammps_coeff_params("bond", "harmonic", ["268", "1.529"]) == {"k": 536.0, "r0": 1.529}`;
          the charmm call gives phase `math.radians(-180)` and w 0.0; the `pair lj/cut` call gives epsilon 0.066 and sigma 3.5;
          the `improper cvff` and `bond morse` calls raise `ValueError`.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-003
    summary: Q0 and SITE removed from molpy.core.fields
    type: code
    pass_when: |
      src/molpy/core/fields.py assigns neither `Q0` nor `SITE`, and its `__all__` lists neither.
      `python -c "import molpy.core.fields as f; assert not hasattr(f,'SITE') and not hasattr(f,'Q0')"` exits 0.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-004
    summary: parameter-formatter registry and CL&Pol formatters deleted
    type: code
    pass_when: |
      `rg -n "ForceFieldFormatter|_param_formatters|register_param_formatter|format_params|_format_pair_" src/molpy` returns no hits.
      tests/test_io/test_forcefield/test_clpol_formatters.py does not exist.
      `molpy.core.fields.FieldFormatter is molrs.fields.FieldFormatter`.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-005
    summary: dead RB→OPLS duplicate deleted, not moved
    type: code
    pass_when: |
      src/molpy/io/forcefield/_rb_opls.py does not exist.
      `git ls-files tests/test_potential` prints nothing, and the directory tests/test_potential/ does not exist on disk.
      `rg -n "_rb_opls|rb_to_opls|format_lammps_dihedral_coeff" src tests` returns no hits.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-006
    summary: core.forcefield is a pure identity re-export incl. PotentialCompiler; alias has no user
    type: code
    pass_when: |
      src/molpy/core/forcefield.py defines no class, no function and no alias assignment.
      `mp.PotentialCompiler is molrs.ff.PotentialCompiler`, `mp.ForceField is molrs.ff.ForceField` and `mp.Potentials is molrs.ff.Potentials`.
      Neither `mp` nor `mp.core` has any of the 23 deleted `*Style` names or `AtomisticForcefield`.
      `rg -n "AtomisticForcefield" src` returns no hits; under tests/ it hits only files owned by deferred links (amended 2026-09-27: tests/test_typifier/*, tests/test_core/test_ops/test_scale_lj.py, tests/test_builder/test_finalize.py:52,64), which link 03's deferral entry records.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-007
    summary: molpy.potential package deleted as a directory; docstrings no longer cite to_potentials
    type: code
    pass_when: |
      `git ls-files src/molpy/potential` prints nothing, and the directory src/molpy/potential/ does not exist on disk.
      `uv run --extra dev python -c "import molpy.potential"` fails with `ModuleNotFoundError`.
      `rg -n "molpy\.potential|to_potentials" src/molpy` returns no hits.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-008
    summary: test_fields.py and test_openmm.py pass; tests of deleted or re-exported code are gone
    type: runtime
    pass_when: |
      `uv run --extra dev python -m pytest tests/test_core/test_fields.py tests/test_engine/test_openmm.py` passes.
      test_fields.py contains no `TestForceFieldFormatter`, `test_site_is_a_molpy_owned_column` or `test_every_molrs_key_is_a_named_string_constant`.
      tests/test_core/test_forcefield.py, tests/test_core/test_forcefield_editing.py and tests/test_md/test_driver.py do not exist.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "python -m pytest test_fields.py test_openmm.py: 16 passed"
  - id: ac-009
    summary: molrs pin unchanged (covered by the R1 § VII exception, P0)
    type: code
    pass_when: |
      pyproject.toml `dependencies` contains exactly "molcrafts-molrs>=0.15.0,<0.16", identical to 60a3455.
      The law-16 exception for it is P0 (ac-002), and the deferred close link ac-004 re-checks it at chain end.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-010
    summary: suite and lint clean for files touched by this link
    type: runtime
    note: chain-end gate
    pass_when: |
      Discharged by the deferred close link ac-008 (full suite) and ac-009 (tox -e lint):
      no failure or lint finding in any file listed under this link's "Files to create or modify".
    status: pending
    note: chain-end gate
---

# Acceptance criteria

ac-002 and ac-006 are one-off evaluator checks. They run once, outside the suite, so they do not violate law 17 (no native-parity check in the suite). ac-006 checks identity (`is`), not equality.

The directory checks (ac-005, ac-007) use `git ls-files` plus absence on disk. After a `git rm`, ignored `__pycache__/` folders would otherwise stay behind, and a leftover `src/molpy/potential/` would still import as an empty namespace package.

P4 (the inverse coefficient conversion) is not checked here. molrs-015-align-03-moltemplate checks it in its first task.
