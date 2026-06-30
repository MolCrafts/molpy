---
slug: mmff-native-02-typifier-ff
criteria:
  - id: ac-001
    summary: MMFFTypifier is a public class exported from molpy.typifier
    type: code
    pass_when: |
      `from molpy.typifier import MMFFTypifier` succeeds and MMFFTypifier
      is a class (not a factory function); it is listed in
      src/molpy/typifier/__init__.py __all__ alongside OplsTypifier/GaffTypifier.
    status: pending
  - id: ac-002
    summary: No factory function is added for the MMFF path
    type: code
    pass_when: |
      grep across src/molpy finds no new module-level def named
      optimize_mmff/build_mmff/build_* for MMFF; the only public surface is
      the MMFFTypifier class and molpy.ForceField.
    status: pending
  - id: ac-003
    summary: molrs imported at module top (mandatory backend, not optional)
    type: code
    pass_when: |
      src/molpy/typifier/mmff.py contains a top-level `import molrs`
      and no try/except ImportError guard around it.
    status: pending
  - id: ac-004
    summary: typify returns a to_potentials-ready molrs.Frame
    type: runtime
    pass_when: |
      For a small typeable molecule, ff = MMFFTypifier().forcefield() and
      frame = MMFFTypifier().typify(mol); ff.to_potentials(frame) returns
      without raising the 'mmff_stbn: missing "r0_ij" column' ValueError.
    status: pending
  - id: ac-005
    summary: forcefield() returns molpy public ForceField
    type: runtime
    pass_when: |
      MMFFTypifier().forcefield() returns an instance of molpy.ForceField
      (the public re-exported class from molpy.core.forcefield).
    status: pending
  - id: ac-006
    summary: unknown variant and untypeable molecule fail fast with ValueError
    type: runtime
    pass_when: |
      MMFFTypifier(variant="BOGUS") raises ValueError; calling typify on an
      MMFF-untypeable molecule raises ValueError carrying a parameter-availability
      message.
    status: pending
  - id: ac-007
    summary: MMFF94s variant typifies successfully
    type: runtime
    pass_when: |
      MMFFTypifier(variant="MMFF94s").typify(mol) returns a frame on which
      forcefield().to_potentials(frame) does not raise, for a small molecule.
    status: pending
  - id: ac-008
    summary: energy is non-increasing along the Frame-based LBFGS trajectory
    type: scientific
    pass_when: |
      Running LBFGS(ForceFieldPotential(MMFFTypifier().forcefield())).run(frame)
      on a distorted small molecule yields per-step energies that are
      monotonically non-increasing within 1e-6 kcal/mol slack.
    status: pending
  - id: ac-009
    summary: native molrs-MMFF matches RDKit MMFF94 energy + geometry within tolerance
    type: scientific
    evaluator_hint: pytest.mark.external (requires RDKit)
    pass_when: |
      For each molecule in the small test set, native-optimized final energy
      matches OptimizeGeometry(forcefield="MMFF94") within the documented energy
      tolerance and optimized geometry within the documented RMSD tolerance (angstrom).
    status: pending
  - id: ac-010
    summary: MMFFTypifier and methods carry Google-style docstrings with units + DOIs
    type: docs
    pass_when: |
      MMFFTypifier and its typify/forcefield methods have Google-style docstrings
      stating kcal/mol + angstrom units and citing Halgren 1996 (MMFF94) and
      Halgren 1999 (MMFF94s) DOIs.
    status: pending
  - id: ac-011
    summary: full check + test suite passes
    type: runtime
    pass_when: |
      `ruff check src tests && ty check src/molpy/` passes and
      `pytest tests/ -m "not external" -v` passes.
    status: pending
---

# Acceptance criteria

- ac-001 / ac-002 / ac-003 — binding architecture rules: public `MMFFTypifier` class,
  no factory functions, molrs as a mandatory top-level import.
- ac-004 — the critical blocker: the typified frame must be assembly-complete
  (`merge_stbn_r0` run) so spec 01's `ForceFieldPotential -> to_potentials(frame)` works.
- ac-005 — MMFF force field surfaced through molpy's public `ForceField`.
- ac-006 / ac-007 — fail-fast error handling and variant coverage.
- ac-008 — minimizer invariant (no new physics; checks the wired optimizer).
- ac-009 — external parity against the untouched RDKit `OptimizeGeometry` peer.
- ac-010 / ac-011 — docs and the build/test gate.
