---
slug: clp-typifier-02-forcefield
criteria:
  - id: ac-001
    summary: ClpTypifier importable from molpy.typifier and subclass of OplsTypifier
    type: code
    pass_when: |
      `from molpy.typifier import ClpTypifier` succeeds and
      `issubclass(ClpTypifier, OplsTypifier)` is True.
    status: verified
    last_checked: 2026-06-10
  - id: ac-002
    summary: clp.xml resolves via get_forcefield_path
    type: code
    pass_when: |
      get_forcefield_path("clp.xml") returns a path to an existing file
      under src/molpy/data/forcefield/, and "clp.xml" appears in
      list_forcefields().
    status: verified
    last_checked: 2026-06-10
  - id: ac-003
    summary: oplsaa.xml left byte-unchanged
    type: code
    pass_when: |
      git diff against the base shows zero changes to
      src/molpy/data/forcefield/oplsaa.xml (byte-identical).
    status: verified
    last_checked: 2026-06-10
  - id: ac-004
    summary: clp.xml read through OPLSAAForceFieldReader (no new reader class)
    type: code
    evaluator_hint: "marker: clp"
    pass_when: |
      read_xml_forcefield(get_forcefield_path("clp.xml")) returns a populated
      ForceField, and no new ClpForceFieldReader class exists in
      io/forcefield/xml.py (clp.xml dispatches to read_oplsaa_forcefield).
    status: verified
    last_checked: 2026-06-10
  - id: ac-005
    summary: imidazolium ring atoms CR vs CW vs NA discriminated on [C4C1im]+
    type: code
    evaluator_hint: "marker: clp"
    pass_when: |
      Typifying [C4C1im]+ with ClpTypifier assigns distinct CL&P types to the
      ring carbon between the two N (CR), the back-ring carbons (CW), and the
      ring nitrogen (NA), matching expected types in the test.
    status: verified
    last_checked: 2026-06-10
  - id: ac-006
    summary: four named anions typify without error
    type: code
    evaluator_hint: "marker: clp"
    pass_when: |
      ClpTypifier typifies [BF4]-, [PF6]-, [NTf2]-, and [dca]- without raising,
      and every atom in each anion receives a non-None CL&P type.
    status: verified
    last_checked: 2026-06-10
  - id: ac-007
    summary: assigned types/charges/LJ match il.ff reference within tolerance
    type: scientific
    evaluator_hint: "marker: clp; fixture: tests/test_typifier/fixtures/clp_ilff_reference.json"
    pass_when: |
      For [C4C1im][NTf2] and [C4C1im][DCA], per-atom CL&P type, partial charge,
      and LJ sigma/epsilon match the il.ff reference fixture (charge atol 1e-4,
      sigma/epsilon rtol 1e-4).
    status: pending
  - id: ac-008
    summary: each ion summed partial charge equals integer ±1
    type: scientific
    pass_when: |
      Sum of partial charges over each typified ion equals +1 (cation) or
      -1 (anion) within atol 1e-6.
    status: pending
  - id: ac-009
    summary: produced ForceField records geometric combining + 0.5/0.5 1-4 scaling
    type: scientific
    pass_when: |
      The ForceField from clp.xml reports geometric combining rule and
      1-4 scaling factors of 0.5 (electrostatic) and 0.5 (LJ).
    status: pending
  - id: ac-010
    summary: lint and type check clean
    type: code
    pass_when: |
      `ruff check src tests` and `ty check src/molpy/` both exit 0 on the
      changed files.
    status: verified
    last_checked: 2026-06-10
---

# Acceptance criteria

- **ac-001 / ac-002 / ac-004**: wiring sanity — the new typifier and data file are reachable through public APIs, and the reader reuse decision is enforced (no `ClpForceFieldReader`).
- **ac-003**: enforces the user's 继承不合并 requirement — `oplsaa.xml` must not be touched.
- **ac-005 / ac-006**: typing-engine correctness for the bounded first-version scope (imidazolium ring discrimination + four anions).
- **ac-007 / ac-008 / ac-009**: scientific validation against il.ff. ac-007 and ac-008 require the il.ff reference values to be transcribed into `tests/test_typifier/fixtures/clp_ilff_reference.json`; sourcing these reference values is a prerequisite for these criteria to be evaluable.
- **ac-010**: repo quality gate.
