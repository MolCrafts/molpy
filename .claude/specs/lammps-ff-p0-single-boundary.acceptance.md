---
spec: lammps-ff-p0-single-boundary
created: 2026-08-06
revised: 2026-08-06
criteria:
  - id: ac-001
    summary: "molrs write_data_coeffs emits *Coeffs sections only"
    type: runtime
    pass_when: "Public API returns Pair/Bond Coeffs headings and type_id lines; no pair_style/bond_style/units keywords"
    status: verified
    last_checked: 2026-08-06
  - id: ac-002
    summary: "data-coeffs and *.ff writers share form map and units"
    type: runtime
    pass_when: "Same ForceField + units=real|metal → same bond K and pair eps numbers in both layouts (same precision)"
    status: verified
    last_checked: 2026-08-06
  - id: ac-003
    summary: "molpy has zero form/unit arithmetic for FF coeffs"
    type: code
    pass_when: "src/molpy/io/data/lammps.py does not use k/2, /2.0, math.degrees for building *Coeffs; calls molrs"
    status: verified
    last_checked: 2026-08-06
  - id: ac-004
    summary: "peptide structure+coeffs round-trip via composition API"
    type: runtime
    pass_when: "write_lammps_data then write_lammps_data_coeffs; re-read pair/bond coeffs match; no LammpsDataWriter(forcefield=)"
    status: verified
    last_checked: 2026-08-06
  - id: ac-005
    summary: "Frame type columns own ids; pure int skips Type Labels"
    type: runtime
    pass_when: "Numeric type names write without Atom Type Labels; missing both type and type_id raises; test_sorted_type_names still green if kept"
    status: verified
    last_checked: 2026-08-06
  - id: ac-006
    summary: "ForceField.map_type stamps usable type info on Frame"
    type: runtime
    pass_when: "map_type on frame with atom type names yields type_id (or preserves consistent type); missing type and type_id raises"
    status: verified
    last_checked: 2026-08-06
  - id: ac-007
    summary: "LammpsDataWriter has no forcefield parameter"
    type: code
    pass_when: "LammpsDataWriter.__init__ and write_lammps_data do not accept forcefield= as owned writer state"
    status: verified
    last_checked: 2026-08-06
  - id: ac-008
    summary: "data_coeffs write resolves ids from Frame not user inventory"
    type: runtime
    pass_when: "write_lammps_data_coeffs(path, frame, ff) aligns coeff type ids with frame type/type_id; unknown FF type name raises"
    status: verified
    last_checked: 2026-08-06
out_of_scope:
  - "New pair/bond styles beyond tier A"
  - "ForceField.store_units field"
  - "One-pass structure+FF read in molrs"
  - "si/cgs"
  - "Thole/coul.tt in molrs writer"
  - "SMARTS typifier inside map_type"
---

# Acceptance — lammps-ff-p0-single-boundary

**Done** means: (1) one molrs arithmetic boundary for data coeffs and `*.ff`; (2) Frame owns type/`type_id`; (3) `ff.map_type(frame)` is the mapping primitive; (4) Writer does not own forcefield; composition API for coeffs insert.

## AC-001 … AC-003

Unchanged intent from prior audit: molrs emitter, parity, molpy grep gate.

## AC-004

Tests must use two-step write, not `LammpsDataWriter(..., forcefield=ff)`.

## AC-005

Frame completeness; numeric types identity.

## AC-006

`map_type` minimal contract (not full typifier).

## AC-007

API shape lock from grill.

## AC-008

Ids from Frame when emitting *Coeffs.
