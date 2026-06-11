---
title: CL&P fixed-charge force field for ionic liquids (inherit from OPLS typifier)
status: code-complete
created: 2026-06-10
---

# CL&P fixed-charge force field for ionic liquids (inherit from OPLS typifier)

> Depends on `clp-typifier-01-consolidation` having landed (base class is `OplsTypifier`).

## Summary
This spec adds the CL&P (Canongia Lopes & Pádua) all-atom fixed-charge force field for ionic liquids to molpy, implemented strictly by inheritance from the OPLS-AA typifier rather than by merging CL&P data into `oplsaa.xml`. A user can construct a `ClpTypifier`, load the new built-in `clp.xml` force-field data, and typify imidazolium-based ionic-liquid structures so that each atom receives its CL&P atom type, partial charge, and LJ parameters, with bonded terms assigned through the existing OPLS machinery. First-version coverage is bounded to the 1,3-dialkylimidazolium cation family and four common anions ([BF4]-, [PF6]-, [NTf2]-, [dca]-); the OPLS-AA force field and GAFF behavior are left byte-for-byte and logic unchanged.

## Domain basis
CL&P is a systematic all-atom force field for ionic liquids whose functional form is fully OPLS-AA compatible:

- Primary reference: Canongia Lopes, Deschamps, Pádua, "Modeling Ionic Liquids Using a Systematic All-Atom Force Field", *J. Phys. Chem. B* **2004**, 108, 2038–2047. DOI 10.1021/jp0362133. Plus the JPCB extension series (2004 anion set, later cation extensions).
- Functional form (identical to OPLS-AA):
  - Harmonic bonds: `E = K (r − r0)^2`
  - Harmonic angles: `E = K (θ − θ0)^2`
  - Dihedrals: OPLS cosine (Fourier) series.
  - Nonbonded: LJ 12-6 with **geometric** combining rules.
  - 1-4 scaling: **0.5** electrostatic / **0.5** LJ.
- Units (same as `oplsaa.xml`, so OPLS reader conversions apply unchanged): energies in kJ/mol, lengths in nm; epsilon kJ/mol→kcal/mol (÷4.184), sigma nm→Å (×10).
- CL&P-specific content: ion partial charges from ab initio (CHelpG/MP2), ion-specific torsions, and new atom types for ionic ring systems. Most LJ, bond, and angle parameters are transferred unchanged from OPLS-AA.
- Integer-charge invariant: the summed partial charge of each ion is exactly ±1.
- Atom-typing note: the official CL&P distribution (`il.ff`, used with fftool) identifies atoms by name + connectivity, not SMARTS. Each CL&P atom type is therefore authored here as a SMARTS `def_` pattern. Same-element ring atoms must be discriminated by ring position — e.g. imidazolium ring carbon CR between the two ring nitrogens vs CW on the back of the ring, and the ring nitrogen NA — using the same `overrides`/priority mechanism `oplsaa.xml` uses for benzene ring carbons.

## Design
- **Engine layer** — `ClpTypifier(OplsTypifier)` in `src/molpy/typifier/clp.py`. `OplsTypifier` already accepts a `ForceField` as a constructor argument and shares the entire SMARTS/overrides/priority + `LayeredTypingEngine` matching machinery; CL&P reuses all of it. The subclass is near-trivial: a convenience constructor that, when no force field is passed, loads the built-in `clp.xml` via `OPLSAAForceFieldReader` (CL&P units == OPLS units) and delegates the rest to `OplsTypifier`. No override of the atom/pair/bond/angle/dihedral sub-typifiers is required because the matching engine is parameter-driven by the loaded `ForceField`. Export `ClpTypifier` from `typifier/__init__.py`.
- **Data layer** — NEW file `src/molpy/data/forcefield/clp.xml`, authored from scratch in the same schema as `oplsaa.xml` (`AtomTypes` with `name`/`class`/`element`/`mass`/`def`[SMARTS]/`overrides`/`desc`/`doi`; `NonbondedForce` `Atom` charge/sigma/epsilon; `HarmonicBondForce`; `HarmonicAngleForce`; `Proper`/`RBTorsion`). `oplsaa.xml` is **not** edited. `get_forcefield_path("clp.xml")` resolves automatically because `data/forcefield` discovers any file in the directory; no registry edit needed beyond placing the file.
- **Reader decision** — CL&P uses the same units, geometric combining rule, and 0.5/0.5 1-4 scaling as OPLS-AA, so **no new reader class is added**. `clp.xml` is read through the existing `OPLSAAForceFieldReader` (which performs the kJ→kcal and nm→Å conversions). The only wiring needed is routing the `clp.xml` filename to the OPLS reader path: extend the filename dispatch in `read_xml_forcefield` (currently special-cases `oplsaa.xml`) so `clp.xml` also dispatches to `read_oplsaa_forcefield`. A dedicated `ClpForceFieldReader` subclass is explicitly rejected as unnecessary given identical unit handling.
- **Lifecycle / ownership** — `ClpTypifier` owns no new data; it borrows `clp.xml` (shipped in package data) and the shared matching engine. New public symbol: `ClpTypifier`. New data artifact: `clp.xml`.

## Files to create or modify
- `src/molpy/typifier/clp.py` (new) — `ClpTypifier(OplsTypifier)` + convenience loader.
- `src/molpy/typifier/__init__.py` — export `ClpTypifier`, add to `__all__`.
- `src/molpy/data/forcefield/clp.xml` (new) — CL&P atom types (imidazolium ring + alkyl), four anions, nonbonded/bonded params, SMARTS `def_`/`overrides`.
- `src/molpy/io/forcefield/xml.py` — route `clp.xml` filename through `read_oplsaa_forcefield` in the `read_xml_forcefield` dispatch.
- `tests/test_typifier/test_clp.py` (new) — typifier import/inheritance, ring-atom discrimination, anion typing, scientific regression vs il.ff fixtures.
- `tests/test_typifier/fixtures/clp_ilff_reference.json` (new) — il.ff reference atom types / charges / LJ params for [C4C1im][NTf2] and [C4C1im][DCA].

## Tasks
- [x] Write failing tests for ClpTypifier import and inheritance (tests/test_typifier/test_clp.py)
- [x] Author clp.xml data file with imidazolium ring + alkyl atom types, four anions, nonbonded/bonded params and SMARTS def/overrides (src/molpy/data/forcefield/clp.xml)
- [x] Implement ClpTypifier(OplsTypifier) with built-in clp.xml loader (src/molpy/typifier/clp.py)
- [x] Export ClpTypifier from typifier package (src/molpy/typifier/__init__.py)
- [x] Route clp.xml filename through read_oplsaa_forcefield dispatch (src/molpy/io/forcefield/xml.py)
- [x] Source il.ff reference values into test fixtures (tests/test_typifier/fixtures/clp_ilff_reference.json)
- [x] Write failing tests for imidazolium ring-atom discrimination CR vs CW vs NA on [C4C1im]+ (tests/test_typifier/test_clp.py)
- [x] Write failing tests for the four anions typifying without error (tests/test_typifier/test_clp.py)
- [x] Write failing scientific regression + integer-charge + combining/scaling assertions against il.ff fixtures (tests/test_typifier/test_clp.py)
- [x] Run full check + test suite

## Testing strategy
- **Happy path**: `ClpTypifier` is importable from `molpy.typifier` and is a subclass of `OplsTypifier`; `get_forcefield_path("clp.xml")` resolves to an existing file; building [C4C1im][NTf2] and [C4C1im][DCA], then typifying, completes without raising.
- **Edge cases**: imidazolium ring atoms are discriminated by ring position — CR (carbon between the two ring N), CW (back-ring carbon), and NA (ring nitrogen) on [C4C1im]+ each receive their distinct CL&P type via the overrides/priority mechanism; each of the four named anions ([BF4]-, [PF6]-, [NTf2]-, [dca]-) typifies without error and yields the expected per-atom types.
- **Domain validation** (`science.required`): assigned atom types, partial charges, and LJ sigma/epsilon for [C4C1im][NTf2] and [C4C1im][DCA] match the il.ff reference fixtures within tolerance; each ion's summed partial charge equals ±1 exactly (integer-ion invariant); the produced ForceField records geometric combining rule and 0.5 electrostatic / 0.5 LJ 1-4 scaling.
- **Regression guard**: `oplsaa.xml` is byte-identical before and after this change (no merge); OPLS/GAFF typifier behavior unchanged.
- **Fixture sourcing**: il.ff reference atom types/charges/LJ values must be transcribed from the official CL&P/il.ff distribution into `clp_ilff_reference.json` as a committed fixture; without it the scientific criteria cannot be evaluated.

## Out of scope
- Cations other than 1,3-dialkylimidazolium: pyridinium, pyrrolidinium, ammonium, phosphonium, cholinium — deferred to future specs.
- Anions other than the four named: triflate, alkylsulfate, halides — deferred to future specs.
- CL&Pol (polarizable CL&P): requires Drude infrastructure not present in molpy — separate project, explicitly excluded.
- Any modification to `oplsaa.xml` or to OPLS/GAFF typifier logic (user requirement: 继承不合并).
- A dedicated `ClpForceFieldReader` class (rejected: CL&P unit/charge handling is identical to OPLS-AA; `OPLSAAForceFieldReader` is reused).
