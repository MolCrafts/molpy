# Notes

Evolving architectural decisions and project-level rules. Populated by `/mol:note`.

## release-molrs-first

**Promoted** → `.claude/notes/release.md` + CLAUDE.md § Release with molrs.
Monorepo merge under molpy: **retracted**. Pin-parity scripts: **forbidden**.

## pathlike-boundary-path-internal

**Public / user-facing** path parameters accept `str | Path` (path-like).
**At the boundary**, convert with `Path(...)` / `Path(...).expanduser()`.
**Internal fields and logic use only `pathlib.Path`** — never keep bare `str`
paths for filesystem locations after construction. Subprocess argv / env dicts
are the only place to re-emit `str(path)`.

Exceptions: symbolic names that are *not* paths (e.g. conda env name
`"AmberTools25"`) stay `str`. Path-like conda prefixes are stored as `Path`.

## io-bond-react-emit-stay-molpy

**bond_react** (`io/data/lammps_bond_react.py`) and **emit** (`io/emit/`) stay
in **molpy** until their product design is fixed. Do **not** sink map
serialization, REACTER packaging, or engine-input emission to molrs yet —
format boundaries that *are* settled (data/FF/trajectory) continue molrs-first.

## 2026-09-27 — molrs-015-align narrowed to 01–03; deferred alignment (routed `/mol:spec`)

The operator narrowed the molrs-015-align chain to links 01–03 (2026-09-27:
"按收窄方案走，先把脚本跑通，然后再去做其他部分收尾") so the CG→all-atom
backmap script can run against molpy first. What links 01–03 leave undone is
recorded here, one item per debt, until the deferred links land. Every item is
owned by the operator and routed through `/mol:spec`. The deferred links are
the staged `molrs-015-align` links renumbered from 04.

### `molpy.builder` and `molpy.typifier` do not import

`molpy.typifier` imports again since trace-assembly-08 (its package
`__init__` is narrowed to native re-exports; see the trace-assembly-08
entry). `molpy.builder` still does not import.

    Law:               13 (native facade) — typifier subclassing and assembly
                       still target retired molrs APIs.
    Evidence:          `python -c "import molpy.builder"` and
                       `python -c "import molpy.typifier"` raise `TypeError:
                       Typifier defines typify; a Typifier subclass implements
                       match (and optionally library) only`.
    Scope:             src/molpy/builder/, src/molpy/typifier/.
    Removal condition: the deferred builder and typifier links land.
    Owner:             operator.
    Route:             `/mol:spec` (deferred builder and typifier links).

### moltemplate and the LAMMPS force-field writers/emitter/engine break at runtime

    Law:               13 (native facade), 15 (I/O and force-field file
                       contracts).
    Evidence:          src/molpy/parser/moltemplate/* and the LAMMPS
                       force-field writer, emitter and engine paths import but
                       call force-field style names and graph members removed
                       by links 01–02; the design round-3 findings stay with
                       the deferred links.
    Scope:             src/molpy/parser/moltemplate/, src/molpy/io/forcefield/
                       lammps.py, src/molpy/io/emit/, src/molpy/engine/.
    Removal condition: the deferred moltemplate and LAMMPS links land.
    Owner:             operator.
    Route:             `/mol:spec` (deferred moltemplate and LAMMPS links).

### Tests still use removed API

    Law:               11 (tests verify owned behavior), 17 (unit-only test
                       gate).
    Evidence:          tests/test_typifier/*;
                       tests/test_core/test_ops/test_scale_lj.py
                       (`AtomisticForcefield`, `def_atomstyle`/`def_pairstyle`);
                       tests/test_builder/test_finalize.py:52,64
                       (`AtomisticForcefield` for bonded typing);
                       tests/test_io/test_forcefield/test_lammps.py (deleted
                       style names); callers of graph members removed by link
                       02 in tests that import builder/typifier:
                       tests/test_io/test_data/test_lammps_bond_react.py:125,160
                       (`get_neighbors`, `del_bond`),
                       tests/test_io/test_data/test_lammps_drude_flags.py:42,74
                       (`get_topo`, `def_atoms`),
                       tests/test_io/test_forcefield/test_xml.py:254
                       (`get_topo`),
                       tests/test_builder/test_polymer/test_ambertools/
                       test_amber_builder.py:24 (`def_bonds`).
    Scope:             the test files listed above.
    Removal condition: their owning deferred links land.
    Owner:             operator.
    Route:             `/mol:spec` (the owning deferred links).

### rdkit `_positions` diverges from link 02's spec table

    Law:               14 (never fall back silently — raise).
    Evidence:          link 02's spec table names `atoms["x","y","z"]` for
                       src/molpy/adapter/rdkit.py `_positions`, but that view
                       returns `None` on a coordinate hole; the port uses
                       `np.stack` of native `column()`, which raises `KeyError`.
    Scope:             src/molpy/adapter/rdkit.py `_positions`; link 02 spec
                       table.
    Removal condition: the deferred close link amends the table.
    Owner:             operator.
    Route:             `/mol:spec` (deferred close link).

### Stale developer docs and CLAUDE.md sections

    Law:               9 (one home per fact), 10 (no silent debt).
    Evidence:          CLAUDE.md, docs/developer/*, docs/api/potential.md and
                       user-guide pages still name `LammpsForceFieldFormatter`,
                       `BondMorseStyle`, `ForceFieldFormatter`, `SITE`/`Q0`
                       and the `potential` package.
    Scope:             CLAUDE.md, docs/.
    Removal condition: the deferred docs and close links land.
    Owner:             operator.
    Route:             `/mol:spec` (deferred docs and close links).

### Full suite and lint gate not run for this chain

    Law:               17 (unit-only test gate).
    Evidence:          links 01–03 ran only their targeted tests; the full
                       suite and `tox -e lint` have not run, and the
                       builder/typifier/assembly tests fail at collection.
    Scope:             the whole molrs-015-align chain.
    Removal condition: the deferred close link runs the gate green.
    Owner:             operator.
    Route:             `/mol:spec` (deferred close link).

### Lint gate red on `molpy.builder.assembly`; molpy commits wait

    Law:               10 (no silent debt), 17 (unit-only test gate).
    Evidence:          `ty` reports 3 warnings on `fields.SITE` in
                       src/molpy/builder/assembly/ (a field removed by link
                       02), so the pre-commit lint gate is red; molpy commits
                       for this chain and for trace-assembly-08 are held.
    Scope:             src/molpy/builder/assembly/; every pending molpy commit
                       on branch `ci/narrow-gitignore`.
    Removal condition: the deferred builder link lands and the lint gate
                       runs green; the held commits are then made.
    Owner:             operator.
    Route:             `/mol:spec` (deferred builder link).

### `mp.Region` is molpy's predicate model beside the native shapes

    Law:               1 (conceptual integrity — parallel model).
    Evidence:          `mp.Region` is molpy's `MaskPredicate` model in
                       src/molpy/core/region.py; the concrete regions
                       (`BoxRegion`, `SphereRegion`, the combinators) subclass
                       the native `Cuboid`/`Sphere` shapes. The native region
                       family needs `molpy.builder` (`Lattice.build`), which
                       does not import yet.
    Scope:             src/molpy/core/region.py, `Lattice.build`.
    Removal condition: the deferred region migration lands.
    Owner:             operator.
    Route:             `/mol:spec` (deferred core re-export link).

### `molpy.op` sits beside `molpy.core.ops`

    Law:               1 (conceptual integrity), 9 (one home per fact) — a
                       near-homonym.
    Evidence:          `molpy.op` is the verbatim native numeric base
                       (`superpose`, `centroid`); `molpy.core.ops` holds the
                       LJ scaling helpers (`FragmentScaling`).
    Scope:             src/molpy/op/, src/molpy/core/ops/.
    Removal condition: the deferred deletion of `core/ops` into `mp.ff` lands.
    Owner:             operator.
    Route:             `/mol:spec` (deferred core re-export link).

### Deferred links renumbered from 04

    Law:               10 (no silent debt).
    Evidence:          the staged molrs-015-align links after 03 were not
                       written; link 03 took over `Fragment`, `Port`,
                       `FrameMeta`, `MetaDocument`, `SubgraphMatcher`,
                       `mp.op`, the CGSmilesIR family and the SmilesReader
                       brace-refusal message.
    Scope:             the deferred molrs-015-align links (04+).
    Removal condition: the deferred links are written from 04 and drop what
                       link 03 owns.
    Owner:             operator.
    Route:             `/mol:spec` (deferred molrs-015-align links 04+).

## 2026-09-27 — trace-assembly-08: molpy re-exports for the backmap script

Root identity re-exports added: `mp.Trace`, `mp.Coarsener`, `mp.Assembler`,
`mp.TracePlacer` (`molrs.Trace`, `molrs.perceive.Coarsener`,
`molrs.builder.Assembler`, `molrs.builder.TracePlacer`). Two Law-10 items:

### `molpy.typifier` narrowed to native re-exports

    Law:               10 (no silent debt), 13 (native facade).
    Evidence:          src/molpy/typifier/__init__.py re-exports only
                       `ElementTypifier`, `OPLSAATypifier` and `MMFFTypifier`
                       (= `molrs.ff.typifier.MMFF94Typifier`) by identity.
                       The molpy modules `base`, `clp`, `smarts`,
                       `ambertools`, `scope`, `forcefield`, `region`,
                       `affected_region`, `cache` and `_matching` are no
                       longer imported by the package; importing one directly
                       still fails. The names dropped from `__all__`
                       (`Typifier`, `Match`, `TypeScope`,
                       `UnboundedPatternSet`, `LocalTypifier`,
                       `SmartsTypifier`, `ClpTypifier`, `AmberToolsTypifier`,
                       `ForceFieldParams`) were unreachable, since the package
                       failed at import.
    Scope:             src/molpy/typifier/.
    Removal condition: the deferred typifier link deletes or ports those
                       modules, per the operator ruling "molpy deletes its
                       typifiers, re-exports molrs".
    Owner:             operator.
    Route:             `/mol:spec` (deferred typifier link).

### `mp.io.write_lammps_data` is a free function over a façade

    Law:               10 (no silent debt), 13 (native facade), 14 (never
                       fall back silently); conflicts with ruling (a), OOP
                       surface over free functions.
    Evidence:          src/molpy/io/writers.py:23-48 defines
                       `write_lammps_data` as a module-level function under
                       `mp.io`; it forwards to the "thin native façade"
                       `LammpsDataWriter` and silently discards its
                       `atom_style` argument (`del atom_style`). The backmap
                       script (/home/jicli594/work/backmap_pe_pma/backmap.py)
                       depends on it, so it stays for trace-assembly-08.
    Scope:             src/molpy/io/writers.py `write_lammps_data`.
    Removal condition: the refactor lands the OOP writer surface.
    Owner:             operator.
    Route:             `/mol:refactor`.

## 2026-09-28 — backmap script run on the real data (trace-assembly-08 acceptance)

- Script: `/home/jicli594/work/backmap_pe_pma/backmap.py` (imports molpy only), input `PE_100_chains_length_100_solvPerc_0.9_equil_50M.data`, sigma = 4.2 A.
- Host: naiss login node; molrs feat/cgsmiles (trace-assembly 01–05, 07 uncommitted at run time), release `maturin develop --release`.
- Result: exit 0; `wrote .../pe_pma_aa.data: 6170200 atoms`; wall 37.9 s; max RSS 11.1 GB (bars: < 15 min, < 64 GB).
- Written file: `Atoms # molecular`, 6,170,200 atoms, 6,160,100 bonds, 460,100 distinct molecule ids, 7 atom type labels (C F H Li N O S), 7 bond type labels (C-C C-F C-H C-O C-S N-S O-S); box origin 0 and lengths 352.3400856 A = CG box x 4.2; every atom inside the box after wrap.
- Read-back: `mp.io.read_lammps_data(OUT, atom_style="molecular")` gives the same counts.
- LAMMPS: `lmp` (`units real`, `atom_style molecular`, `read_data`) reads the file with no error (read_data 16.5 s).
- Conformation only: no force field; relax before production.
