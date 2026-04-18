# Moltemplate CLI

MolPy ships a native moltemplate engine. The `molpy moltemplate` subcommand parses `.lt` scripts and emits ready-to-run inputs for **LAMMPS**, **OpenMM**, **GROMACS** — or MolPy's canonical XML force-field format, or a self-contained **MolPy Python script**.

## Quick start

```bash
# Summarise a script (atom types, molecule count, styles)
molpy moltemplate info water.lt

# Generate LAMMPS inputs (data + in.settings + in.init + starter in)
molpy moltemplate run water.lt --emit lammps --out-dir out/

# Generate every engine at once
molpy moltemplate run water.lt --emit all --out-dir out/

# FF-only: convert .lt force field to MolPy XML
molpy moltemplate convert gaff2.lt gaff2.xml

# .lt → MolPy Python script (inverse of moltemplate — hand-editable)
molpy moltemplate convert water.lt water.py

# MolPy system → .lt (ltemplify: bundle an Atomistic+ForceField back as .lt)
molpy moltemplate ltemplify water.lt water_regen.lt

# Dump parsed IR (debug)
molpy moltemplate parse water.lt --json ir.json
```

## Subcommands

### `run` — emit engine inputs

```
molpy moltemplate run SCRIPT [--emit ENGINE ...] [--out-dir DIR] [--prefix NAME]
```

| Engine    | Files produced (given `--prefix system`)                               |
|-----------|------------------------------------------------------------------------|
| `lammps`  | `system.data`, `system.in.settings`, `system.in.init`, `system.in`     |
| `openmm`  | `system.xml`, `system.pdb`, `system.py`                                |
| `gromacs` | `system.gro`, `system.top`, `em.mdp`, `nvt.mdp`                        |
| `xml`     | `system.xml`, `system.pdb` (MolPy canonical)                           |
| `all`     | every engine above                                                     |

`--emit` may be repeated: `--emit lammps --emit openmm`.

### `parse` — debug the IR

```
molpy moltemplate parse SCRIPT [--json OUT]
```

Without `--json`, prints a one-line-per-kind statement summary.

### `info` — one-liner summary

```
molpy moltemplate info SCRIPT
```

Prints atom-type/atom/bond/angle/dihedral counts after all `new` instances are expanded.

### `convert` — transform `.lt` into XML or Python

```
molpy moltemplate convert SRC.lt DST.{xml,py}
```

Output format is inferred from the destination extension:

| Extension | Output |
|-----------|--------|
| `.xml`    | MolPy canonical XML force field (FF-only). |
| `.py`     | Self-contained MolPy Python script with `build_forcefield()`, one `build_<ClassName>()` per moltemplate class, and a top-level `build_system()`. The emitted script has no runtime dependency on the original `.lt` file — users can edit freely. |

### `ltemplify` — `.lt` / `.data` → `.lt`

```
molpy moltemplate ltemplify SRC.lt DST.lt [--class-name NAME]
molpy moltemplate ltemplify SRC.data DST.lt --ff SRC.in.settings
```

Serialise an `Atomistic + ForceField` back into a moltemplate template. The first form re-emits a `.lt` after parsing; the second converts a LAMMPS data file plus a coefficients file.

Round-trip `read_moltemplate_system` → `ltemplify` is count-preserving on atoms / bonds / angles / dihedrals / impropers. It does **not** reconstruct the class hierarchy — everything is flattened into a single class.

## Supported moltemplate features

Tested against real examples in the
[`moltemplate/examples`](https://github.com/jewettaij/moltemplate/tree/master/examples)
directory. Coverage is intentionally incremental — the table below tracks
what has been validated on upstream fixtures versus what silently degrades.

| Feature                                              | Status |
|------------------------------------------------------|--------|
| `ClassName { ... }`, nested classes                  | ✔      |
| `inherits Parent1, Parent2`                          | ✔      |
| `import "file.lt"` (recursive)                       | ✔      |
| `write("...")`, `write_once("...")`                  | ✔      |
| `write('...')` single-quoted section names           | ✔      |
| Section names containing `(...)`                     | ✔      |
| `Data Masses`, `Data Charges`, `In Charges`          | ✔      |
| `In Settings` coeff lines (pair/bond/angle/…)        | ✔      |
| `Data Atoms`, `Data Bonds`, `Data Angles`            | ✔      |
| `Data Dihedrals`, `Data Impropers`                   | ✔      |
| `inst = new Cls`                                     | ✔      |
| `.move(x,y,z)`, `.rot(θ,ax,ay,az)`, `.scale(s)`      | ✔      |
| `new Cls [N].move(dx,dy,dz)` 1-D array               | ✔      |
| `new Cls [N].move(...) [M].move(...) [K].move(...)` (3-D array) | ✔      |
| `.rotvv(v1,v2)`                                      | partial (parsed but unused) |
| `new random([Cls1, Cls2], [w1, w2] [, seed])`        | ✔      |
| `$atom:submol/atom` scoped references                | ✔      |
| `Data Bond List` (no `@bond:T` column)               | ✔      |
| `Bonds/Angles/Dihedrals/Impropers By Type` wildcards | ✔ (rule matching applied after auto-topology) |
| `replace{ @atom:A @atom:B }`                         | ✔ (decoration applied during Data Atoms) |
| `create_var`, `delete_var`, `category`               | ✘ (silently ignored) |
| `Impropers` as first-class `Improper` link           | ✔      |
| oplsaa.lt (full ~10k line file)                      | parses; FF load OK; bond/angle/dihedral types resolved via By-Type wildcards |

### Honest caveat

`tip3p_2004_oplsaa.lt` + `oplsaa2024.lt` (the real moltemplate OPLS file,
~10k lines) parse end-to-end and produce a ForceField with thousands of
`AtomType` entries. `replace{ @atom:A @atom:B }` decoration is applied so
atom types carry their bond / angle / dihedral / improper partners, and
the `Bonds/Angles/Dihedrals/Impropers By Type` wildcard rules are
consulted *after* auto-topology to fill in the concrete type name on each
bond/angle/dihedral/improper. Edge cases where the wildcard does not
match still fall back to the synthetic placeholder name generated from
`In Settings` coeffs, so downstream emitters never crash.

### Verifying your own `.lt` file

```bash
molpy moltemplate parse my_system.lt         # IR summary
molpy moltemplate info my_system.lt          # atom/bond counts after expansion
molpy moltemplate run my_system.lt --emit lammps --out-dir out/
```

## Python API

Everything the CLI does is available programmatically.

```python
from molpy.io.forcefield.moltemplate import read_moltemplate_system
from molpy.io.emit import emit, emit_all
from molpy.parser.moltemplate import (
    emit_python,      # .lt → .py
    ltemplify,        # (atomistic, ff) → .lt string
    parse_file,       # .lt → IR Document
    write_moltemplate,  # (atomistic, ff) → .lt file
)

atomistic, ff = read_moltemplate_system("water.lt")

# Single engine
emit("lammps", atomistic, ff, "out/", prefix="w")

# All engines
emit_all(atomistic, ff, "out/", prefix="w")

# .lt → .py
emit_python(parse_file("water.lt"), "water.py")

# ltemplify: back to a .lt template
write_moltemplate(atomistic, ff, "water_regen.lt", class_name="Water")
```

**Python hooks**: moltemplate's own `include "foo.py"` mechanism is not
supported directly. Instead, use `molpy moltemplate convert foo.lt foo.py`
and edit the emitted script — the output is plain MolPy so every MolPy
Python API (builders, reacter, compute, wrapper) becomes available at the
same composition point you used to attach Python logic in moltemplate.

Core editing primitives on `Atomistic` / `CoarseGrain` / `ForceField` (see `core.atomistic`, `core.cg`, `core.forcefield`) include:

- `del_atom`, `del_bond`, `del_angle`, `del_dihedral`
- `rename_type(old, new, *, kind=Atom)`
- `set_property(selector, key, value, *, kind=Atom)`
- `select(predicate) -> Atomistic`
- `ForceField.rename_type / remove_type / remove_style`

These are kernel-level operations usable outside the moltemplate pipeline.
