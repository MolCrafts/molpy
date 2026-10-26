# Adding an I/O Format

`mp.io` has one door per format and direction, named after the format:
`read_<fmt>(source)` / `write_<fmt>(sink, frame)` for one frame (`read_pdb`,
`read_lammps_data`). A source is an `os.PathLike`, `bytes` or
`mp.io.Text(text)` — a bare `str` is refused, being ambiguous between a path
and text — and a sink an `os.PathLike` or a `bytearray`. Every frame of a
multi-frame file is read by the `<Fmt>Reader` class of the format's submodule
(`mp.io.pdb.PdbReader`), and a force-field format has a reader and a writer
class (`mp.io.gromacs.GromacsTopForcefieldReader`). No door picks a format for
the caller from a file extension. There are no reader or writer base classes
to subclass. Parsing and serialization belong in the Rust core:
every door on `mp.io` is native, force-field formats included
(`mp.ff.forcefield` is the `ForceField` data model only). A class that
belongs to one format lives in that format's submodule (`mp.io.smiles`,
`mp.io.lammps`, `mp.io.mrec`, …).

## A new format goes into the Rust core

Add the parser and writer to the Rust core (`crates/molrs/src/io/`) and bind them in the Python binding
(`molpy/src/io/`): a one-frame door at the top of `mp.io` (`read_<fmt>` /
`write_<fmt>`), and the format's classes — multi-frame reader and writer,
force-field reader and writer — in a `mp.io.<fmt>` submodule. A new function is listed in `molpy/python/molpy/io/__init__.py`'s import
from `molpy._core` and its `__all__`; a new format module is a file
`molpy/python/molpy/io/<fmt>.py` importing its classes from `molpy._core`, imported
by `molpy/python/molpy/io/__init__.py`, and its stub entries go into
`molpy/python/molpy/_core.pyi` (the `types` gate holds the stub to the module).

No Python function coerces paths, re-raises native errors under another type or
buffers frames for the native writer: the native doors take `os.PathLike`,
`bytes` and `mp.io.Text` sources themselves and report an unreadable file as
`OSError`.

## Behaviour belongs in the native door

A format's behaviour — merging inpcrd coordinates into an existing frame,
joining an n-wide XYZ property, Type Labels and `fix drude` flags of a LAMMPS
data file, `fix bond/react` maps, an AMBER prmtop's per-pair 1-4 weights
(`mp.io.amber.AmberPrmtopForcefieldReader`) — is the core's, so every caller
gets it. Python keeps no reader of its own, and a convenience that only composes native doors (a prmtop plus its inpcrd,
a SMILES string to a graph) is two native calls at the call site, not a Python
function.

## Canonical field names

The data model uses one column vocabulary, the core's: `mp.core.keys`
(`mp.core.keys.CHARGE.key == "charge"`), and `mp.core.schema` says
each column's dtype. Every native reader emits these names — it maps a
format's own spelling (`q`, `mol`, `resSeq`) at the boundary — and every
writer takes them:

```python
import molpy as mp

assert mp.core.keys.CHARGE.key == "charge"
assert mp.core.keys.MOL_ID.key == "mol_id"
```

Key canonical fields: `charge` (not `q`), `mol_id` (not `mol`), `id`, `type`,
`mass`, `element`, `x`/`y`/`z`.

## Force field readers and writers live in the Rust core

Every force-field reader and writer — LAMMPS `*.ff` includes and data-file
`* Coeffs`, GROMACS directives and systems, OpenMM XML, AMBER prmtop / frcmod
— is a native class in its format's submodule, `reader.read(source)` →
`ForceField` and `writer.write(sink, forcefield)`
(`mp.io.openmm_xml.OpenmmXmlReader` / `OpenmmXmlWriter`,
`mp.io.gromacs.GromacsTopForcefieldReader` / `GromacsTopForcefieldWriter`,
`mp.io.amber.AmberPrmtopForcefieldReader` / `AmberFrcmodWriter`,
`mp.io.lammps.LammpsForcefieldReader` / `LammpsForcefieldWriter`). The
LAMMPS writer is built on the system as well: the coefficients written are
selected by the frame's type labels, each matched to a type name exactly. A
file that states no pair cutoff (an OpenMM force field, a `.top`, a prmtop)
reads as a field with none; the caller declares it on the pair styles, and no
reader or writer invents one.

```python
import molpy as mp

ff = mp.io.openmm_xml.OpenmmXmlReader().read(mp.resources.get_path("forcefield/tip3p.xml"))
water = mp.Frame(
    blocks={
        "atoms": {
            "type": ["tip3p-O", "tip3p-H", "tip3p-H"],
            "x": [0.0, 0.9572, -0.24],
            "y": [0.0, 0.0, 0.927],
            "z": [0.0, 0.0, 0.0],
        },
        "bonds": {"atomi": [0, 0], "atomj": [1, 2], "type": ["tip3p-O::tip3p-H"] * 2},
    }
)
for style in ("lj/cut", "coul/cut"):
    ff.get_style("pair", style).set(cutoff=10.0)
buf = bytearray()  # in memory; a path writes the file
mp.io.lammps.LammpsForcefieldWriter(water, precision=4).write(buf, ff)
assert "bond_coeff" in buf.decode()
```

A new **style** is exported by what the core's writers make of its registration;
see [Extending the Force Field](extending-forcefield.md#engines).

## Checklist

- [ ] Parser and writer added to the Rust core (`molrs::io`, `crates/molrs/src/io/`) and bound in `molpy/src/io/`
- [ ] Round-trip tests (`write → read → compare`) in the Rust core; the binding's seam in `molpy/tests/test_bindings/`
- [ ] Box stored on `frame.box`; exact-dtype metadata stored on `frame.meta`
