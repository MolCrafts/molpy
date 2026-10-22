# Adding an I/O Format

`mp.io` has one door per format and direction, named after the format:
`read_<fmt>[_<what>]` / `write_<fmt>[_<what>]` for a path (`read_pdb`,
`read_lammps_data`, `read_top_forcefield`), `_str` for text in memory
(`read_smiles_str`), `_bytes` for bytes and `_trajectory` for every frame of a
multi-frame file. No door picks a format for the caller from a file
extension. There are no reader or writer classes to subclass. Parsing and serialization belong in the
native core (molrs): every function on `mp.io` is native, force-field formats
included (`mp.ff.forcefield` is the `ForceField` data model only). A class
that belongs to one format lives in that format's submodule (`mp.io.smiles`,
`mp.io.lammps`, `mp.io.mrec`, …).

## A new format goes into molrs

Add the parser and writer to molrs and bind them in the Python binding
(`molpy/src/io/`): a function at the top of `mp.io` (`read_<fmt>` /
`write_<fmt>`, force-field formats included), a class in a `mp.io.<fmt>`
submodule. A new function is listed in `molpy/molpy/io/__init__.py`'s import
from `molpy._core` and its `__all__`; a new format module is a file
`molpy/molpy/io/<fmt>.py` importing its classes from `molpy._core`, imported
by `molpy/molpy/io/__init__.py`, and its stub entries go into
`molpy/molpy/_core.pyi` (the `types` gate holds the stub to the module).

No molpy function coerces paths, re-raises native errors under another type or
buffers frames for the native writer: the native doors accept `str` and
`os.PathLike`, take any sequence of frames and report an unreadable file as
`OSError`.

## Behaviour belongs in the native door

A format's behaviour — merging inpcrd coordinates into an existing frame,
joining an n-wide XYZ property, Type Labels and `fix drude` flags of a LAMMPS
data file, `fix bond/react` maps, an AMBER prmtop's per-pair 1-4 weights
(`mp.io.amber.AmberPrmtopForcefieldReader`) — is molrs's, so every caller
gets it. Python keeps no reader of its own, and a convenience that only composes native doors (a prmtop plus its inpcrd,
a SMILES string to a graph) is two native calls at the call site, not a molpy
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

## Force field readers and writers live in molrs

Every force-field reader and writer — LAMMPS `*.ff` includes and data-file
`* Coeffs`, GROMACS directives and systems, OpenMM XML, AMBER prmtop / frcmod
— is a native function on `mp.io`
(`read_openmm_forcefield` / `write_openmm_forcefield`, `read_top_forcefield` /
`write_top_forcefield`, the LAMMPS family). The
LAMMPS writers take the system as well: the coefficients written are selected by
the frame's type labels, each matched to a type name exactly. The pair cutoff is
a run setting the caller declares on the pair styles; no reader or writer
invents one.

```python
import molpy as mp

ff = mp.io.read_openmm_forcefield(mp.resources.get_path("forcefield/tip3p.xml"))
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
ff.get_style("pair", "lj/cut")["cutoff"] = 10.0
ff.get_style("pair", "coul/cut")["cutoff"] = 10.0
text = mp.io.write_lammps_forcefield_str(ff, water, precision=4)
assert "bond_coeff" in text
```

A new **style** is exported by what molrs's writers make of its registration;
see [Extending the Force Field](extending-forcefield.md#engines).

## Checklist

- [ ] Parser and writer added to molrs (`molpy.io`) and bound in `molpy/src/`
- [ ] Round-trip tests (`write → read → compare`) in molrs; the binding's seam in `molpy/tests/test_bindings/`
- [ ] Box stored on `frame.box`; exact-dtype metadata stored on `frame.meta`
