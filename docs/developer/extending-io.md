# Adding an I/O Format

`mp.io` has one door per format and direction: `read_X` / `write_X` for one
frame, `read_X_trajectory` / `write_X_trajectory` for a sequence. There are no
reader or writer classes to subclass. Parsing and serialization belong in the
native core (molrs); molpy re-exports the native door by identity, and writes a
function of its own only when it adds behaviour the native door lacks.

## A new format goes into molrs

Add the parser and writer to molrs, bind them in molrs-python, and re-export the
bound functions from `molpy/io/__init__.py` by identity:

```python
import molrs
import molpy as mp

assert mp.io.read_gro is molrs.io.read_gro
assert mp.io.write_lammps_forcefield is molrs.ff.write_lammps_forcefield
```

No molpy wrapper coerces paths, re-raises native errors under another type or
buffers frames for the native writer: the native doors accept `str` and
`os.PathLike`, take any sequence of frames and report an unreadable file as
`OSError`.

## When molpy adds behaviour

A molpy function is justified by behaviour, not by spelling. The current ones
live in `molpy/io/readers.py`, `molpy/io/writers.py` and
`molpy/io/data/lammps.py`: merging inpcrd coordinates into an existing frame,
dropping the duplicated CONECT bonds of a PDB, joining split XYZ property
columns, the `LammpsDataResult` bundle and the `fix drude` header of a LAMMPS
data file. Such a function calls the native door and post-processes its frame.

## Canonical field names

The internal data model uses canonical field names, listed on `mp.fields`. The native readers already translate the formats they
parse. A format whose column names molpy translates itself declares a
`FieldFormatter` subclass with a `_field_formatters` mapping:

```python
from molpy import fields


class MyFieldFormatter(fields.FieldFormatter):
    _field_formatters = {
        "q": fields.CHARGE,  # format "q" → canonical "charge"
        "mol": fields.MOL_ID,  # format "mol" → canonical "mol_id"
    }
```

Key canonical fields: `charge` (not `q`), `mol_id` (not `mol`), `id`, `type`,
`mass`, `element`, `x`/`y`/`z`.

## Force field readers and writers live in molrs

Every force-field reader and writer — LAMMPS `*.ff` includes and data-file
`* Coeffs`, GROMACS directives, OpenMM XML — is a native molrs function that
`mp.io` re-exports by identity (`read_xml_forcefield` / `write_xml_forcefield`,
`read_gromacs_forcefield` / `write_gromacs_forcefield`, the LAMMPS family). The
LAMMPS writers take the system as well: the coefficients written are selected by
the frame's type labels, each matched to a type name exactly. The pair cutoff is
a run setting the caller declares on the pair styles; no reader or writer
invents one.

```python
import molpy as mp

ff = mp.io.read_xml_forcefield(mp.data.get_forcefield_path("tip3p.xml"))
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

- [ ] Parser and writer added to molrs and bound in molrs-python
- [ ] `read_X` / `write_X` (and `_trajectory`) re-exported by identity from `molpy/io/__init__.py`
- [ ] A molpy function only where it adds behaviour, calling the native door
- [ ] Box stored on `frame.box`; exact-dtype metadata stored on `frame.meta`
- [ ] Round-trip tests (`write → read → compare`) of molpy-owned behaviour in `tests/test_io/`
