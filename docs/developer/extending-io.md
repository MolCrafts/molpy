# Adding an I/O Format

`mp.io` has one door per format and direction: `read_X` / `write_X` for one
frame, `read_X_trajectory` / `write_X_trajectory` for a sequence. There are no
reader or writer classes to subclass. Parsing and serialization belong in the
native core (molrs): `mp.io` is `molrs.io` re-exported by identity, and
force-field formats are `mp.ff.forcefield`, which is `molrs.ff.forcefield`.

## A new format goes into molrs

Add the parser and writer to molrs and bind them in molrs-python: a
structure or trajectory format under `molrs.io`, a force-field format under
`molrs.ff.forcefield`. molpy picks the new names up with no edit, since its
modules re-export the native ones wholesale:

```python
import molrs
import molpy as mp

assert mp.io.read_gro is molrs.io.read_gro
assert (
    mp.ff.forcefield.write_lammps_forcefield
    is molrs.ff.forcefield.write_lammps_forcefield
)
```

No molpy wrapper coerces paths, re-raises native errors under another type or
buffers frames for the native writer: the native doors accept `str` and
`os.PathLike`, take any sequence of frames and report an unreadable file as
`OSError`.

## Behaviour belongs in the native door

A format's behaviour — merging inpcrd coordinates into an existing frame,
joining an n-wide XYZ property, Type Labels and `fix drude` flags of a LAMMPS
data file, `fix bond/react` maps, an AMBER prmtop's per-pair 1-4 weights
(`mp.ff.forcefield.read_amber_prmtop_system`) — is molrs's, so every caller
gets it. molpy keeps no reader of its own: `mp.io` is `molrs.io` by identity,
and a convenience that only composes native doors (a prmtop plus its inpcrd,
a SMILES string to a graph) is two native calls at the call site, not a molpy
function.

## Canonical field names

The data model uses one column vocabulary, molrs's: `mp.keys` is
`molrs.store.keys` (`mp.keys.CHARGE.key == "charge"`), and `mp.schema` says
each column's dtype. Every native reader emits these names — it maps a
format's own spelling (`q`, `mol`, `resSeq`) at the boundary — and every
writer takes them:

```python
import molpy as mp

assert mp.keys.CHARGE.key == "charge"
assert mp.keys.MOL_ID.key == "mol_id"
```

Key canonical fields: `charge` (not `q`), `mol_id` (not `mol`), `id`, `type`,
`mass`, `element`, `x`/`y`/`z`.

## Force field readers and writers live in molrs

Every force-field reader and writer — LAMMPS `*.ff` includes and data-file
`* Coeffs`, GROMACS directives and systems, OpenMM XML, AMBER prmtop / frcmod
— is a native molrs function on `mp.ff.forcefield`, by identity
(`read_forcefield_xml` / `write_forcefield_xml`, `read_gromacs_top_ff` /
`write_gromacs_top_ff`, the LAMMPS family). The
LAMMPS writers take the system as well: the coefficients written are selected by
the frame's type labels, each matched to a type name exactly. The pair cutoff is
a run setting the caller declares on the pair styles; no reader or writer
invents one.

```python
import molpy as mp

ff = mp.ff.forcefield.read_forcefield_xml(mp.data.get_forcefield_path("tip3p.xml"))
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
text = mp.ff.forcefield.write_lammps_forcefield_str(ff, water, precision=4)
assert "bond_coeff" in text
```

A new **style** is exported by what molrs's writers make of its registration;
see [Extending the Force Field](extending-forcefield.md#engines).

## Checklist

- [ ] Parser and writer added to molrs (`molrs.io` or `molrs.ff.forcefield`) and bound in molrs-python
- [ ] Round-trip tests (`write → read → compare`) in molrs; molpy needs no edit
- [ ] Box stored on `frame.box`; exact-dtype metadata stored on `frame.meta`
