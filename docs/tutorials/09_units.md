# Unit Systems

This page explains how MolPy represents physical units with `UnitSystem`, how it
relates to the (unitless) coordinates stored in a `Frame`, and how to pick or
define a LAMMPS-style unit convention.

## Coordinates are unitless; a convention gives them meaning

A `Frame` stores plain numbers: `x/y/z`, `mass`, `charge` are arrays with **no
attached unit**. MolPy does not force a length unit on you — the *convention* you
work in (and the force field you load) fixes what those numbers mean. A TIP3P
force field authored in nanometres expects nm coordinates; an OPLS field in
ångström expects Å. Mixing them silently produces wrong physics.

**`UnitSystem` is the object that names a convention and converts between units.**
It is a `pint.UnitRegistry` pre-loaded with MolPy/LAMMPS presets, so you get
`pint` quantities and conversions with the right base units already registered.

## Using a preset

`UnitSystem` ships the LAMMPS `units` conventions as named presets:

```python
from molpy.core.unit import UnitSystem

print(UnitSystem.preset_names())
# ('real', 'metal', 'si', 'cgs', 'electron', 'micro', 'nano')

u = UnitSystem.preset("real")      # LAMMPS 'real': Å, fs, kcal/mol, amu, e
length = 3.0 * u.angstrom
print(length.to(u.nanometer))      # 0.3 nanometer
```

| Preset | Convention (LAMMPS `units`) |
|---|---|
| `real` | Å, fs, kcal/mol, amu, e — the OPLS/AMBER default. |
| `metal` | Å, ps, eV, amu, e. |
| `si` / `cgs` | SI / CGS base units. |
| `electron` | atomic (Hartree) units. |
| `micro` / `nano` | micro- and nano-scale presets. |

## Defining your own preset

Register a custom convention once, then reuse it by name:

```python
UnitSystem.register_preset(
    "my_units",
    base_units={"length": "nm", "time": "ps", "energy": "kJ/mol", "mass": "amu"},
    overwrite=False,
)
u = UnitSystem.preset("my_units")
```

- `base_units` maps each physical dimension to a unit string.
- `overwrite=False` refuses to clobber an existing preset (set `True` to replace).

For coarse-grained work, `UnitSystem.lj(mass=..., sigma=..., epsilon=...)` builds
a reduced (Lennard-Jones) unit system from your reference `pint` quantities.

## Converting quantities

Because a `UnitSystem` *is* a `pint` registry, you get the full `pint` API —
attach units, convert, and check dimensionality:

```python
u = UnitSystem.preset("metal")
e = 2.5 * u.eV
print(e.to("J"))                   # convert energy (per particle)
print((5 * u.angstrom).to("nm"))   # convert length

# eV is per particle; scale by Avogadro's number to reach a molar energy
print((e * u.avogadro_constant).to("kJ/mol"))   # -> 241.2 kJ/mol
```

## Pitfalls

- **A `Frame`'s numbers still carry no unit.** `UnitSystem` converts *quantities*
  you build; it does not tag your coordinate arrays. Keep your inputs consistent
  with the force field's convention.
- **Match the force field.** If a field was authored in `real` (Å), don't feed it
  nm coordinates.
- `register_preset(..., overwrite=False)` raises if the name exists — pass
  `overwrite=True` deliberately.

## See also

- [Naming Conventions](naming-conventions.md) — the column
  schema those unitless arrays follow.
- [Force Field](04_force_field.md) — where a convention becomes physical.
