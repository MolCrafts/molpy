# Unit Systems

A frame stores `x = 3.0`. Is that 3 Å or 3 nm? The array does not say.

**`UnitSystem` names the convention that gives bare numbers meaning**, and
converts explicitly when two conventions meet (for example LAMMPS `real` vs
`metal`).

What it is **not**: automatic unit tracking on every `Frame` column. Numbers stay
plain; *you* attach a convention when you convert or compare.

## Why conventions matter

A TIP3P field authored in nanometres expects nm coordinates; an OPLS field in
ångström expects Å. Mixing them silently produces wrong physics. MolPy does not
guess: the force field and the unit system you choose fix the interpretation.

## Using a preset

Named presets mirror common LAMMPS `units` lines:

```python
from molpy.core.unit import UnitSystem

print(UnitSystem.preset_names())
# ('real', 'metal', 'si', 'cgs', 'electron', 'micro', 'nano')

u = UnitSystem.preset("real") # LAMMPS 'real': Å, fs, kcal/mol, amu, e
length = 3.0 * u.angstrom
print(length.to(u.nanometer)) # 0.3 nanometer
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
a reduced (Lennard-Jones) unit system from reference `Quantity` values
(for example `39.948 * UnitSystem().amu`).

## Converting quantities

Quantities are `Quantity`. Multiply a number by a unit attribute, then
`.to(...)` to convert; `.magnitude` reads the bare number:

```python
u = UnitSystem.preset("metal")
e = 2.5 * u.eV
print(e.to("J")) # convert energy (per particle)
print((5 * u.angstrom).to("nm")) # convert length
print(e.magnitude) # 2.5
print((1.0 * u.kilocalorie_per_mole).to("eV").magnitude)
```

`UnitSystem` also exposes `parse`, `define`, `quantity`, and `convert` from the
native registry when you need to register extra units or convert against the
current LJ scales.

## Pitfalls

- **A `Frame`'s numbers still carry no unit.** `UnitSystem` converts *quantities*
 you build; it does not tag your coordinate arrays. Keep your inputs consistent
 with the force field's convention.
- **Match the force field.** If a field was authored in `real` (Å), don't feed it
 nm coordinates.
- `register_preset(..., overwrite=False)` raises if the name exists — pass
 `overwrite=True` deliberately.
- **Not Pint.** There is no `pint` runtime dependency and no Pint-only context
 API; unit math is the unit engine.

## See also

- [Naming Conventions](naming-conventions.md) — the column
 schema those unitless arrays follow.
- [Force Field](04_force_field.md) — where a convention becomes physical.
