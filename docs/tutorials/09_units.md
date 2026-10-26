# Unit Systems

A frame stores `x = 3.0`. Is that 3 Å or 3 nm? The array does not say; the
convention does: a `Frame`'s lengths are always Å, and a reader of an nm format
(GROMACS) converts at its boundary.

**A unit preset names the convention that gives bare numbers meaning**, and a
unit registry converts explicitly when two conventions meet (for example
LAMMPS `real` vs `metal`). Both live in the core: `mp.core.UnitPreset` and
`mp.core.UnitRegistry` are `molpy.core.UnitPreset` and `molpy.core.UnitRegistry`.

What it is **not**: automatic unit tracking on every `Frame` column. Numbers stay
plain; *you* attach a convention when you convert or compare.

## Why conventions matter

A TIP3P field authored in nanometres and an OPLS field authored in ångström
write the same lengths as different numbers. Mixing them silently produces wrong
physics. MolPy does not guess: a `ForceField` declares its unit preset
(`ForceField.units`), every force-field reader but LAMMPS's converts the file
into `real` (Å, kcal/mol), and the LAMMPS reader keeps the file's `units`.

## Using a preset

Named presets mirror the LAMMPS `units` styles, plus `openmm`:

```python
import molpy as mp

print(mp.core.UnitPreset.names())
# ['cgs', 'electron', 'lj', 'metal', 'micro', 'nano', 'openmm', 'real', 'si']

real = mp.core.UnitPreset("real")  # LAMMPS 'real': Å, fs, kcal/mol, amu, e
print(real.length(), real.energy())  # angstrom kilocalorie_per_mole
print(real.boltzmann())  # k_B in kcal/(mol·K)

u = mp.core.UnitRegistry()
length = 3.0 * u.parse(real.length())
print(length.to(u.nanometer))  # 0.3 nanometer
```

| Preset | Convention |
|---|---|
| `real` | Å, fs, kcal/mol, amu, e — the OPLS/AMBER default. |
| `metal` | Å, ps, eV, amu, e. |
| `si` / `cgs` | SI / CGS base units. |
| `electron` | atomic (Hartree) units. |
| `micro` / `nano` | micro- and nano-scale presets. |
| `openmm` | nm, ps, kJ/mol, amu, e — OpenMM's native units (not a LAMMPS `units` style). |

## Defining your own preset

Register a custom convention once, then reuse it by name. A preset names one
unit for each of its ten dimensions and carries its two constants:

```python
real = mp.core.UnitPreset("real")
dims = ("mass", "length", "time", "energy", "temperature", "charge",
        "pressure", "velocity", "force", "density")
units = {dim: getattr(real, dim)() for dim in dims}
mine = mp.core.UnitPreset.register(
    "my_units",
    {**units, "length": "nanometer"},
    boltzmann=real.boltzmann(),
    coulomb=real.coulomb(),
)
print(mp.core.UnitPreset("my_units").length())  # nanometer
```

`overwrite=False` (the default) refuses to replace an existing preset.

For coarse-grained work, `UnitRegistry.define_lj_units(mass, sigma, epsilon)`
adds the reduced (Lennard-Jones) units `lj_sigma`, `lj_tau`,
`lj_epsilon_over_kB`, … to a registry from reference `Quantity` values:

```python
argon = mp.core.UnitRegistry()
argon.define_lj_units(
    39.948 * argon.amu, 3.405 * argon.angstrom, 0.2381 * argon.kilocalorie_per_mole
)
print((1.0 * argon.lj_tau).to(argon.ps).value)  # ≈ 2.16
```

## Converting quantities

Quantities are `mp.core.Quantity`. Multiply a number by a unit, then `.to(...)` to
convert; `.value` reads the bare number. The Boltzmann constant is the
unit `k_B`:

```python
u = mp.core.UnitRegistry()
e = 2.5 * u.eV
print(e.to("J"))  # convert energy (per particle)
print((5 * u.angstrom).to("nm"))  # convert length
print(e.value)  # 2.5
print((1.0 * u.kilocalorie_per_mole).to("eV").value)
print((1.0 * u.k_B).to("electron_volt / kelvin").value)  # 8.617e-05
```

A unit only one registry knows (an LJ scale, a `define`d unit) converts with
that registry's unit object: `q.to(argon.lj_sigma)`, or
`q.to(argon.parse("lj_sigma"))`.

## Pitfalls

- **A `Frame`'s numbers still carry no unit.** The registry converts
  *quantities* you build; it does not tag your coordinate arrays. Keep your
  inputs consistent with the force field's convention.
- **Match the force field.** A number you supply by hand (a cutoff, a
  parameter) is in the force field's preset (`ff.units`); coordinates are Å.
- `UnitPreset.register(..., overwrite=False)` raises if the name exists — pass
  `overwrite=True` deliberately.
- **Not Pint.** There is no `pint` runtime dependency and no Pint-only context
  API; unit math is the unit engine.

## See also

- [Naming Conventions](naming-conventions.md) — the column
  schema those unitless arrays follow.
- [Force Field](04_force_field.md) — where a convention becomes physical.
