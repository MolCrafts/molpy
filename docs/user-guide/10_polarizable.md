# Polarizable & Virtual-Site Models

Drude shells and TIP4P M-sites put charge where no nucleus is. The
`molpy.builder.virtualsite` builders add these **virtual sites** to a copy of
your structure and redistribute charge automatically.

## What virtual sites are for

Some force-field models place interaction sites off the nuclei:

- **Drude oscillators** (CL&Pol) attach a mobile charged shell to a heavy atom to
  represent induced polarization.
- **TIP4P water** puts the negative charge on an off-atom **M-site** on the HOH
  bisector, not on the oxygen.

**A `VirtualSiteBuilder` copies your structure, selects the host atoms, builds the
extra sites, and redistributes charge — without mutating the input.**

## The builder protocol

Every builder follows the same four-step pipeline and exposes one entry point,
`apply`:

```python
new_struct = builder.apply(struct)      # struct: Atomistic -> Atomistic (a copy)
```

Internally `apply` runs `select` (which hosts?) → `build_sites` (make the extra
particles) → `redistribute` (move charge onto them). You normally only call
`apply`; the other three are extension hooks for custom builders.

## Drude polarization (CL&Pol)

`DrudeBuilder` adds a Drude shell (`DrudeParticle`) to every polarizable heavy
atom, driven by per-atom-type polarizabilities:

```python
from molpy.builder.virtualsite import DrudeBuilder, load_polarizability

alpha = load_polarizability()                 # bundled alpha.ff parameters
drude = DrudeBuilder(polarizability=alpha, drude_prefix="D")
polarized = drude.apply(struct)
```

| Parameter | Meaning |
|---|---|
| `polarizability` | `dict[type -> dict[param -> float]]` of Drude parameters. `None` falls back to the bundled `alpha.ff`; `load_polarizability(path)` reads a custom file. |
| `drude_prefix` | Name prefix for the generated Drude particles (default `"D"`). |

## TIP4P M-sites

`Tip4pBuilder` places a `MasslessSite` on each water's HOH bisector — the same
protocol, a different rule:

```python
from molpy.builder.virtualsite import Tip4pBuilder

tip4p = Tip4pBuilder(d_om=0.1546)     # O–M distance in nm
water4p = tip4p.apply(water)
```

`d_om` is the oxygen-to-M-site distance; the default matches the TIP4P geometry.

## Writing your own

Subclass `VirtualSiteBuilder` and implement `select`, `build_sites`, and
`redistribute`; `apply` orchestrates them and handles the copy. `DrudeBuilder`
and `Tip4pBuilder` are the two reference implementations.

## Pitfalls

- **`apply` returns a copy** — the original `struct` is unchanged; use the return
  value.
- Drude output needs a force field that understands the shell particles;
  pair it with the CL&Pol / polarizable typification path, not a plain
  fixed-charge field.
- `MasslessSite` / `DrudeParticle` are auxiliary particles — downstream exporters
  and engines must be told to treat them as virtual sites, not atoms.

## See also

- [Force Field Typification](06_typifier.md) — assigning parameters, including the
  CL&Pol path.
- [API Reference — Builder](../api/builder.md) — full `virtualsite` reference.
