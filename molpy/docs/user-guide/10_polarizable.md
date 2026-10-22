# Polarizable & Virtual-Site Models

Drude shells and TIP4P M-sites put charge where no nucleus is. The
`mp.builder` virtual-site builders add these **virtual sites** to a copy of
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
import molpy as mp
from molpy.builder import DrudeBuilder

struct, _ = mp.conformer.Conformer(seed=42).generate(mp.io.smiles.read_smiles("CCO").to_atomistic())
# DrudeBuilder looks up each atom's CL&P type and splits its charge, so the
# structure must be typed and charged first (normally by a typifier).
clp_types = {"C": "CT", "O": "OH", "H": "HC"}
for atom in struct.atoms:
 atom["type"] = clp_types[atom["element"]]
 atom["charge"] = 0.0
builder = DrudeBuilder(drude_prefix="D")

new_struct = builder.apply(struct) # struct: Atomistic -> Atomistic (a copy)
```

Internally `apply` runs `select` (which hosts?) → `build_sites` (make the extra
particles) → `redistribute` (move charge onto them). You normally only call
`apply`; the other three are extension hooks for custom builders.

## Drude polarization (CL&Pol)

`DrudeBuilder` adds a Drude shell (`DrudeParticle`) to every polarizable heavy
atom, driven by per-atom-type polarizabilities:

```python
from molpy.builder import DrudeBuilder

alpha = mp.ff.params.clpol_polarizability()  # the CL&Pol alpha.ff table molrs ships
drude = DrudeBuilder(polarizability=alpha, drude_prefix="D")
polarized = drude.apply(struct)
```

| Parameter | Meaning |
|---|---|
| `polarizability` | The CL&Pol Drude table: `dict[type -> dict[param -> float]]` (`m_D`, `q_D_sign`, `k_D`, `alpha`, `a_thole`), or the list of rows `mp.io.read_clpol_alpha(path)` returns for a custom `alpha.ff` (one dict per row, keyed by `type_name`; a type given twice keeps its later row). `None` uses `mp.ff.params.clpol_polarizability()`, molrs's `alpha.ff`. |
| `drude_prefix` | Type prefix for the generated Drude particles (default `"D"`: core `NBT` gets shell type `DNBT`). |

The structure must be typed first: `apply` raises `ValueError` if an atom has
no `type`, since its polarizability cannot be looked up.

Each shell is a `DrudeParticle` (`vsite == "drude"`, no `element`) tied to its
core by a row of molrs's `drudes` relation (`atomi` the core, `atomj` the shell)
with style `harmonic` and the `drude/harmonic` parameters: `k` (kcal/mol/Å² in
molrs's `k*r^2` form, i.e. `k_D / 2` converted from kJ), `alpha` (Å³) and
`thole` (CL&Pol's `a_thole`). It shows up as a `drudes` block in
`polarized.to_frame()`, and molrs's LAMMPS data writer derives the `fix drude`
C/D/N flags from it.

## TIP4P M-sites

`Tip4pBuilder` places a `MasslessSite` on each water's HOH bisector — the same
protocol, a different rule:

```python
from molpy.builder import Tip4pBuilder

water, _ = mp.conformer.Conformer(add_hydrogens=True, seed=1).generate(mp.io.smiles.read_smiles("O").to_atomistic())
# The M-site carries the oxygen's charge, so the input must already have one.
for atom in water.atoms:
 atom["charge"] = -0.834 if atom["element"] == "O" else 0.417

tip4p = Tip4pBuilder(d_om=0.1546) # O–M distance in Å
water4p = tip4p.apply(water)
```

`d_om` is the oxygen-to-M-site distance in Å; the default, 0.1546 Å, is the TIP4P/2005 geometry.

The M-site is recorded as a `virtual_sites` row (style `average3`), so its position stays defined by O and the two H. LAMMPS has no data-file form for such a site: `write_lammps_data` refuses a frame with `virtual_sites` rows. For LAMMPS, drop the M atoms, put the M charge back on O, and use a `*/tip4p/*` pair style, which places M implicitly.
Pass `box=` (a `mp.Box`) for a periodic system: the O–H vectors are then
minimum-image deltas, so a water split by the boundary still gets M on its
bisector. Hydrogens are found through the bond adjacency of each oxygen.

The M-site is a `MasslessSite` (`vsite == "massless"`, no `element`) and is
recorded as a row of molrs's `virtual_sites` relation (`atomi` = M, `atomj`,
`atomk`, `atoml` = O, H, H) with style `average3` and weights `w1 = 1 - 2a`,
`w2 = w3 = a`, so an engine can rebuild M from the moving water.

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
- molrs's LAMMPS data writer does not yet write the `drudes` springs into the
 `Bonds` section, which the LAMMPS DRUDE package expects.
- `mp.Atomistic.from_frame` does not yet read `drudes` / `virtual_sites` blocks
 back; keep the builder's output graph, or its frame, rather than round-tripping.

## See also

- [Force Field Typification](06_typifier.md) — assigning parameters, including the
 CL&Pol path.
- [API Reference — Builder](../api/builder.md) — full `virtualsite` reference.
