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

**A virtual-site builder copies your structure, selects the host atoms, builds the
extra sites, and redistributes charge — without mutating the input.**

## The builder protocol

Each builder is configured by its constructor and exposes one entry point,
`apply`:

```python
import molpy as mp
from molpy.builder import DrudeBuilder

struct, _ = mp.conformer.Conformer(seed=42).generate(mp.io.smiles.read_smiles("CCO").to_atomistic())
# DrudeBuilder looks up each atom's CL&P type and splits its charge, so the
# structure must be typed and charged first (normally by a typifier).
clp_types = {6: "CT", 8: "OH", 1: "HC"}  # by atomic number
for atom in struct.atoms:
 atom["type"] = clp_types[atom["atomic_number"]]
 atom["charge"] = 0.0
builder = DrudeBuilder(drude_prefix="D")

new_struct = builder.apply(struct) # struct: Atomistic -> Atomistic (a copy)
```

`apply` copies the structure, selects the hosts, makes the extra particles on
them and moves charge onto those particles; it is the builder's only method.

## Drude polarization (CL&Pol)

`DrudeBuilder` adds a Drude shell to every atom whose type has `k_D > 0` in a
table of per-atom-type polarizabilities:

```python
from molpy.builder import DrudeBuilder

alpha = DrudeBuilder().polarizability  # the CL&Pol alpha.ff table molpy ships
drude = DrudeBuilder(polarizability=alpha, drude_prefix="D")
polarized = drude.apply(struct)
```

| Parameter | Meaning |
|---|---|
| `polarizability` | The CL&Pol Drude table: `dict[type -> dict[param -> float]]` (`m_D`, `q_D_sign`, `k_D`, `alpha`, `a_thole`), or the list of rows `mp.io.read_clpol_alpha(path)` returns for a custom `alpha.ff` (one dict per row, keyed by `type_name`; a type given twice keeps its later row). `None` uses the shipped `alpha.ff` (`DrudeBuilder().polarizability`). |
| `drude_prefix` | Type prefix for the generated Drude particles (default `"D"`: core `NBT` gets shell type `DNBT`). |

The structure must be typed and charged first: `apply` raises `ValueError` if an
atom has no `type`, since its polarizability cannot be looked up, or if a
polarizable atom has no `charge` to split with its shell.

Each shell is an atom with `vsite == "drude"` (`atomic_number` 0: no element) tied to its
core by a row of the `drudes` relation (`atomi` the core, `atomj` the shell)
with style `harmonic` and the `drude/harmonic` parameters: `k` (kcal/mol/Å² in
the `k*r^2` form, i.e. `k_D / 2` converted from kJ), `alpha` (Å³) and
`thole` (CL&Pol's `a_thole`). It shows up as a `drudes` block in
`polarized.to_frame()`, and the LAMMPS data writer derives the `fix drude`
C/D/N flags from it.

## TIP4P M-sites

`Tip4pBuilder` places a massless M-site on each water's HOH bisector — the same
protocol, a different rule:

```python
from molpy.builder import Tip4pBuilder

water, _ = mp.conformer.Conformer(add_hydrogens=True, seed=1).generate(mp.io.smiles.read_smiles("O").to_atomistic())
# The M-site carries the oxygen's charge, so the input must already have one.
for atom in water.atoms:
 atom["charge"] = -0.834 if atom["atomic_number"] == 8 else 0.417

tip4p = Tip4pBuilder(d_om=0.1546) # O–M distance in Å
water4p = tip4p.apply(water)
```

`d_om` is the oxygen-to-M-site distance in Å; the default, 0.1546 Å, is the TIP4P/2005 geometry.

The M-site is recorded as a `virtual_sites` row (style `average3`), so its position stays defined by O and the two H. LAMMPS has no data-file form for such a site: `write_lammps_data` refuses a frame with `virtual_sites` rows. For LAMMPS, drop the M atoms, put the M charge back on O, and use a `*/tip4p/*` pair style, which places M implicitly.
Pass `box=` (a `mp.Box`) for a periodic system: the O–H vectors are then
minimum-image deltas, so a water split by the boundary still gets M on its
bisector. Hydrogens are found through the bond adjacency of each oxygen.

The M-site is an atom with `vsite == "massless"` (`atomic_number` 0: no element) and is
recorded as a row of the `virtual_sites` relation (`atomi` = M, `atomj`,
`atomk`, `atoml` = O, H, H) with style `average3` and weights `w1 = 1 - 2a`,
`w2 = w3 = a`, so an engine can rebuild M from the moving water.

## Pitfalls

- **`apply` returns a copy** — the original `struct` is unchanged; use the return
 value.
- Drude output needs a force field that understands the shell particles;
 pair it with the CL&Pol / polarizable typification path, not a plain
 fixed-charge field.
- Drude shells and M-sites (atoms with a `vsite` field) are auxiliary particles — downstream exporters
 and engines must be told to treat them as virtual sites, not atoms.
- The LAMMPS data writer writes each `drudes` spring as a row of the
 `Bonds` section (bond type `<core>-<shell>`, e.g. `CT-DCT`), as the LAMMPS
 DRUDE package expects.
- `mp.Atomistic.from_frame` reads `drudes` / `virtual_sites` blocks back, so
 the builder's output survives a round trip through its frame.

## See also

- [Force Field Typification](06_typifier.md) — assigning the types and charges
 a builder needs.
- [API Reference — Builder](../api/builder.md) — the virtual-site builders.
