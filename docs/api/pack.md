# Pack

Spatial packing of molecules into simulation boxes via
**[molpack](https://docs.molcrafts.org/molpack/)** (`molcrafts-molpack`).

Install separately:

```bash
pip install molcrafts-molpack
```

Packing is **not** a molpy runtime dependency and is **not** part of the docs
build. Examples below are frozen illustrations — full API lives in the
[molpack Python guide](https://docs.molcrafts.org/molpack/python/).

## Quick reference

| Symbol | Package | Preferred for |
|--------|---------|---------------|
| `GencanPack` | `molpack` | Rigid-body packing of several species |
| `Target` | `molpack` | One species + count + restraints |
| `Cuboid` / `Sphere` / `HalfSpace` | `molpy` (molrs regions) | Restraints: box, droplet, slab; combine with `&`, `\|`, `~` |
| `State` | `molpack` | `run` result: `.frame`, `.converged`, `.fdist`, … |
| `PackingTemplate` | `molpy.builder` | A built molecule's `.frame` plus the `.hydrogens` indices into it, for `Target(...).with_hydrogens(...)` / `.with_atom_radius(...)` |

## Canonical example

```python
# docs: skip — optional molcrafts-molpack; not a molpy runtime/doc dep
import molpy as mp
from molpack import GencanPack, Target

water = mp.Atomistic(name="water")
o = water.def_atom(element="O", x=0.0, y=0.0, z=0.0)
water.def_bond(o, water.def_atom(element="H", x=0.957, y=0.0, z=0.0))
water.def_bond(o, water.def_atom(element="H", x=-0.239, y=0.927, z=0.0))

ion = mp.Atomistic(name="sodium")
ion.def_atom(element="Na", x=0.0, y=0.0, z=0.0, charge=1.0)

box = mp.core.Cuboid([0.0, 0.0, 0.0], [30.0, 30.0, 30.0])
targets = [
 Target(water.to_frame(), count=100).with_name("water").with_restraint(box),
 Target(ion.to_frame(), count=10).with_name("na").with_restraint(box),
]
packed = GencanPack().with_seed(42).run(targets, max_loops=200).frame
```

A built polymer goes in through `mp.builder.PackingTemplate`, so the atom
indices a `Target` takes are read off the same frame it packs:

```python
# docs: skip — optional molcrafts-molpack; not a molpy runtime/doc dep
template = mp.builder.PackingTemplate(chain)  # chain: an assembled mp.Atomistic
target = (
    Target(template.frame, count=10)
    .with_restraint(box)
    .with_hydrogens(template.hydrogens)
    .with_atom_radius(template.hydrogens, 0.2)
)
```

## Related

- [Guide: Packing Systems](../user-guide/09_packing.md)
- [Guide: Assembly](../user-guide/02_assembly.md)
- [Guide: Polydisperse Systems](../user-guide/05_polydisperse_systems.md)
- [molpack documentation](https://docs.molcrafts.org/molpack/)
