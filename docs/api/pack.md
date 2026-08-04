# Pack

Spatial packing of molecules into simulation boxes via
**[molpack](https://molcrafts.github.io/molpack/)** (`molcrafts-molpack`).

Install separately:

```bash
pip install molcrafts-molpack
```

Packing is **not** a molpy runtime dependency and is **not** part of the docs
build. Examples below are frozen illustrations — full API lives in the
[molpack Python guide](https://molcrafts.github.io/molpack/python/).

## Quick reference

| Symbol | Package | Preferred for |
|--------|---------|---------------|
| `Molpack` | `molpack` | Multi-component packing session |
| `Target` | `molpack` | One species + count + restraints |
| `InsideBoxRestraint` | `molpack` | Axis-aligned box confinement |
| `InsideSphereRestraint` | `molpack` | Droplet / cluster geometries |
| `PackResult` | `molpack` | Diagnostics from `pack_with_report` |

## Canonical example

```python
# docs: skip — optional molcrafts-molpack; not a molpy runtime/doc dep
import molpy as mp
from molpack import InsideBoxRestraint, Molpack, Target

water = mp.Atomistic(name="water")
o = water.def_atom(element="O", x=0.0, y=0.0, z=0.0)
water.def_bond(o, water.def_atom(element="H", x=0.957, y=0.0, z=0.0))
water.def_bond(o, water.def_atom(element="H", x=-0.239, y=0.927, z=0.0))

ion = mp.Atomistic(name="sodium")
ion.def_atom(element="Na", x=0.0, y=0.0, z=0.0, charge=1.0)

box = InsideBoxRestraint([0.0, 0.0, 0.0], [30.0, 30.0, 30.0])
targets = [
    Target(water.to_frame(), count=100).with_name("water").with_restraint(box),
    Target(ion.to_frame(), count=10).with_name("na").with_restraint(box),
]
packed = Molpack().with_seed(42).pack(targets, max_loops=200)
```

## Related

- [Guide: Packing Systems](../user-guide/09_packing.md)
- [Guide: Assembly](../user-guide/02_assembly.md)
- [Guide: Polydisperse Systems](../user-guide/05_polydisperse_systems.md)
- [molpack documentation](https://molcrafts.github.io/molpack/)
