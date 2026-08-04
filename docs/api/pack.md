# Pack

Spatial packing of molecules into periodic simulation boxes via Packmol.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `Packmol` | High-level packing interface | Multi-component systems |
| `InsideBoxConstraint` | Cubic/orthorhombic box constraint | Standard periodic boxes |
| `InsideSphereConstraint` | Spherical region constraint | Droplet / cluster geometries |
| `Target` | One molecule species + count + constraint | Defining packing targets |

## Canonical example

```python
# docs: skip — Packmol binary; pack unit-tested with mocks and script literals
import numpy as np
import molpy as mp
from molpy.pack import Packmol, InsideBoxConstraint

water = mp.Atomistic(name="water")
o = water.def_atom(element="O", x=0.0, y=0.0, z=0.0)
water.def_bond(o, water.def_atom(element="H", x=0.957, y=0.0, z=0.0))
water.def_bond(o, water.def_atom(element="H", x=-0.239, y=0.927, z=0.0))
water_frame = water.to_frame()

ion = mp.Atomistic(name="sodium")
ion.def_atom(element="Na", x=0.0, y=0.0, z=0.0, charge=1.0)
ion_frame = ion.to_frame()

packer = Packmol(workdir="pack_output")
constraint = InsideBoxConstraint(
    length=np.array([30.0, 30.0, 30.0]),
    origin=np.zeros(3),
)
packer.def_target(water_frame, number=100, constraint=constraint)
packer.def_target(ion_frame, number=10, constraint=constraint)

packed = packer(max_steps=10000, seed=42)
```

## Related

- [Guide: Assembly](../user-guide/02_assembly.md)
- [Guide: Polydisperse Systems](../user-guide/05_polydisperse_systems.md)

---

## Full API

### Constraint

::: molpy.pack.constraint

### Target

::: molpy.pack.target

### Packer

::: molpy.pack.packer
