# Pack

Spatial packing of molecules into periodic simulation boxes via Packmol.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `Molpack` | High-level packing interface | Multi-component systems |
| `InsideBoxConstraint` | Cubic/orthorhombic box constraint | Standard periodic boxes |
| `InsideSphereConstraint` | Spherical region constraint | Droplet / cluster geometries |
| `Target` | One molecule species + count + constraint | Defining packing targets |

## Canonical example

```python
import numpy as np
from molpy.pack import Molpack, InsideBoxConstraint

packer = Molpack(workdir="pack_output")
constraint = InsideBoxConstraint(
    length=np.array([30.0, 30.0, 30.0]),
    origin=np.zeros(3),
)
packer.add_target(water_frame, number=100, constraint=constraint)
packer.add_target(ion_frame, number=10, constraint=constraint)

packed = packer.optimize(max_steps=10000, seed=42)
```

## Related

- [Guide: Crosslinked Networks](../user-guide/04_crosslinking.ipynb)
- [Guide: Polydisperse Systems](../user-guide/05_polydisperse_systems.ipynb)

---

## Full API

### MolPack

::: molpy.pack.molpack

### Constraint

::: molpy.pack.constraint

### Target

::: molpy.pack.target

### Packer

::: molpy.pack.packer
