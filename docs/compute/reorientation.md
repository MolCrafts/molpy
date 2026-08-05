# Reorientation

Overview

| Class / entry | Description |
|---------------|-------------|
| [`LegendreReorientation`](#legendrereorientation) | Legendre $C_1$ / $C_2$ from bond vectors carried on each frame. |

Details

The `molpy.compute.reorientation` module: reorientational TCFs from the `bonds` block.

## `LegendreReorientation`

Legendre $C_1$ / $C_2$ from bond vectors carried on each frame.

```python
import molpy as mp

mol = mp.Atomistic()
atoms = [
    mol.def_atom(element="C", x=float(i), y=0.3 * (i % 2), z=0.0) for i in range(6)
]
for i in range(5):
    mol.def_bond(atoms[i], atoms[i + 1])
frame = mol.get_topo(gen_angle=True, gen_dihe=True).to_frame()
```

```python
from molpy.compute import LegendreReorientation

frames = [frame]  # time-ordered list in real use
C = LegendreReorientation(max_lag=2)(frames)
C.c1, C.c2
```

## See also

- [Van Hove](van_hove.md)
- [Dielectric](dielectric.md)
