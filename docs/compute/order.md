# Order

Overview

| Class / entry | Description |
|---------------|-------------|
| [`Steinhardt`](#steinhardt) | Bond-orientational $q_\ell$ / $w_\ell$. `l` is a sequence of degrees. |
| [`Hexatic`](#hexatic) | 2-D hexatic order $\psi_k$. |
| [`Nematic`](#nematic) | Nematic order $S$ and director (reads orientation axes from the frame when present). |
| [`SolidLiquid`](#solidliquid) | Per-particle solid/liquid labels from bond-order correlation. |

Details

The `molpy.compute.order` module: bond-orientational and orientational order.

## `Steinhardt`

Bond-orientational $q_\ell$ / $w_\ell$. `l` is a sequence of degrees.

```python
import numpy as np
import molpy as mp

rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 10.0, size=(40, 3))
frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box.cubic(10.0)
```

```python
from molpy.compute import NeighborList, Steinhardt

nlist = NeighborList(cutoff=3.0)(frame)
q = Steinhardt([4, 6], average=True)(frame, nlist)
```

## `Hexatic`

2-D hexatic order $\psi_k$.

```python
from molpy.compute import Hexatic

psi = Hexatic(6)(frame, nlist)
```

## `Nematic`

Nematic order $S$ and director (reads orientation axes from the frame when present).

```python
import numpy as np
from molpy.compute import Nematic

# Per-particle orientation axes: (head, tail) atom-index pairs
n = len(frame["atoms"]["x"])
idx = np.arange(n, dtype=np.uint32)
frame["orientations"] = {"atomi": idx, "atomj": (idx + 1) % n}

order, eigenvalues, director, q_tensor = Nematic()(frame)
```

## `SolidLiquid`

Per-particle solid/liquid labels from bond-order correlation.

```python
from molpy.compute import SolidLiquid

labels = SolidLiquid(6, q_threshold=0.7, n_threshold=6)(frame, nlist)
```

## See also

- [Environment](environment.md)
- [NeighborList](neighborlist.md)
