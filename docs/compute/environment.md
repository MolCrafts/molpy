# Environment

Overview

| Class / entry | Description |
|---------------|-------------|
| [`BondOrder`](#bondorder) | Bond-direction histogram on a $(\\theta,\\phi)$ grid. |

Details

The `molpy.compute.environment` module: local-environment bond-order diagrams.

## `BondOrder`

Bond-direction histogram on a $(\\theta,\\phi)$ grid.

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
from molpy.compute import NeighborList, BondOrder

nlist = NeighborList(cutoff=3.0)(frame)
diagram = BondOrder(n_theta=6, n_phi=6)(frame, nlist)
```

## See also

- [Order](order.md)
- [NeighborList](neighborlist.md)
