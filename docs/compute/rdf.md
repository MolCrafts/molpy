# RDF

Overview

| Class / entry | Description |
|---------------|-------------|
| [`RDF`](#rdf) | Radial distribution $g(r)$ from a `NeighborList`. Keep $r_\max \le L/2$; neighbor cutoff $\ge r_\max$. |

Details

The `molpy.compute.rdf` module: radial distribution $g(r)$.

## `RDF`

Radial distribution $g(r)$ from a `NeighborList`. Keep $r_\max \le L/2$; neighbor cutoff $\ge r_\max$.

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
from molpy.compute import NeighborList, RDF

nlist = NeighborList(cutoff=5.0)(frame)
result = RDF(n_bins=40, r_max=5.0)([frame], [nlist])
result.rdf, result.bin_centers
```

## See also

- [NeighborList](neighborlist.md)
- [Diffraction](diffraction.md)
- [Density](density.md)
