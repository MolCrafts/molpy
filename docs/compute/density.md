# Density

Overview

| Class / entry | Description |
|---------------|-------------|
| [`LocalDensity`](#localdensity) | Local number density in a sphere of radius `r_max`. Needs a neighbor list. |
| [`GaussianDensity`](#gaussiandensity) | Smear positions onto a grid with Gaussian kernels. |

Details

The `molpy.compute.density` module: local and grid densities.

## `LocalDensity`

Local number density in a sphere of radius `r_max`. Needs a neighbor list.

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
from molpy.compute import NeighborList, LocalDensity

nlist = NeighborList(cutoff=3.0)(frame)
dens = LocalDensity(r_max=3.0, diameter=0.0)(frame, nlist)
```

## `GaussianDensity`

Smear positions onto a grid with Gaussian kernels.

```python
from molpy.compute import GaussianDensity

grid = GaussianDensity(nx=8, ny=8, nz=8, sigma=1.0)(frame)
```

## See also

- [RDF](rdf.md)
- [Voronoi](voronoi.md)
