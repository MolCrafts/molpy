# PMFT

Overview

| Class / entry | Description |
|---------------|-------------|
| [`PMFTXY`](#pmftxy) | 2-D PMFT on a local $(x,y)$ grid: $\mathrm{{PMFT}}=-k_B T\ln g$. |

Details

The `molpy.compute.pmft` module: potentials of mean force and torque.

## `PMFTXY`

2-D PMFT on a local $(x,y)$ grid: $\mathrm{{PMFT}}=-k_B T\ln g$.

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
from molpy.compute import NeighborList, PMFTXY

nlist = NeighborList(cutoff=3.0)(frame)
out = PMFTXY(x_max=5.0, y_max=5.0, n_x=20, n_y=20)(frame, nlist)
```

## See also

- [RDF](rdf.md)
- [Distribution](distribution.md)
