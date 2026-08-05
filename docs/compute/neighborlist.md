# NeighborList

Overview

| Class / entry | Description |
|---------------|-------------|
| [`NeighborList`](#neighborlist) | Pairs within a cutoff. |

Details

The `molpy.compute.neighborlist` module: neighbor search for pair analyses.

## `NeighborList`

Pairs within a cutoff.

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
from molpy.compute import NeighborList

nlist = NeighborList(cutoff=5.0)(frame)
```

## See also

- [RDF](rdf.md)
- [Order](order.md)
