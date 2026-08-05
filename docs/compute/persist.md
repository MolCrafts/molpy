# Persist

Overview

| Class / entry | Description |
|---------------|-------------|
| [`Persist`](#persist) | Pair-survival TCF within a distance cutoff. |

Details

The `molpy.compute.persist` module: pair persistence / residence times.

## `Persist`

Pair-survival TCF within a distance cutoff.

```python
import numpy as np
from molpy.compute import Persist

coords_i = np.zeros((4, 1, 3))
coords_j = np.zeros((4, 1, 3))
coords_j[:, 0, 0] = 1.0  # 1 Å away
box = np.tile(np.array([[10.0, 10.0, 10.0]]), (4, 1))
out = Persist.pair_survival_tcf(
    coords_i, coords_j, box, 0.5, 3.5, "intermittent", 1.0, 2, False
)
out["correlation"]
```

## See also

- [HBond](hbond.md)
- [RDF](rdf.md)
