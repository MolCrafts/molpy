# Spatial

Overview

| Class / entry | Description |
|---------------|-------------|
| [`SpatialDistribution`](#spatialdistribution) | Target-atom density on a molecule body-fixed grid (SDF). |

Details

The `molpy.compute.spatial` module: 3-D spatial distribution functions.

## `SpatialDistribution`

Target-atom density on a molecule body-fixed grid (SDF).

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
import numpy as np
from molpy.compute import SpatialDistribution

# Body-fixed frame from atoms 0,1,2; density of remaining atoms.
template = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
)
sdf = SpatialDistribution(
    reference=[0, 1, 2],
    template=template,
    target=list(range(3, 40)),
    n=(8, 8, 8),
    extent=(6.0, 6.0, 6.0),
)
res = sdf([frame])
```

## See also

- [Distribution](distribution.md)
