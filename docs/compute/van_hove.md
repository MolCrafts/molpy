# Van Hove

Overview

| Class / entry | Description |
|---------------|-------------|
| [`VanHove`](#vanhove) | $G(r,t)$ — $g(r)$ extended in lag time. Args: `n_rbins`, `r_max`, `lags`. |

Details

The `molpy.compute.van_hove` module: time-resolved pair correlations $G(r,t)$.

## `VanHove`

$G(r,t)$ — $g(r)$ extended in lag time. Args: `n_rbins`, `r_max`, `lags`.

```python
import numpy as np
import molpy as mp
from molpy.compute import VanHove

def make_frame(seed):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0.0, 10.0, size=(20, 3))
    f = mp.Frame()
    f["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    f.box = mp.Box.cubic(10.0)
    return f

frames = [make_frame(i) for i in range(4)]
G = VanHove(n_rbins=10, r_max=5.0, lags=[0, 1, 2])(frames)
```

## See also

- [RDF](rdf.md)
- [Reorientation](reorientation.md)
