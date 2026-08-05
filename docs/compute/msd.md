# MSD

Overview

| Class / entry | Description |
|---------------|-------------|
| [`MSD`](#msd) | MSD of particle positions (**unwrapped**). `method=\"window\"` averages every origin. |

Details

The `molpy.compute.msd` module: mean squared displacement (self-diffusion).

## `MSD`

MSD of particle positions (**unwrapped**). `method=\"window\"` averages every origin.

```python
import numpy as np
import molpy as mp
from molpy.compute import MSD, LinearFit

def make_frame(seed):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0.0, 10.0, size=(20, 3))
    f = mp.Frame()
    f["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    f.box = mp.Box.cubic(10.0)
    return f

frames = [make_frame(i) for i in range(8)]
series = MSD(method="window")(frames)
# Optional slope fit when you have lag_times on the result:
# fit = LinearFit(0.1, 0.5).fit(lag_times, series.mean)
series.mean.shape
```

## See also

- [PMSD](pmsd.md)
- [Onsager](onsager.md)
