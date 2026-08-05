# Onsager

Overview

| Class / entry | Description |
|---------------|-------------|
| [`Onsager`](#onsager) | Cross-correlation $L_{{ij}}(\\tau)$ from unwrapped collective $P_\\alpha(t)$. |

Details

The `molpy.compute.onsager` module: Onsager collective transport coefficients.

## `Onsager`

Cross-correlation $L_{{ij}}(\\tau)$ from unwrapped collective $P_\\alpha(t)$.

```python
import numpy as np
from molpy.compute import Onsager

rng = np.random.default_rng(0)
P_i = np.ascontiguousarray(np.cumsum(rng.normal(0, 0.01, size=(40, 3)), axis=0))
P_j = np.ascontiguousarray(np.cumsum(rng.normal(0, 0.01, size=(40, 3)), axis=0))
L = Onsager.correlation(P_i, P_j, 10.0, 10)
L["lag_times"], L["correlation"]
```

## See also

- [MSD](msd.md)
- [PMSD](pmsd.md)
