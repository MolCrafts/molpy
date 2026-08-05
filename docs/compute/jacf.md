# JACF

Overview

| Class / entry | Description |
|---------------|-------------|
| [`GreenKuboConductivity`](#greenkuboconductivity) | Raw $\langle J(0)\cdot J(t)\rangle$. Integrate with `CumulativeTrapezoid`. |

Details

The `molpy.compute.jacf` module: Green–Kubo ionic conductivity (raw current ACF).

## `GreenKuboConductivity`

Raw $\langle J(0)\cdot J(t)\rangle$. Integrate with `CumulativeTrapezoid`.

```python
import numpy as np
from molpy.compute import GreenKuboConductivity, CumulativeTrapezoid

rng = np.random.default_rng(0)
J = np.ascontiguousarray(rng.normal(0, 1.0, size=(40, 3)))
jacf = GreenKuboConductivity().compute(J, 10.0, 15)
integ = CumulativeTrapezoid().fit(jacf["jacf"], dt=10.0)
integ["integral"]
```

## See also

- [PMSD](pmsd.md)
- [Dielectric](dielectric.md)
