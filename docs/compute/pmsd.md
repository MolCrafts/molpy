# PMSD

Overview

| Class / entry | Description |
|---------------|-------------|
| [`EinsteinConductivity`](#einsteinconductivity) | Raw MSD of $M(t)=\sum q r$. Fit with `LinearFit`; SI-scale yourself. |

Details

The `molpy.compute.pmsd` module: Einstein–Helfand ionic conductivity (raw PMSD of $M(t)$).

## `EinsteinConductivity`

Raw MSD of $M(t)=\sum q r$. Fit with `LinearFit`; SI-scale yourself.

```python
import numpy as np
from molpy.compute import EinsteinConductivity, LinearFit

rng = np.random.default_rng(0)
M = np.ascontiguousarray(np.cumsum(rng.normal(0, 0.01, size=(40, 3)), axis=0))
raw = EinsteinConductivity().compute(M, 10.0, 15)
fit = LinearFit(0.1, 0.5).fit(raw["lag_times"], raw["msd"])
fit["slope"]
```

## See also

- [JACF](jacf.md)
- [Dielectric](dielectric.md)
- [MSD](msd.md)
