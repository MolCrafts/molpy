# Dielectric

Overview

| Class / entry | Description |
|---------------|-------------|
| [`Dielectric`](#dielectric) | Static permittivity and dipole assembly helpers. |
| [`DebyeRelaxation`](#debyerelaxation) | Raw dipole ACF for Debye-type analysis. |
| [`DebyeFit`](#debyefit) | Single-$\tau$ Debye fit on a **normalized** $\Phi(t)$. |
| [`EinsteinHelfandSpectrum`](#einsteinhelfandspectrum) | EH transform of a raw dipole ACF → $\varepsilon^*(\omega)$. |
| [`GreenKuboSpectrum`](#greenkubospectrum) | GK transform of a raw current ACF → $\varepsilon^*(\omega)$. |
| [`LinearFit`](#linearfit) | Linear slope fit over a fractional lag window. |
| [`CumulativeTrapezoid`](#cumulativetrapezoid) | Trapezoidal cumulative integral of an ACF. |

Details

The `molpy.compute.dielectric` module: dielectric permittivity and related spectra. Compose **raw → fit → SI scale**.

## `Dielectric`

Static permittivity and dipole assembly helpers.

```python
import numpy as np
from molpy.compute import Dielectric

dm = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
eps = Dielectric.static_dielectric_constant(dm, 1000.0, 300.0, 1.0)

charges = np.array([1.0, -1.0])
pos = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
M = Dielectric.compute_dipole_moment(charges, pos)
```

## `DebyeRelaxation`

Raw dipole ACF for Debye-type analysis.

```python
import numpy as np
from molpy.compute import DebyeRelaxation

rng = np.random.default_rng(0)
M = np.ascontiguousarray(np.cumsum(rng.normal(0, 0.01, size=(40, 3)), axis=0))
raw = DebyeRelaxation(volume=1000.0, temperature=300.0).compute(M, 10.0, 10)
raw["acf"], raw["zero_lag_variance"]
```

## `DebyeFit`

Single-$\tau$ Debye fit on a **normalized** $\Phi(t)$.

```python
from molpy.compute import DebyeFit

phi = raw["acf"] / raw["zero_lag_variance"]
fit = DebyeFit().fit(phi, 10.0)  # sample step dt in fs
fit["tau"], fit["amplitude"]
```

## `EinsteinHelfandSpectrum`

EH transform of a raw dipole ACF → $\varepsilon^*(\omega)$.

```python
from molpy.compute import EinsteinHelfandSpectrum

eh = EinsteinHelfandSpectrum(
    dt=10.0,
    volume=1000.0,
    temperature=300.0,
    epsilon_inf=1.0,
    zero_lag_variance=float(raw["zero_lag_variance"]),
)
spec = eh.fit(raw["acf"])
```

## `GreenKuboSpectrum`

GK transform of a raw current ACF → $\varepsilon^*(\omega)$.

```python
import numpy as np
from molpy.compute import GreenKuboConductivity, GreenKuboSpectrum

J = np.ascontiguousarray(np.random.default_rng(1).normal(0, 1.0, size=(40, 3)))
jacf = GreenKuboConductivity().compute(J, 10.0, 15)
gk = GreenKuboSpectrum(dt=10.0, volume=1000.0, temperature=300.0, epsilon_inf=1.0)
spec = gk.fit(jacf["jacf"])
```

## `LinearFit`

Linear slope fit over a fractional lag window.

```python
from molpy.compute import EinsteinConductivity, LinearFit

m = np.ascontiguousarray(np.cumsum(np.random.default_rng(2).normal(0, 0.01, size=(40, 3)), axis=0))
raw_m = EinsteinConductivity().compute(m, 10.0, 15)
lin = LinearFit(0.1, 0.5).fit(raw_m["lag_times"], raw_m["msd"])
lin["slope"]
```

## `CumulativeTrapezoid`

Trapezoidal cumulative integral of an ACF.

```python
from molpy.compute import CumulativeTrapezoid

running = CumulativeTrapezoid().fit(jacf["jacf"], dt=10.0)
running["integral"]
```

## See also

- [PMSD](pmsd.md)
- [JACF](jacf.md)
- [Spectra](spectra.md)
