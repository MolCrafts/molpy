# PMSD

Textbook guide to the **Einstein (charge-dipole MSD) route to ionic
conductivity** — collective polarization displacement of charge carriers.

!!! note "Conventions"
    - Build $\mathbf{M}(t)=\sum_a q_a\mathbf{r}_a$ from **unwrapped** positions.
    - Time fs; compose raw curve → `LinearFit` → SI scale in your script.

---

## 1. Einstein conductivity

$$
\mathrm{MSD}_M(\tau)
= \big\langle|\mathbf{M}(t+\tau)-\mathbf{M}(t)|^2\big\rangle_t,
\qquad
\sigma = \lim_{\tau\to\infty}
\frac{1}{6\,V k_B T}\frac{\mathrm{d}{\mathrm{d}\tau}\mathrm{MSD}_M(\tau).
$$

There is no trajectory-scanning recipe class: assemble $\mathbf{M}(t)$ yourself,
then `EinsteinConductivity` → `LinearFit` → SI prefactor.

Equivalent Green–Kubo current route: [JACF](jacf.md). Frequency-dependent
response: [Dielectric](dielectric.md).

---

## 2. Usage

```python
import numpy as np
from molpy.compute import EinsteinConductivity, LinearFit

rng = np.random.default_rng(1)
M = np.ascontiguousarray(np.cumsum(rng.normal(0, 0.02, size=(60, 3)), axis=0))
raw = EinsteinConductivity().compute(M, dt=10.0, max_correlation_time=20)
fit = LinearFit(start_frac=0.1, end_frac=0.5).fit(raw["lag_times"], raw["msd"])
slope = fit["slope"]  # then sigma = slope / (6 V k_B T) * SI_prefactor
```

---

## 3. Pitfalls

1. Wrapped ion coordinates → nonsense $\mathbf{M}$.
2. Quoting $\sigma$ at the last lag instead of the linear window.
3. Mixing solvent total dipole into $\mathbf{M}$ for electrolytes — see
   [Dielectric](dielectric.md) decomposition.

## See also

- [JACF](jacf.md) · [Onsager](onsager.md) · [Dielectric](dielectric.md) · [MSD](msd.md)
- [API reference](../api/compute.md)
