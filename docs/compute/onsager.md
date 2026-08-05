# Onsager

Textbook guide to **Onsager phenomenological coefficients** — collective
displacement cross-correlations that encode multi-component transport coupling.

---

## 1. Collective coordinates and $\Omega_{\alpha\beta}$

$$
\mathbf{P}_\alpha(t) = \sum_{i\in\alpha}\mathbf{r}_i(t),
\qquad
L_{\alpha\beta}(\tau)
= \big\langle\Delta\mathbf{P}_\alpha(\tau)\cdot\Delta\mathbf{P}_\beta(\tau)\big\rangle_t,
$$

$$
\boxed{
\Omega_{\alpha\beta}
= \lim_{\tau\to\infty}
\frac{L_{\alpha\beta}(\tau)}{6\,k_B T\,V\,N_A\,\tau}
}
$$

- Diagonal: collective MSD of species $\alpha$.
- Off-diagonal: cation–anion coupling (often negative under ion pairing).

Conductivity from the Onsager matrix:

$$
\sigma = \frac{e^2}{V k_B T}\sum_{\alpha\beta} z_\alpha z_\beta\,\Omega_{\alpha\beta}.
$$

If off-diagonals vanish this is Nernst–Einstein from self-diffusion alone.
The ratio $\sigma/\sigma_\mathrm{NE}$ measures correlation suppression.

`Onsager.correlation` returns the raw $L_{\alpha\beta}(\tau)$ only — fit the
slope with `LinearFit` and apply the prefactor yourself.

---

## 2. Usage

```python
import numpy as np
from molpy.compute import Onsager, LinearFit

rng = np.random.default_rng(0)
P_cat = np.ascontiguousarray(np.cumsum(rng.normal(0, 0.01, size=(40, 3)), axis=0))
P_an = np.ascontiguousarray(np.cumsum(rng.normal(0, 0.01, size=(40, 3)), axis=0))
L11 = Onsager.correlation(P_cat, P_cat, dt=10.0, max_correlation_time=10)
L12 = Onsager.correlation(P_cat, P_an, dt=10.0, max_correlation_time=10)
fit = LinearFit(0.1, 0.5).fit(L11["lag_times"], L11["correlation"])
```

---

## 3. Pitfalls

1. Wrapped species COM sums.
2. Fitting before the linear regime.
3. Comparing $\Omega$ across definitions of $N_A$ / volume units.

## See also

- [MSD](msd.md) · [PMSD](pmsd.md) · [Persist](persist.md)
- [API reference](../api/compute.md)
