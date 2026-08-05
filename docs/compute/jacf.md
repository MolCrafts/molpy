# JACF

Textbook guide to the **current autocorrelation** (Green–Kubo) route to ionic
conductivity.

!!! note "Conventions"
    - $\mathbf{J}(t)=\sum_a q_a\mathbf{v}_a$; velocities in Å/fs for real units.
    - Compose: raw JACF → `CumulativeTrapezoid` → SI scale; quote the **plateau**.

---

## 1. Green–Kubo conductivity

$$
\boxed{
\sigma = \frac{1}{3\,V k_B T}
\int_0^\infty \big\langle\mathbf{J}(0)\cdot\mathbf{J}(t)\big\rangle\,\mathrm{d}t
}
$$

Mathematically equivalent to the Einstein [PMSD](pmsd.md) route; JACF needs
finer velocity sampling but exposes memory and integral convergence.
Frequency-dependent $\sigma(\omega)$: [Dielectric](dielectric.md).

---

## 2. Usage

```python
import numpy as np
from molpy.compute import GreenKuboConductivity, CumulativeTrapezoid

rng = np.random.default_rng(2)
J = np.ascontiguousarray(rng.normal(0, 1.0, size=(60, 3)))
raw = GreenKuboConductivity().compute(J, dt=10.0, max_correlation_time=20)
running = CumulativeTrapezoid().fit(raw["jacf"], dt=10.0)["integral"]
# sigma from plateau of running / (3 V k_B T) * SI_prefactor
```

---

## 3. Pitfalls

1. Coarse velocity dumps → missed JACF decay.
2. Quoting the integral at the last lag without a plateau.
3. COM / total-current drift spikes.

## See also

- [PMSD](pmsd.md) · [VACF](vacf.md) · [Dielectric](dielectric.md)
- [API reference](../api/compute.md)
