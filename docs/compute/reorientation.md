# Reorientation

Textbook guide to **Legendre reorientational correlation functions**
$C_1(t)$, $C_2(t)$ — how fast molecular vectors forget their direction.

---

## 1. Legendre correlations

For a unit vector $\mathbf{u}(t)$ (bond, dipole, axis),

$$
\boxed{
C_\ell(t)
= \big\langle P_\ell\big(\mathbf{u}(0)\cdot\mathbf{u}(t)\big)\big\rangle,
\quad
P_1(x)=x,\;
P_2(x)=\tfrac12(3x^2-1)
}
$$

| Probe | Order |
|---|---|
| Dielectric / IR | $C_1$ |
| NMR, fluorescence, Raman | $C_2$ |

In Debye rotational diffusion $\tau_\ell = 1/(\ell(\ell+1)D_R)$ so
$\tau_1/\tau_2=3$. Large deviations signal jump reorientation.

Fit the **long-time exponential tail**, not the librational head.

---

## 2. Usage

```python
import numpy as np
import molpy as mp
from molpy.compute import LegendreReorientation

rng = np.random.default_rng(0)
frames = []
for step in range(20):
    xyz = rng.uniform(0.0, 20.0, size=(30, 3)) + 0.1 * step
    f = mp.Frame()
    f["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    f.box = mp.Box.cubic(20.0)
    f["bonds"] = {"atomi": np.array([0, 0]), "atomj": np.array([1, 2])}
    frames.append(f)

result = LegendreReorientation(max_lag=5)(frames)
result.lags, result.c1, result.c2
```

---

## 3. Pitfalls

1. Fitting the sub-ps librational decay as $\tau_\ell$.
2. Degenerate head/tail atoms.
3. Comparing $\tau_1/\tau_2$ across different vector definitions.

## See also

- [Van Hove](van_hove.md) · [Dielectric](dielectric.md) · [Spectra](spectra.md)
- [API reference](../api/compute.md)
