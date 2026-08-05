# Diffraction

Textbook guide to the **static structure factor** $S(k)$ — $g(r)$ seen in
reciprocal space, the quantity scattering experiments measure.

!!! note "Conventions"
    - Wavenumber $k$ in Å⁻¹; $S(k)$ dimensionless.
    - Avoid $k=0$; smallest meaningful $k \approx 2\pi/L$.

---

## 1. Debye equation and $g(r)\leftrightarrow S(k)$

MolPy evaluates the **Debye scattering equation** directly from coordinates:

$$
S(k) = \frac{1}{N}\Big\langle\sum_i\sum_j
\frac{\sin(k\,r_{ij})}{k\,r_{ij}\Big\rangle.
$$

For an isotropic fluid,

$$
S(k) = 1 + 4\pi\rho\int_0^\infty r^2\,[g(r)-1]\,
\frac{\sin(kr)}{kr}\,\mathrm{d}r,
$$

so $S(k)$ and $g(r)$ carry the same pair information. Use $S(k)$ to compare with
X-ray/neutron diffraction, locate the first sharp diffraction peak, or read
long-wavelength compressibility from $S(k\to 0)$.

Cost is $\mathcal{O}(N^2)$ per $k$ per frame — fine for moderate $N$; for huge
boxes prefer a grid FFT route (not covered here).

---

## 2. Computing $S(k)$

```python
import numpy as np
import molpy as mp

rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 20.0, size=(200, 3))
frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box.cubic(20.0)
```

```python
import numpy as np
from molpy.compute import StaticStructureFactorDebye

k = np.linspace(0.2, 12.0, 300)  # Å^-1
sk = StaticStructureFactorDebye(k)([frame])
```

---

## 3. Pitfalls

1. Including $k=0$ → division by zero.
2. Over-dense $k$-grid on large $N$ → wasteful $\mathcal{O}(N^2)$ sums.
3. Interpreting $k < 2\pi/L$ as bulk thermodynamics.

## See also

- [RDF](rdf.md) · [NeighborList](neighborlist.md)
- [API reference](../api/compute.md)
