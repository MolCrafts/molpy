# Spatial

Textbook guide to the **spatial distribution function** (SDF) — the full 3-D
generalization of $g(r)$ in a body-fixed molecular frame.

---

## 1. Body-fixed density clouds

1. Pick reference atoms on the central molecule and a **template geometry**.
2. Kabsch-align those atoms each frame → body-fixed axes.
3. Accumulate target-atom density $\rho(\mathbf{x}_\mathrm{body})$ on a grid.

Normalized by bulk density,

$$
g_\mathrm{SDF}(\mathbf{x}) = \rho(\mathbf{x}_\mathrm{body})/\rho_\mathrm{bulk},
$$

so isotropic RDF shells become **lobes** (lone pairs, $\pi$-stacking, ion
approaches). An optional `orientations` block adds per-voxel mean orientation of
a head–tail vector on the target species.

---

## 2. Usage

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
from molpy.compute import SpatialDistribution

sdf = SpatialDistribution(
    reference=[0, 1, 2],
    template=np.array([[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0]]),
    target=[2],
    n=(32, 32, 32),
    extent=(8.0, 8.0, 8.0),
    bulk_density=0.033,
)
result = sdf([frame])
result.density, result.g_sdf
```

---

## 3. Pitfalls

1. Template atom order mismatch → garbage body frame.
2. Sparse 3-D histograms need many frames.
3. Extent too small clips the first shell.

## See also

- [Distribution](distribution.md) · [RDF](rdf.md) · [PMFT](pmft.md)
- [API reference](../api/compute.md)
