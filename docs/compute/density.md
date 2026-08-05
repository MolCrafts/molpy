# Density

Textbook guide to **local and grid number densities** — where matter sits in
space, beyond the isotropic $g(r)$.

!!! note "Conventions"
    - Density in Å⁻³; length Å.
    - `LocalDensity` needs a [NeighborList](neighborlist.md); `GaussianDensity`
      smears onto a grid without one.

---

## 1. Two density pictures

- **`LocalDensity`** — per-particle scalar: number density inside a sphere of
  radius `r_max` about each particle. Interfaces, voids, packing variations.
- **`GaussianDensity`** — continuous field $\rho(\mathbf{r})$ on a fixed 3-D
  grid by Gaussian smearing of width `sigma`. Visualization and adsorption maps.

Too small a length scale → shot noise; too large → washed-out features. Match
`r_max` / `sigma` to a physical length (first $g(r)$ minimum, interface width).

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
from molpy.compute import NeighborList, LocalDensity, GaussianDensity

nlist = NeighborList(cutoff=4.0)(frame)
local = LocalDensity(r_max=4.0)([frame], [nlist])
grid = GaussianDensity(nx=64, ny=64, nz=64, sigma=1.0)([frame])
```

Optional `diameter` on `LocalDensity` applies a particle-size correction
(`0.0` counts centres only).

---

## 3. Pitfalls

1. `r_max` / `sigma` not matched to structure → noise or over-smoothing.
2. Free box → periodic `box` required.
3. Comparing densities across systems without reporting the smoothing length.

## See also

- [RDF](rdf.md) · [NeighborList](neighborlist.md) · [Voronoi](voronoi.md)
- [API reference](../api/compute.md)
