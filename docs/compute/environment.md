# Environment

Textbook guide to the **bond-orientational diagram** — the full angular map of
neighbour directions, complementary to the scalar invariants on [Order](order.md).

---

## 1. Why keep the full sphere

Order parameters compress a local shell to $q_\ell$ or $\psi_6$. **`BondOrder`**
histograms every bond direction onto a spherical $(\theta,\phi)$ grid. The
diagram shows the angular signature directly — four tetrahedral lobes, six
octahedral ones, hexagonal rings in 2-D.

Bonds come from a [NeighborList](neighborlist.md); the cutoff is again the
physics.

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
from molpy.compute import NeighborList, BondOrder

nlist = NeighborList(cutoff=5.0)(frame)
diagram = BondOrder(n_theta=80, n_phi=160)([frame], [nlist])
```

---

## 3. Pitfalls

1. Cutoff not at the first $g(r)$ minimum.
2. Mixing lab-frame orientations without a consistent molecular frame.
3. Under-sampling rare environments.

## See also

- [Order](order.md) · [NeighborList](neighborlist.md) · [Spatial](spatial.md)
- [API reference](../api/compute.md)
