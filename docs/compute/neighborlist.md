# NeighborList

Textbook guide to the **neighbor list** — the shared spatial primitive behind
RDF, density, order parameters, clustering, and PMFT. A `NeighborList` returns
every pair of particles within a cutoff under the minimum-image convention.

Heavy pair finding runs in the high-performance backend; MolPy extracts
coordinates and the box.

!!! note "Conventions"
    - Length in Å. Cutoff must be $\le L/2$ for a clean minimum-image shell.
    - Build once per frame; reuse for every cutoff-based analysis on that frame.

---

## 1. Why a neighbor list is the shared primitive

Most structural analyses are **local**: they need every pair closer than some
$r_c$, not the full $N^2$ distance matrix. A neighbor list answers that query
once; `RDF`, `LocalDensity`, `Steinhardt`, `Cluster`, and `PMFTXY` all consume
the same object.

Under periodic boundaries the **minimum-image convention** maps every partner
into the image closest to the reference particle. The list is therefore only
well-defined for $r_c \le L/2$ (or half the shortest box edge).

---

## 2. Computing neighbors

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
from molpy.compute import NeighborList

nlist = NeighborList(cutoff=5.0)(frame)
nlist.n_pairs      # number of pairs found
nlist.pairs        # (n_pairs, 2) index array
nlist.distances    # pair distances, Å
```

### 2.1 Choosing the cutoff

| Consumer | Rule of thumb |
|---|---|
| `RDF(r_max=…)` | `cutoff ≥ r_max` |
| Order / cluster | first minimum of $g(r)$ |
| PMFT | covers the free-energy region of interest |

Cost scales with the number of pairs $\sim \rho\, r_c^3$. Do not build a 12 Å
list to histogram 6 Å of $g(r)$.

---

## 3. Pitfalls

1. **Cutoff smaller than consumer `r_max`** → truncated histograms / missing bonds.
2. **`r_c > L/2`** → minimum-image shells wrap and double-count images.
3. **Free box** → requires a periodic `frame.box`.
4. **Stale list** → rebuild after coordinates change.

---

## 4. References

- V. Ramasubramani et al., *Comput. Phys. Commun.* **254**, 107275 (2020) —
  freud locality / neighbor queries.
- M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed. (2017).

## See also

- [RDF](rdf.md) · [Density](density.md) · [PMFT](pmft.md) · [Order](order.md)
- [Compute overview](index.md) · [API reference](../api/compute.md)
