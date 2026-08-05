# RDF

Textbook guide to the **radial distribution function** $g(r)$ — the probability
of finding a particle at distance $r$ from a reference particle, normalized by
an ideal gas of the same density.

!!! note "Conventions"
    - Length Å; $g(r)$ dimensionless; keep $r_\max \le L/2$.
    - Workflow: build a [NeighborList](neighborlist.md), then histogram.

---

## 1. Definition

For $N$ particles in volume $V$ with density $\rho = N/V$,

$$
g(r) = \frac{V}{N^2}\Big\langle\sum_i\sum_{j\ne i}\delta(r-r_{ij})\Big\rangle
 \Big/ 4\pi r^2.
$$

Equivalently, if $n(r)\,\mathrm{d}r$ is the mean neighbour count in
$[r,r+\mathrm{d}r]$,

$$
g(r) = \frac{n(r)}{4\pi r^2\,\rho\,\mathrm{d}r}.
$$

Limits: $g\to 0$ as $r\to 0$ (excluded volume); $g\to 1$ as $r\to\infty$;
**peaks are coordination shells**. The first minimum after the first peak is the
natural cutoff for [Persist](persist.md), [Cluster](cluster.md), and
[Order](order.md).

### 1.1 Coordination number

$$
n(R) = 4\pi\rho \int_0^R r^2\, g(r)\,\mathrm{d}r.
$$

Evaluate at the first minimum of $g(r)$ for the first-shell coordination number.

---

## 2. Computing $g(r)$

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
from molpy.compute import NeighborList, RDF

nlist = NeighborList(cutoff=10.0)(frame)
result = RDF(n_bins=200, r_max=10.0)([frame], [nlist])
result.rdf, result.bin_centers
```

Average over a trajectory by passing parallel lists of frames and neighbor lists.
Neighbor `cutoff` must be $\ge$ `r_max`.

<figure id="fig-rdf" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:9"
mark:
  type: line
  strokeWidth: 2.2
  interpolate: monotone
data:
  values:
    - {r: 0.5, g: 0.0}
    - {r: 2.5, g: 0.05}
    - {r: 3.2, g: 2.8}
    - {r: 3.5, g: 1.6}
    - {r: 4.5, g: 1.1}
    - {r: 6.0, g: 0.95}
    - {r: 10.0, g: 1.0}
encoding:
  x:
    field: r
    type: quantitative
    title: r (Å)
  y:
    field: g
    type: quantitative
    title: g(r)
    scale: {zero: false}
  color:
    value: "#0284c7"
```

</div>

**Figure 1.** Schematic liquid $g(r)$: core exclusion, first shell peak, decay to 1.
</figure>

---

## 3. Pitfalls

1. `r_max > L/2` → MIC contamination.
2. Neighbor cutoff `< r_max` → truncated $g(r)$.
3. Too few bins → low first peak / coordination number.
4. Single frame → $g(r)$ is an ensemble average.

---

## 4. References

- M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed. (2017).
- J.-P. Hansen, I. R. McDonald, *Theory of Simple Liquids*, 4th ed. (2013).

## See also

- [NeighborList](neighborlist.md) · [Diffraction](diffraction.md) · [Density](density.md) · [PMFT](pmft.md)
- [API reference](../api/compute.md)
