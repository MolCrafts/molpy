# PMFT

Textbook guide to the **potential of mean force and torque** — free energy from
structure, including orientation-resolved 2-D maps.

!!! note "Conventions"
    - Free energy in $k_B T$ units when $w = -\ln g$.
    - Optional `orientations` topology block rotates bonds into a local frame.

---

## 1. From $g(r)$ to free energy

A pair distribution is a Boltzmann factor in disguise:

$$
w(r) = -k_B T \ln g(r).
$$

Minima are coordination shells; barriers are desolvation penalties.
**`PMFTXY`** generalizes this to a 2-D map of neighbour positions in the local
$(x,y)$ frame of each reference particle — face-to-face vs edge-to-edge
contacts that isotropic $g(r)$ averages away.

With an `orientations` block `(head, tail)` per particle, every bond is rotated
into that particle's body frame (`atan2` of `head - tail`). Without it, the
analyzer works in the lab frame.

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
from molpy.compute import NeighborList, PMFTXY

nlist = NeighborList(cutoff=6.0)(frame)
result = PMFTXY(x_max=6.0, y_max=6.0, n_x=120, n_y=120)([frame], [nlist])
```

---

## 3. Pitfalls

1. Neighbor cutoff too short for the free-energy region of interest.
2. Misaligned orientation endpoints → garbage body frame.
3. Sparse sampling of 2-D histograms — average many frames.

## See also

- [RDF](rdf.md) · [NeighborList](neighborlist.md) · [Spatial](spatial.md)
- [API reference](../api/compute.md)
