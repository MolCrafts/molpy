# Environment

[Order](order.md) compresses an atom's neighbourhood into a single number,
$q_\ell$. That is what you want for classifying thousands of atoms, but
compression loses information: two quite different arrangements can share a
$q_6$.

`BondOrientationalOrder` keeps the whole thing. It histograms the **direction** of every bond
onto a sphere — polar angle $\theta$, azimuth $\phi$ — so you see the angular
signature of the local environment directly, rather than a scalar summary of it.

## What the diagram shows

Each neighbour of each atom contributes one point on the unit sphere: the
direction from the central atom to that neighbour. Accumulate over all atoms and
you get a map. Tetrahedral coordination shows four lobes; octahedral shows six;
a close-packed crystal shows the twelve directions of its first shell; a liquid
shows a smear.

The distinction is not subtle. Run it on a perfect FCC crystal of 256 atoms with
a first-shell cutoff:

<figure id="fig-bond-order" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
data: {$file: series/environment/fcc_bond_order.json}
mark: {type: circle, filled: true, opacity: 0.85}
encoding:
  x:
    field: phi
    type: quantitative
    title: "φ (deg)"
    scale: {domain: [0, 360]}
  y:
    field: theta
    type: quantitative
    title: "θ (deg)"
    scale: {domain: [0, 180]}
  size:
    field: n
    type: quantitative
    title: "bonds"
```

</div>

**Figure 1.** Bond-direction map of a perfect FCC lattice. All 3072 bonds
(256 atoms × 12 neighbours) fall into **28** of the 2592 $(\theta,\phi)$ cells.
The occupied cells are the crystallographic directions of the first shell.
</figure>

That is the whole point of the diagram. A crystal has a discrete set of bond
directions, and the map is almost entirely empty. The same calculation on liquid
argon at its first-shell cutoff fills **2442** of the 2592 cells: every direction
occurs, and the map carries no angular information beyond the trivial
$\sin\theta$ weighting of the spherical grid.

Two consequences worth stating plainly. First, this compute is most useful on
ordered or partially ordered systems — crystals, adsorbed layers, coordination
complexes — and least useful on isotropic liquids, where it tells you nothing
$g(r)$ did not. Second, the map is in the **lab frame**. For molecules that
tumble, bond directions are smeared by the molecule's own rotation, and you need
a body-fixed frame to see anything; that is [Spatial](spatial.md) and
[PMFT](pmft.md).

## Computing it

```python
import numpy as np
import molpy as mp
from molpy.compute import BondOrientationalOrder

a = 5.26
basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
xyz = np.array(
    [(np.array([i, j, k]) + b) * a
     for i in range(4) for j in range(4) for k in range(4) for b in basis]
)
crystal = mp.Frame()
crystal["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
crystal.box = mp.Box(mp.core.Cell.lengths([4 * a] * 3))
```

The diagram is built from one thing, the neighbour table. Positions and box
are read from the frame and indexed by a `NeighborList`; the grid resolution is
configuration and goes into the constructor:

```python
nl = mp.core.NeighborList(4.5)
nl.build(crystal.coords, crystal.box)
nlist = nl.neighbors()
diagram = BondOrientationalOrder(n_theta=36, n_phi=72).compute(neighbors=nlist)

counts = diagram.raw_counts
print(counts.shape, diagram.theta_edges.shape)       # -> (36, 72) (37,)
print(int(counts.sum()), int((counts > 0).sum()))    # -> 3072 28
```

`raw_counts` is the integer histogram and `bond_order` the same grid
normalized as a density on the sphere; the two edge arrays are one longer
than their axis, as histogram edges always are.

3072 is 256 × 12: each bond is counted once **from each end**, so the total is
twice the pair count from the [neighbour list](neighborlist.md). Do not confuse
it with `n_pairs`.

Convert edges to centres the usual way when plotting:

```python
theta = 0.5 * (diagram.theta_edges[:-1] + diagram.theta_edges[1:])
print(round(float(np.degrees(theta[0])), 1))         # -> 2.5
```

Angles are in radians, like the rest of the angular machinery.

## Environments can be matched atom by atom

The diagram pools every atom into one map. Sometimes the question is the
opposite one: which atoms share the *same* environment? `EnvironmentMatch`
compares each atom's set of neighbour vectors with every other atom's, and
groups atoms whose sets agree to within an RMSD threshold. In a perfect
crystal every site is equivalent, so there is exactly one group:

```python
from molpy.compute import EnvironmentMatch

matched = EnvironmentMatch(rmsd_threshold=0.1).compute(neighbors=nlist)
print(matched.n_clusters, len(matched.cluster_idx))  # -> 1 256
```

The result labels every atom with its environment class, and it is a
partition: it can be passed wherever a compute takes a `partition`, for
instance to `CenterOfMass` to locate each class (see [Shape](shape.md)).
`LocalDescriptors(l_max)` is the continuous counterpart — the spherical
harmonics of every bond up to `l_max` as a feature matrix, ready for
[Decomposition](decomposition.md).

## When it goes wrong

**The map is uniformly filled and you expected lobes.**
Three candidates, in order of likelihood: the system really is isotropic; the
cutoff reaches past the first shell and mixes several coordination environments;
or the molecules tumble and you need a body-fixed frame.

**Bands of intensity near $\theta = 0$ and $\theta = \pi$.**
Cells near the poles subtend less solid angle, so a *uniform* distribution of
directions does **not** give uniform counts per cell — it gives counts
proportional to $\sin\theta$. Divide it out before reading the map, exactly as on
the [Distribution](distribution.md) page.

**Everything lands in a few cells and the system is not a crystal.**
Check that the cutoff is not so small that only one or two neighbours are found
per atom.

**The diagram changes when you rotate the box.**
Expected, and it is the limitation of the lab frame. A crystal rotated is the
same crystal but a different map. Use $q_\ell$ from [Order](order.md) when you
need rotational invariance.

## Check yourself

- Run it on a simple cubic lattice: you should get six occupied cells, on the
  $\pm x$, $\pm y$, $\pm z$ axes.
- Run it on random points and confirm the map is featureless once the
  $\sin\theta$ weight is accounted for.
- Compare `counts.sum()` with `2 * nlist.n_pairs`. They must agree.

## References

- P. J. Steinhardt, D. R. Nelson, M. Ronchetti, *Phys. Rev. B* **28**, 784
  (1983) — bond-orientational order, of which this is the uncompressed form.
- V. Ramasubramani et al., *Comput. Phys. Commun.* **254**, 107275 (2020) — the
  freud `environment` module.

## See also

- [Order](order.md) — the rotationally invariant summary of this map
- [PMFT](pmft.md) · [Spatial](spatial.md) — body-fixed alternatives
- [NeighborList](neighborlist.md) · [API reference](../api/compute.md)
