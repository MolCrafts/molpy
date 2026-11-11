# Free volume and pore size

A radial distribution says how far atoms sit from each other. It does not say
how much of the cell a guest molecule can occupy, or how wide the cavities
are. Those are two different questions, and they use two computes.

Both take the van der Waals radius of each atom from the caller. Neither
reads a frame, and neither looks up an element table: the radius you pass is
the radius that is used.

## Where a probe fits

`FreeVolume` draws uniform points in the cell and asks, for each one, whether
a hard sphere of radius `probe_radius` (Å) can sit there. A point is
accessible when its clearance to every van der Waals sphere is strictly
larger than the probe. A point on an atom's surface is occupied.

The accessible fraction is that hit rate. Its uncertainty is the binomial
standard error $\sqrt{p(1-p)/n}$.

A point probe, `probe_radius = 0`, measures the complement of the union of
the van der Waals spheres. Bondi's fractional free volume (Bondi, *J. Phys.
Chem.* **1954**, 68, 441) uses an occupied volume $1.3$ times that union:

$$
\mathrm{FFV} = 1 - 1.3\,\frac{V_w}{V}.
$$

`bondi_fraction` is that number, and only for a point probe. A positive probe
asks the other question — where the centre of a sphere of that radius can
sit — and Bondi's factor does not apply, so `bondi_fraction` is absent. The
formula can be negative: a cell whose van der Waals volume exceeds $V/1.3$
has a negative Bondi fraction, which is the definition rather than a missing
cavity.

## How wide the cavities are

`PoreSizeDistribution` assigns every voxel the radius of the largest sphere
that covers the voxel centre and does not intersect an atom. The sphere may
be centred anywhere. That radius is the local thickness of Gelb and Gubbins
(*Langmuir* **2006**, 22, 7726), which is the distribution Poreblazer and
Zeo++ report.

The histogram is in **radius**, Å. A plot in diameter, the way Poreblazer
draws it, is twice these values. Bin $i$ of width $w$ holds radii in
$(i w,\ (i+1) w]$. `fraction` is the share of the cell volume in the bin.
`cumulative` is the share whose radius exceeds the bin's lower edge, which is
the volume a probe of that radius can cover. Voxels inside an atom are left
out of the histogram and counted in the complement of `void_fraction`.

The grid is the box's own lattice, so the cell has to be orthogonal. A
triclinic cell is an error.

## Computing it

A 10 Å cube with one sphere at its centre. The filled-cell check is exact:
every sample lands inside a sphere of radius 20 Å. The point-probe fraction
is a Monte Carlo estimate, so the example checks Bondi's identity on those
same samples and does not pin the estimate to the sphere's volume.

```python
import numpy as np
import molpy as mp
from molpy.compute import FreeVolume, PoreSizeDistribution

box = mp.Box(mp.core.Cell.lengths([10.0] * 3))
centre = np.array([[5.0, 5.0, 5.0]])

filled = FreeVolume(0.0, n_samples=400).compute(
    positions=centre, radii=np.array([20.0]), simbox=box
)
assert filled.accessible_fraction == 0.0
assert filled.bondi_fraction == 1.0 - 1.3

point = FreeVolume(0.0, n_samples=2000, seed=1).compute(
    positions=centre, radii=np.array([2.0]), simbox=box
)
bondi = 1.0 - 1.3 * (1.0 - point.accessible_fraction)
assert abs(point.bondi_fraction - bondi) < 1e-12
print(round(point.accessible_fraction, 3), round(point.bondi_fraction, 3))

sphere = FreeVolume(1.2, n_samples=400, seed=1).compute(
    positions=centre, radii=np.array([2.0]), simbox=box
)
assert sphere.bondi_fraction is None

pores = PoreSizeDistribution(1.0, bin_width=0.5).compute(
    positions=np.array([[0.0, 0.0, 0.0]]),
    radii=np.array([0.0]),
    simbox=box,
)
expected = 4.5 * np.sqrt(3.0)
assert abs(pores.largest_cavity_radius - expected) < 1e-6
assert abs(float(pores.fraction.sum()) - 1.0) < 1e-12
print(round(pores.largest_cavity_radius, 3), pores.radius.shape)
```

The corner atom of radius 0 leaves the largest cavity at the cell centre.
With 1 Å voxels the centre sits at a half-integer, $4.5\sqrt{3}$ Å from
every periodic image of the origin.

A water network is not a third kernel. The oxygen subset goes to `Cluster`,
and `Percolation` says whether a cluster spans the cell. A diffusivity is
`Msd` on the unwrapped water coordinates.
