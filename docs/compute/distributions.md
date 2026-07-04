# Distribution Functions: Angle, Dihedral, Combined & Spatial

This page is a self-contained, textbook-style introduction to the **geometric
distribution functions** MolPy ports from the reference implementation: the angular (ADF), dihedral
(DDF), and distance distribution functions, their joint **combined distribution
function** (CDF), and the orientation-resolved **spatial distribution function**
(SDF). Where the [radial distribution function](structure.md) answers "how far?",
these answer "at what angle, in what combination, and in which direction?" — the
tools for reading the *geometry* of local order, not just its radial extent.

As with every `compute` operator, the histogram kernels run in Rust (`molrs`); the
MolPy layer extracts coordinates and the box and returns a typed result. The
geometric distributions take **one input** — the frames. The atom tuples to
histogram are read from each frame's core topology blocks: `bonds` (pairs) for the
distance distribution, `angles` (triplets) for the ADF, `dihedrals` (quadruplets)
for the DDF. No separate index array is passed.

!!! note "Conventions used throughout"
    - Distances are in Å, angles and dihedrals in degrees, densities in Å⁻³.
    - The atom tuples come from the frame's topology blocks — pairs `(i, j)` from
      `bonds`, triplets `(i, j, k)` from `angles` (vertex at `j`), quadruplets
      from `dihedrals`. Perceive them with
      `Atomistic.get_topo(gen_angle=True, gen_dihe=True)`.
    - Angular distributions carry a trivial `sin θ` solid-angle weighting; the
      result's `density_sin_corrected` removes it so a structureless distribution
      is flat.

---

## 1. Distance, angle and dihedral distributions histogram an internal coordinate

Each of these functions is a normalized histogram of one internal coordinate over
a chosen set of atom tuples. For a set of triplets, the **angular distribution
function** is

$$
p(\theta) = \frac{1}{N_\text{groups}}\Big\langle\sum_{(i,j,k)}\delta\big(\theta - \theta_{ijk}\big)\Big\rangle,
$$

with $\theta_{ijk}$ the angle at the vertex $j$. The **dihedral distribution**
replaces $\theta$ with the torsion of a quadruplet, and the **distance
distribution** with the separation of a pair (an RDF restricted to specific pairs,
without the $4\pi r^2$ shell normalization). Together they characterize bond-angle
stiffness, conformer populations (gauche vs. anti from the DDF), and selective pair
structure.

---

## 2. Computing the geometric distributions

Perceive the topology (so the frame carries `bonds` / `angles` / `dihedrals`),
then call the operator on one or more frames:

```python
from molpy.compute import AngleDistribution, DihedralDistribution, DistanceDistribution

# The frame must carry the relevant topology block, e.g. from a built structure:
frame = mol.get_topo(gen_angle=True, gen_dihe=True).to_frame()

adf = AngleDistribution(n_bins=180, min=0.0, max=180.0)
result = adf([frame])        # angle triplets read from frame["angles"]

result.bin_centers           # angle at each bin, degrees
result.density               # normalized p(theta)
result.density_sin_corrected # solid-angle-corrected distribution
```

`DistanceDistribution(n_bins, min, max)` and
`DihedralDistribution(n_bins, min=-180, max=180)` follow the same call pattern,
reading the frame's `bonds` and `dihedrals` blocks respectively.

---

## 3. Combined distribution functions expose correlations between coordinates

A 1-D distribution averages away correlations — but the *joint* distribution of,
say, an O···H distance and the O···H–C angle reveals whether short contacts are
also linear, the signature of a hydrogen bond. The **combined distribution
function** (CDF) histograms several coordinates simultaneously onto a multi-axis
grid:

$$
p(x_1, x_2, \dots) = \frac{1}{N}\Big\langle\sum_\text{groups}\prod_a \delta\big(x_a - x_a^\text{group}\big)\Big\rangle.
$$

Each axis is declared as `(kind, n_bins, min, max, sin_weight)`:

```python
from molpy.compute import CombinedDistribution

cdf = CombinedDistribution([
    ("distance", 100, 2.0, 4.0, False),   # read from frame["bonds"]
    ("angle",     90, 90.0, 180.0, True), # read from frame["angles"] (sin-weighted)
])
result = cdf([frame])   # each axis reads the tuples of its kind's topology block
```

The result carries the multi-dimensional histogram plus helpers
(`bin_width_product`, `flat_index`) for integrating or slicing the joint density.

---

## 4. The spatial distribution function maps three-dimensional structure

The **spatial distribution function** (SDF) is the full 3-D generalization: it
fixes a body-fixed frame on a reference molecule (by Kabsch-aligning its reference
atoms to a template geometry) and accumulates the density of target atoms on a
grid in that frame. The result is the three-dimensional cloud of *where*
neighbours sit around a molecule — the lone-pair lobes of water, the stacking
geometry of aromatics — that an isotropic $g(r)$ averages into a single shell.

```python
import numpy as np
from molpy.compute import SpatialDistribution

sdf = SpatialDistribution(
    reference=[o, h1, h2],            # atoms defining the body frame
    template=np.array([[0,0,0],[0.76,0.59,0],[-0.76,0.59,0]]),  # ideal geometry
    target=[o],                       # density of neighbouring O atoms
    n=(64, 64, 64),                   # grid resolution
    extent=(8.0, 8.0, 8.0),           # half-extent per axis, Å
    bulk_density=0.033,               # optional -> result.g_sdf
)
result = sdf(frames)
result.density   # target density on the body-fixed grid
result.g_sdf     # normalized by bulk_density (if supplied)
```

If the frames carry an `orientations` topology block (one `(head, tail)` atom
pair per target atom, in `target` order), the result also exposes a per-voxel
mean body-frame orientation of the unit `head - tail` vector as
`result.orientation`; without the block the SDF is orientation-free.

---

## 5. Pitfalls checklist

1. **Wrong vertex order** → for the ADF the angle is at the *middle* index of each
   triplet; the `angles` topology block already stores them vertex-in-the-middle
   (perceived by `get_topo(gen_angle=True)`).
2. **Forgetting the sin-correction** → raw angular densities peak near 90° purely
   from the solid angle; compare `density_sin_corrected` for structure.
3. **CDF axis-count mismatch** → every axis reads its `kind`'s topology block
   (`bonds` / `angles` / `dihedrals`), and all axes must yield the same number of
   tuples, or the joint sample is undefined.
4. **SDF template misaligned** → the template must match the `reference` atom
   ordering and a sensible geometry, or the body frame (and the whole map) is
   garbage; verify with a small, symmetric reference.
5. **Sparse sampling** → 2-D/3-D histograms need far more samples than 1-D ones to
   fill their bins; average many frames before reading peak heights.

---

## 6. References

- M. Brehm, B. Kirchner, *J. Chem. Inf. Model.* **51**, 2007 (2011) — reference implementation;
  radial/angular/dihedral and combined distribution functions.
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105
  (2020) — reference implementation, current feature set.
- I. M. Svishchev, P. G. Kusalik, *J. Chem. Phys.* **99**, 3049 (1993);
  P. G. Kusalik, I. M. Svishchev, *Science* **265**, 1219 (1994) — spatial
  distribution functions.

## See also

- [Structural Analysis](structure.md) — the radial distribution function and structure factor.
- [Hydrogen-Bond Networks](hbonds.md) — the distance–angle CDF is the natural H-bond map.
- [Compute overview](index.md) — the Compute → Result pattern.
- [API reference: Compute](../api/compute.md).
