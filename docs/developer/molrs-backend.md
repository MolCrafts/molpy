# molrs Backend

MolPy's analysis operators are the analyses of [molrs](https://github.com/MolCrafts/molrs),
a Rust column store and compute kernel, re-exported by identity. molrs is a
**required** runtime dependency — callers use `mp.Frame` and `mp.Block`, which
are the molrs types re-exported unchanged (never `import molrs` in user code);
both are backed by a Rust `Store`, and every class on `mp.compute` is the molrs
class itself. There is no pure-Python fallback and no opt-in flag.

This page shows how that backend surfaces in everyday analysis: how the box
type is shared with molrs, how to build neighbor lists and radial distribution
functions, and what the rest of the molrs analysis catalog looks like from
Python.

## Installation pulls molrs in automatically

molrs ships as the PyPI package `molcrafts-molrs`. Because it is a hard
dependency, a normal install already provides it:

```bash
pip install molcrafts-molpy
```

There is no `molpy[molrs]` extra. molrs is a hard runtime dependency.

### Version policy: same major.minor

`pyproject.toml` pins the molrs **minor line**
(`molcrafts-molrs>=X.Y.0,<X.(Y+1)`). On `import molpy`,
`molpy.version.check_molrs_version()` fails only when major.minor differs —
patch-level drift is allowed (e.g. molpy `0.10.0` with molrs `0.10.1`).
There is no exact-patch requirement and no hand-written CHANGELOG.

## The box is a molrs object, not a copy of one

`molpy.Box` does not wrap a molrs box; it **inherits** from it:

The examples below share this setup:

```python
import numpy as np
import molpy as mp


def _frame(step: int) -> mp.Frame:
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0.0, 20.0, size=(200, 3)) + 0.1 * step
    frame = mp.Frame()
    frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    frame.box = mp.Box.cube(20.0)
    return frame


frames = [_frame(step) for step in range(20)]
```

```python
import molrs
from molpy.core.box import Box


class Box(molrs.Box): ...
```

The practical consequence is that a molpy box can be handed to any molrs API
unchanged — there is no `.to_molrs()` bridge and no coordinate translation:

```python
import molrs
import molpy as mp

box = mp.Box.cube(10.0)
assert isinstance(box, molrs.Box)  # it *is* a molrs box
```

Likewise `frame.box` is accepted directly by Rust-side calls such as
`mp.NeighborList.build`. molpy adds only a constructor that also accepts no
matrix (a free box) or a `(3,)` diagonal, and the `Box.Style` enumeration; the
geometry (`wrap`, `unwrap`, `delta`, `distances`, `to_frac`, …) and the
factories (`Box.cube`, `Box.ortho`, `Box.from_bounds`) are the native box's.

## Neighbor lists come from the linked-cell kernel

`mp.NeighborList` (the molrs class) searches for all pairs within a cutoff
using a linked-cell algorithm (O(N) in the number of atoms). `build` indexes
the coordinates and `neighbors()` returns the pair table, a `mp.Neighbors`:

```python
import numpy as np
import molpy as mp

rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 20.0, size=(500, 3))

frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box.cube(20.0)

nl = mp.NeighborList(8.0)
nl.build(frame.coords, frame.box)
neighbors = nl.neighbors()
print(neighbors.n_pairs)  # number of pairs found
print(np.sqrt(neighbors.dist_sq())[:5])  # pair distances (Å) from stored dist_sq
```

`build` needs a box: a frame without one (`frame.box is None`) raises
`TypeError`. A free box (no periodic axis) searches without minimum images.

## The RDF reuses the neighbor list it is given

`RDF` computes the radial distribution function

$$ g(r) = \frac{V}{N\,N_q}\,\frac{\langle n(r)\rangle}{4\pi r^2\,\Delta r} $$

from one or more frames plus the neighbor list for each. Passing the neighbor
list in explicitly keeps the expensive pair search out of the histogram loop
and lets you reuse a single search for several analyses:

```python
from molpy.compute import RDF

result = RDF(n_bins=50, r_max=8.0).compute(frame, neighbors)
print(result.bin_centers)  # r at each bin centre
print(result.rdf)  # g(r)
```

For an ideal gas (uniformly random points) the middle bins of `result.rdf`
sit near 1.0, which is the standard sanity check for a correct normalization.
Multiple frames are averaged when you pass lists:
`RDF(...).compute(frames, neighbor_lists)`.

## The wider analysis catalog is exposed as molpy operators

A range of standard trajectory analyses live in molrs. `mp.compute`
re-exports each class by identity, so the table is a catalogue of molrs
analyses, each with a `compute(...)` method:

| Operator | What it computes |
|----------|------------------|
| `MSD` | mean-squared displacement vs. lag time |
| `Cluster`, `ClusterCenters`, `ClusterProperties` | connected-component clustering, centroids, and per-cluster size/mass/gyration |
| `CenterOfMass` | mass-weighted centroid |
| `GyrationTensor`, `RadiusOfGyration`, `InertiaTensor` | shape descriptors |
| `Pca2`, `KMeans` | two-component PCA and k-means partitioning |
| `Steinhardt`, `Hexatic`, `SolidLiquid`, `Nematic` | bond-orientational order, hexatic order, solid-liquid classification, nematic Q-tensor |
| `LocalDensity`, `GaussianDensity` | per-particle local density and Gaussian-smeared density grid |
| `StaticStructureFactorDebye` | static structure factor S(k) via the Debye equation |
| `BondOrder` | neighbor bond-direction diagram on a (θ, φ) grid |
| `PMFTXY` | 2-D potential of mean force and torque |

They follow the same call convention as `RDF`. The
neighbor-based operators take `(frames, nlists)`; a few take other inputs
(`GaussianDensity` and `StaticStructureFactorDebye` take just `frames`,
`Nematic` reads per-particle directors from the frame's `orientations` topology
block, `ClusterProperties` takes the `Cluster` result):

```python
from molpy.compute import (
    MSD,
    Cluster,
    ClusterCenters,
    GyrationTensor,
    Steinhardt,
    StaticStructureFactorDebye,
)

trajectory = mp.Trajectory([frame, frame])
clusters = Cluster(min_cluster_size=5).compute([frame], [neighbors])
centers = ClusterCenters().compute([frame], clusters)

msd = MSD(method="window").compute(trajectory)  # time series over a trajectory
rg2 = GyrationTensor().compute([frame], clusters, centers)  # per cluster
q6 = Steinhardt(l=[6]).compute([frame], [neighbors])  # Steinhardt q6 per particle
sk = StaticStructureFactorDebye(np.linspace(0.5, 6.0, 32)).compute([frame])  # S(k)
```

## One coordinate copy, and only one

The boundary between molpy and molrs is deliberately copy-free. Coordinates
cross it exactly once, in `frame.coords`, where three separate columns are
stacked into a single contiguous `(N, 3)` array. That reshape is unavoidable as long as coordinates are
stored as separate `x`/`y`/`z` columns. Everything downstream — pair indices,
distances, histogram bins — is a borrowed read-only view into Rust-owned
buffers, so the operators never defensively `.copy()` their inputs and never
mutate the frame you pass in.

## 3D structures are generated through molrs embed

Generating coordinates from a connectivity-only graph also runs on molrs.
`molpy.Conformer` wraps the molrs distance-geometry + minimization
pipeline (ETKDGv3 → torsion refinement → MMFF94 cleanup):

```python

mol = mp.io.read_smiles("CCO")  # ethanol, heavy-atom graph
mol_3d, report = mp.Conformer(add_hydrogens=True, seed=42).generate(mol)
```

`generate` returns the new structure and a report of what each stage did; the
input is untouched.

The RDKit adapter (`molpy.adapter.rdkit`) remains available as an optional
external backend, but the molrs pipeline is the default trunk.
