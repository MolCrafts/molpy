# molrs Backend

MolPy's analysis operators are thin Python shells over [molrs](https://github.com/MolCrafts/molrs),
a Rust column store and compute kernel. molrs is a **required** runtime
dependency — `molpy.core.frame` imports it at module load, every `Frame` and
`Block` is backed by a Rust `Store`, and the `compute` operators forward
straight into Rust. There is no pure-Python fallback and no opt-in flag.

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

There is no `molpy[molrs]` extra to remember — that key was removed. If you are
upgrading from an older release, see the [changelog](../../changelog.md): molrs
moving from optional to required is a breaking change.

## The box is a molrs object, not a copy of one

`molpy.Box` does not wrap a molrs box; it **inherits** from it:

```python
import molrs
from molpy.core.box import Box

class Box(molrs.Box):
    ...
```

The practical consequence is that a molpy box can be handed to any molrs API
unchanged — there is no `.to_molrs()` bridge and no coordinate translation:

```python
import molrs
import molpy as mp

box = mp.Box.cubic(10.0)
assert isinstance(box, molrs.Box)   # it *is* a molrs box
```

Likewise `frame.box` is accepted directly by Rust-side calls such as
`molrs.NeighborQuery`. The enriched molpy methods (`Style`, `cubic`,
`from_lengths_angles`, `diff_dr`, …) remain available on top of the inherited
Rust core.

## Neighbor lists come from the linked-cell kernel

`NeighborList` searches for all pairs within a cutoff using molrs's
linked-cell algorithm (O(N) in the number of atoms). It returns the molrs
`NeighborList` result object directly — molpy does not re-wrap it:

```python
import numpy as np
import molpy as mp
from molpy.compute import NeighborList

rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 20.0, size=(500, 3))

frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box.cubic(20.0)

neighbors = NeighborList(cutoff=8.0)(frame)
print(neighbors.n_pairs)          # number of pairs found
print(neighbors.distances[:5])    # pair distances, borrowed from Rust
```

A periodic box is required: calling `NeighborList` on a free box raises
`ValueError`, because there is no minimum-image convention to apply.

## The RDF reuses the neighbor list it is given

`RDF` computes the radial distribution function

$$ g(r) = \frac{V}{N\,N_q}\,\frac{\langle n(r)\rangle}{4\pi r^2\,\Delta r} $$

from one or more frames plus the neighbor list for each. Passing the neighbor
list in explicitly keeps the expensive pair search out of the histogram loop
and lets you reuse a single search for several analyses:

```python
from molpy.compute import RDF

result = RDF(n_bins=50, r_max=8.0)(frame, neighbors)
print(result.bin_centers)   # r at each bin centre
print(result.rdf)           # g(r)
```

For an ideal gas (uniformly random points) the middle bins of `result.rdf`
sit near 1.0, which is the standard sanity check for a correct normalization.
Multiple frames are averaged when you pass lists: `RDF(...)(frames, neighbor_lists)`.

## The wider analysis catalog is exposed as molpy operators

A range of standard trajectory analyses already live in molrs. MolPy exposes
each as a small `Compute` operator that forwards arguments and returns the
molrs result type unchanged:

| Operator | What it computes |
|----------|------------------|
| `MSD` | mean-squared displacement vs. lag time |
| `Cluster`, `ClusterCenters`, `ClusterProperties` | connected-component clustering, centroids, and per-cluster size/mass/gyration |
| `CenterOfMass` | mass-weighted centroid |
| `GyrationTensor`, `RadiusOfGyration`, `InertiaTensor` | shape descriptors |
| `Pca`, `KMeans` | dimensionality reduction and partitioning |
| `Steinhardt`, `Hexatic`, `SolidLiquid`, `Nematic` | bond-orientational order, hexatic order, solid-liquid classification, nematic Q-tensor |
| `LocalDensity`, `GaussianDensity` | per-particle local density and Gaussian-smeared density grid |
| `StaticStructureFactorDebye` | static structure factor S(k) via the Debye equation |
| `BondOrder` | neighbor bond-direction diagram on a (θ, φ) grid |
| `PMFTXY` | 2-D potential of mean force and torque |

They follow the same call convention as `NeighborList` / `RDF`. The
neighbor-based operators take `(frames, nlists)`; a few take other inputs
(`GaussianDensity` and `StaticStructureFactorDebye` take just `frames`,
`Nematic` takes per-particle `directors`, `ClusterProperties` takes the
`Cluster` result):

```python
from molpy.compute import MSD, GyrationTensor, Steinhardt, StaticStructureFactorDebye

msd = MSD()(frames)                            # time series over a trajectory
rg2 = GyrationTensor()(frame)                  # gyration tensor for one frame
q6 = Steinhardt([6])(frame, neighbors)         # Steinhardt q6 per particle
sk = StaticStructureFactorDebye(k_values)(frame)   # structure factor S(k)
```

## One coordinate copy, and only one

The boundary between molpy and molrs is deliberately copy-free. Coordinates
cross it exactly once, inside `frame["atoms"][["x", "y", "z"]]`, where three
separate columns are stacked into a single contiguous `(N, 3)` array via
`numpy.column_stack`. That reshape is unavoidable as long as coordinates are
stored as separate `x`/`y`/`z` columns. Everything downstream — pair indices,
distances, histogram bins — is a borrowed read-only view into Rust-owned
buffers, so the operators never defensively `.copy()` their inputs and never
mutate the frame you pass in.

## 3D structures are generated through molrs embed

Generating coordinates from a connectivity-only graph also runs on molrs.
`molpy.compute.Generate3D` wraps the molrs distance-geometry + minimization
pipeline:

```python
from molpy.parser import parse_molecule
from molpy.compute import Generate3D

mol = parse_molecule("CCO")          # ethanol, heavy-atom graph
mol_3d = Generate3D(seed=42)(mol)    # fresh structure, input untouched
```

The RDKit adapter (`molpy.adapter.rdkit`) remains available as an optional
external backend, but the molrs pipeline is the default trunk.
