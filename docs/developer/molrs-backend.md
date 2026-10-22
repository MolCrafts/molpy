# The Rust Core (molrs)

molpy's native classes are the Rust core's, [molrs](https://github.com/MolCrafts/molrs/tree/dev/crates/molrs):
a column store and compute kernels, compiled into the wheel as `molpy._core`
and published in the subsystem module named after their Rust owner
(`mp.core` is `molrs::core`, `mp.compute` is `molrs::compute`). `mp.Frame` and
`mp.Block` hold their columns in Rust storage, and every class on
`mp.compute` is a native class. There is no pure-Python fallback, no opt-in
flag and no second package to install.

This page shows how that core surfaces in everyday analysis: the box type,
neighbor lists and radial distribution functions, and the rest of the
analysis catalog as seen from Python.

## One wheel, one version

`pip install molcrafts-molpy` installs the Python package and its native core
together: they are one wheel, built from one commit, so they cannot drift
apart. `mp.__version__` is the core's version (`[workspace.package].version`
in the project's `Cargo.toml`).

## The box is the native box

`mp.Box` is the core's simulation cell, not a Python wrapper around it, as
`mp.Trajectory` is the core's trajectory.

The examples below share this setup:

```python
import numpy as np
import molpy as mp


def _frame(step: int) -> mp.Frame:
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0.0, 20.0, size=(200, 3)) + 0.1 * step
    frame = mp.Frame()
    frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    frame.box = mp.Box(mp.core.Cell.lengths([20.0] * 3))
    return frame


frames = [_frame(step) for step in range(20)]
```

The practical consequence is that a box goes into any native call unchanged,
as the same object with the same coordinates: `frame.box` is accepted directly by Rust-side calls such as
`mp.core.NeighborList.build`. Everything — the constructor (`Box(cell,
origin=(0, 0, 0), pbc=(True, True, True))`, the cell a `(3, 3)` matrix or a
`mp.core.Cell`), the factories (`Box.cube`, `Box.ortho`,
`Box.from_lengths_tilts`, `Box.from_lengths_angles`, `Box.bounding`) and the
geometry (`wrap`, `unwrap`, `delta`, `distances`, `to_frac`, …) — is the
native box's.

## Neighbor lists come from the linked-cell kernel

`mp.core.NeighborList` (the native class) searches for all pairs within a cutoff
using a linked-cell algorithm (O(N) in the number of atoms). `build` indexes
the coordinates and `neighbors()` returns the pair table, a `mp.core.Neighbors`:

```python
import numpy as np
import molpy as mp

rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 20.0, size=(500, 3))

frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box(mp.core.Cell.lengths([20.0] * 3))

nl = mp.core.NeighborList(8.0)
nl.build(frame.coords, frame.box)
neighbors = nl.neighbors()
print(neighbors.n_pairs)  # number of pairs found
print(np.sqrt(neighbors.dist_sq())[:5])  # pair distances (Å) from stored dist_sq
```

`build` needs a box: a frame without one (`frame.box is None`) raises
`TypeError`. A free box (no periodic axis) searches without minimum images.

## The Rdf reuses the neighbor list it is given

`Rdf` computes the radial distribution function

$$ g(r) = \frac{V}{N\,N_q}\,\frac{\langle n(r)\rangle}{4\pi r^2\,\Delta r} $$

from one pair table per frame. Passing the pair tables in explicitly keeps the
expensive pair search out of the histogram loop and lets you reuse a single
search for several analyses. The constructor takes the binning, and
`compute` takes the data by keyword:

```python
from molpy.compute import Rdf

result = Rdf(n_bins=50, r_max=8.0, r_min=0.0).compute(pairs=[neighbors])
print(result.bin_centers.shape)  # -> (50,)
print(result.n_frames)           # -> 1
```

For an ideal gas (uniformly random points) the middle bins of `result.rdf`
sit near 1.0, which is the standard sanity check for a correct normalization.
Several frames are averaged by passing one pair table per frame; a built
`NeighborList` can be passed in place of its table, and the histogram then
streams its pairs without materialising them.

## The wider analysis catalog

A range of standard trajectory analyses live in the core. Each class on
`mp.compute` is configured by its constructor and fed by its `compute(...)`:

| Operator | What it computes |
|----------|------------------|
| `Msd`, `Vacf`, `ReorientationAcf`, `Survival`, … | time correlations, each a configuration of the one `Correlator` |
| `LinearFit`, `Plateau`, `DebyeFit`, `CumulativeTrapezoid` | fits that read a `TimeCorrelation` |
| `Cluster` | connected-component clustering; its result is a partition |
| `CenterOfMass`, `GyrationTensor`, `InertiaTensor` | per-group shape descriptors over any partition |
| `Pca`, `Kmeans` | principal components and k-means over any feature matrix |
| `Steinhardt`, `Hexatic`, `SolidLiquid`, `Nematic` | bond-orientational order, hexatic order, solid-liquid classification, nematic Q-tensor |
| `LocalDensity`, `GaussianDensity` | per-particle local density and Gaussian-smeared density grid |
| `StaticStructureFactorDebye` | static structure factor S(k) via the Debye equation |
| `BondOrientationalOrder` | neighbor bond-direction diagram on a (θ, φ) grid |
| `PmftXy` | 2-D potential of mean force and torque |

Every one follows the same convention as `Rdf`: configuration in the
constructor, data by keyword in `compute`. None of them reads a frame; the
columns come from the Frame/Block API on the caller's side. Results chain into
the next compute as they are — a `ClusterResult` is the partition
`GyrationTensor` groups by, a `SteinhardtResult` the feature matrix `Pca`
reads, an `Msd` curve what `LinearFit` fits:

```python
from molpy.compute import (
    Cluster,
    GyrationTensor,
    LinearFit,
    Msd,
    StaticStructureFactorDebye,
    Steinhardt,
)

positions = frame.coords
clusters = Cluster(min_cluster_size=5).compute(pairs=neighbors)
rg = GyrationTensor().compute(
    positions=positions,
    simbox=frame.box,
    partition=clusters,
    weights=np.ones(len(positions)),
).radius_of_gyration()                                    # one R_g per cluster
q6 = Steinhardt(l=[6]).compute(neighbors=neighbors).ql   # (N, n_l): q6 per particle
sk = StaticStructureFactorDebye(np.linspace(0.5, 6.0, 32)).compute(positions=positions)

walk = np.cumsum(rng.normal(0.0, 0.1, size=(200, 10, 3)), axis=0)  # (T, N, 3)
msd = Msd(max_lag=50).compute(positions=walk, dt=10.0)
slope = LinearFit(0.2, 0.8).compute(correlation=msd, channel=0).slope
print(rg.shape == (clusters.n_clusters,), q6.shape, sk.sk.shape)  # -> True (500, 1) (32,)
```

## One coordinate copy, and only one

The boundary between Python and the Rust core is deliberately copy-free. Coordinates
cross it exactly once, in `frame.coords`, where three separate columns are
stacked into a single contiguous `(N, 3)` array. That reshape is unavoidable as long as coordinates are
stored as separate `x`/`y`/`z` columns. Everything downstream — pair indices,
distances, histogram bins — is a borrowed read-only view into Rust-owned
buffers, so the operators never defensively `.copy()` their inputs and never
mutate the frame you pass in.

## 3D structures are generated by the native embedder

Generating coordinates from a connectivity-only graph also runs in the core.
`mp.conformer.Conformer` is the native distance-geometry + minimization
pipeline (ETKDGv3 → torsion refinement → MMFF94 cleanup):

```python

mol = mp.io.smiles.read_smiles("CCO").to_atomistic()  # ethanol, heavy-atom graph
mol_3d, report = mp.conformer.Conformer(add_hydrogens=True, seed=42).generate(mol)
```

`generate` returns the new structure and a report of what each stage did; the
input is untouched.

The native pipeline is the only embedder.
