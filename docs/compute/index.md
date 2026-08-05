# Compute

Trajectory and structure analyses. Import with `from molpy.compute import ...`.

Heavy numerics run in **molrs** (Rust). MolPy re-exports those types identity-style.
Transport and dielectric quantities use an explicit **compose** pattern: raw Compute → Fit → optional SI scale.

Layout: sidebar **sections**, one page per `molpy.compute` package module, and on each page **one heading per Compute class**.

## Patterns

### Frame analyses

```python
import numpy as np
import molpy as mp

rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 10.0, size=(40, 3))
frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box.cubic(10.0)
```

```python
from molpy.compute import NeighborList, RDF

nlist = NeighborList(cutoff=5.0)(frame)
result = RDF(n_bins=40, r_max=5.0)([frame], [nlist])
result.rdf, result.bin_centers
```

### Compose (transport / dielectric)

```python
import numpy as np
from molpy.compute import EinsteinConductivity, LinearFit

rng = np.random.default_rng(0)
M = np.ascontiguousarray(np.cumsum(rng.normal(0, 0.01, size=(40, 3)), axis=0))
raw = EinsteinConductivity().compute(M, 10.0, 15)
fit = LinearFit(0.1, 0.5).fit(raw["lag_times"], raw["msd"])
fit["slope"]
```

!!! note "Units"
    Time **fs** (LAMMPS *real*), length Å, charge $e$. Displacement kernels need **unwrapped** coordinates.

## Sections

### Structure

| Package | Page |
|---------|------|
| `molpy.compute.neighborlist` | [NeighborList](neighborlist.md) |
| `molpy.compute.rdf` | [RDF](rdf.md) |
| `molpy.compute.density` | [Density](density.md) |
| `molpy.compute.diffraction` | [Diffraction](diffraction.md) |
| `molpy.compute.pmft` | [PMFT](pmft.md) |

### Distributions

| Package | Page |
|---------|------|
| `molpy.compute.distribution` | [Distribution](distribution.md) |
| `molpy.compute.spatial` | [Spatial](spatial.md) |

### Order

| Package | Page |
|---------|------|
| `molpy.compute.order` | [Order](order.md) |
| `molpy.compute.environment` | [Environment](environment.md) |

### Shape & Clustering

| Package | Page |
|---------|------|
| `molpy.compute.shape` | [Shape](shape.md) |
| `molpy.compute.cluster` | [Cluster](cluster.md) |
| `molpy.compute.decomposition` | [Decomposition](decomposition.md) |

### Bonds & Voronoi

| Package | Page |
|---------|------|
| `molpy.compute.hbond` | [HBond](hbond.md) |
| `molpy.compute.voronoi` | [Voronoi](voronoi.md) |

### Transport

| Package | Page |
|---------|------|
| `molpy.compute.msd` | [MSD](msd.md) |
| `molpy.compute.pmsd` | [PMSD](pmsd.md) |
| `molpy.compute.jacf` | [JACF](jacf.md) |
| `molpy.compute.onsager` | [Onsager](onsager.md) |
| `molpy.compute.persist` | [Persist](persist.md) |

### Dynamics

| Package | Page |
|---------|------|
| `molpy.compute.van_hove` | [Van Hove](van_hove.md) |
| `molpy.compute.reorientation` | [Reorientation](reorientation.md) |

### Spectroscopy

| Package | Page |
|---------|------|
| `molpy.compute.dielectric` | [Dielectric](dielectric.md) |
| `molpy.compute.spectra` | [Spectra](spectra.md) |
| `molpy.compute.signal` | [Signal](signal.md) |

### Workflow

| Package | Page |
|---------|------|
| `molpy.compute.workflow` | [Workflow](workflow.md) |

!!! note "Not re-exported"
    `VACF` / `GreenKuboDiffusion` / `EinsteinDiffusion` live on `molrs.compute.transport`.
    Shared `Compute` / result types: [API: Compute](../api/compute.md).
