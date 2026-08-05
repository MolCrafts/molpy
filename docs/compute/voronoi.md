# Voronoi

Overview

| Class / entry | Description |
|---------------|-------------|
| [`RadicalVoronoi`](#radicalvoronoi) | Radical (power) Voronoi tessellation: `build(positions, radii, box)`. |
| [`VoronoiIntegration`](#voronoiintegration) | Integrate electron density over Voronoi cells (per-molecule moments). |
| [`voronoi_domains`](#voronoi_domains) | Connected domains on labeled Voronoi cells. |
| [`voronoi_voids`](#voronoi_voids) | Void volumes from cells flagged empty. |

Details

The `molpy.compute.voronoi` module: radical Voronoi tessellation, domains, voids, integration.

## `RadicalVoronoi`

Radical (power) Voronoi tessellation: `build(positions, radii, box)`.

```python
import numpy as np
import molpy as mp
from molpy.compute import RadicalVoronoi

positions = np.random.default_rng(0).uniform(0, 5, size=(12, 3))
radii = np.ones(12)
box = mp.Box.cubic(5.0)
cells = RadicalVoronoi()(positions, radii, box)
```

## `VoronoiIntegration`

Integrate electron density over Voronoi cells (per-molecule moments).

```python
# docs: skip — needs a volumetric density grid from AIMD post-processing
from molpy.compute import VoronoiIntegration

moments = VoronoiIntegration()(
    positions, radii, atomic_numbers, atom_to_mol, n_mol, grid, box
)
```

## `voronoi_domains`

Connected domains on labeled Voronoi cells.

```python
# docs: skip — requires labels aligned with a live VoronoiCells object
from molpy.compute import voronoi_domains

domains = voronoi_domains(cells, labels)
```

## `voronoi_voids`

Void volumes from cells flagged empty.

```python
# docs: skip — requires is_void flags aligned with a live VoronoiCells object
from molpy.compute import voronoi_voids

voids = voronoi_voids(cells, is_void, box_volume=125.0)
```

## See also

- [Density](density.md)
- [Cluster](cluster.md)
