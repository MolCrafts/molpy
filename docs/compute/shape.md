# Shape

Overview

| Class / entry | Description |
|---------------|-------------|
| [`CenterOfMass`](#centerofmass) | Center of mass per cluster. |
| [`GyrationTensor`](#gyrationtensor) | Gyration tensor per cluster (pass `ClusterCenters` result). |
| [`InertiaTensor`](#inertiatensor) | Inertia tensor per cluster (pass `CenterOfMass` result). |
| [`RadiusOfGyration`](#radiusofgyration) | Scalar $R_g$ per cluster. |

Details

The `molpy.compute.shape` module: shape tensors of **clusters** (needs a `Cluster` result).

## `CenterOfMass`

Center of mass per cluster.

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
from molpy.compute import NeighborList, Cluster, CenterOfMass, ClusterCenters

nlist = NeighborList(cutoff=3.0)(frame)
clusters = Cluster(min_cluster_size=1)(frame, nlist)
com = CenterOfMass()(frame, clusters)
centers = ClusterCenters()(frame, clusters)
```

## `GyrationTensor`

Gyration tensor per cluster (pass `ClusterCenters` result).

```python
from molpy.compute import GyrationTensor

G = GyrationTensor()(frame, clusters, centers)
```

## `InertiaTensor`

Inertia tensor per cluster (pass `CenterOfMass` result).

```python
from molpy.compute import InertiaTensor

I = InertiaTensor()(frame, clusters, com)
```

## `RadiusOfGyration`

Scalar $R_g$ per cluster.

```python
from molpy.compute import RadiusOfGyration

rg = RadiusOfGyration()(frame, clusters, com)
```

## See also

- [Cluster](cluster.md)
- [Decomposition](decomposition.md)
