# Cluster

Overview

| Class / entry | Description |
|---------------|-------------|
| [`Cluster`](#cluster) | Connected components from a neighbor list. |
| [`ClusterCenters`](#clustercenters) | Geometric centers of clusters. |
| [`ClusterProperties`](#clusterproperties) | Size and shape properties of clusters. |

Details

The `molpy.compute.cluster` module: connected components and cluster properties.

## `Cluster`

Connected components from a neighbor list.

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
from molpy.compute import NeighborList, Cluster

nlist = NeighborList(cutoff=3.0)(frame)
clusters = Cluster(min_cluster_size=1)(frame, nlist)
clusters.num_clusters
```

## `ClusterCenters`

Geometric centers of clusters.

```python
from molpy.compute import ClusterCenters

centers = ClusterCenters()(frame, clusters)
centers.centers
```

## `ClusterProperties`

Size and shape properties of clusters.

```python
from molpy.compute import ClusterProperties

# Sequence of frames + matching ClusterResult sequence
props = ClusterProperties()([frame], [clusters])
```

## See also

- [Shape](shape.md)
- [NeighborList](neighborlist.md)
