# Cluster

Textbook guide to **connectivity clustering** — finding aggregates from a
neighbor graph before shape or property reductions.

---

## 1. Connected components as aggregates

`Cluster` builds a graph from a [NeighborList](neighborlist.md) and returns
connected components larger than `min_cluster_size` — micelles, droplets,
percolating networks. The neighbor cutoff **is** the physical definition of
"bonded"; read it from the first minimum of $g(r)$ ([RDF](rdf.md)).

`ClusterProperties` reduces each cluster to size, center, mass, gyration tensor,
and $R_g$ in one call. Shape operators on [Shape](shape.md) consume the same
cluster assignment.

---

## 2. Usage

```python
import numpy as np
import molpy as mp

rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 20.0, size=(200, 3))
frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box.cubic(20.0)
```

```python
from molpy.compute import NeighborList, Cluster, ClusterProperties

nlist = NeighborList(cutoff=1.6)(frame)
clusters = Cluster(min_cluster_size=20)([frame], [nlist])
props = ClusterProperties()([frame], clusters)
```

---

## 3. Pitfalls

1. Cutoff too large merges distinct aggregates; too small fragments one.
2. Ignoring the cluster-size distribution when validating the cutoff.
3. PBC-split clusters without unwrapping.

## See also

- [Shape](shape.md) · [NeighborList](neighborlist.md) · [RDF](rdf.md)
- [API reference](../api/compute.md)
