# Decomposition

Textbook guide to **PCA and k-means** on descriptor tables — reducing
trajectories to a handful of structural coordinates and discrete states.

---

## 1. PCA

A trajectory analyzed with [Shape](shape.md) / [Order](order.md) yields a
high-dimensional table (one row of descriptors per configuration). **Principal
component analysis** re-expresses that table in the orthogonal directions of
greatest variance. The first two components usually capture the dominant motion.

**Always standardize columns** before PCA — otherwise one large-magnitude
feature dominates.

## 2. K-means

Given reduced coordinates, **k-means** partitions into $k$ clusters (Lloyd).
It turns a continuous PCA map into discrete states (folded/unfolded,
paired/free). $k$ is a modelling choice: try several and check stability.

---

## 3. Usage

```python
import numpy as np
from molpy.compute import Pca, DescriptorRow, KMeans

rng = np.random.default_rng(0)
descriptor_matrix = rng.normal(size=(50, 8))
rows = [DescriptorRow(r) for r in descriptor_matrix]
pca = Pca()(rows)
labels = KMeans(k=3, max_iter=100, seed=0)(pca)
```

---

## 4. Pitfalls

1. Unscaled features before PCA/k-means.
2. Reading too much into $k$ — k-means always returns $k$ clusters.
3. Mixing incomparable units across columns.

## See also

- [Shape](shape.md) · [Cluster](cluster.md) · [Order](order.md)
- [API reference](../api/compute.md)
