# Decomposition

Overview

| Class / entry | Description |
|---------------|-------------|
| [`Pca`](#pca) | Two-component PCA over a list of `DescriptorRow`. |
| [`KMeans`](#kmeans) | k-means over a `Pca` result. |

Details

The `molpy.compute.decomposition` module: PCA and k-means on descriptor rows.

## `Pca`

Two-component PCA over a list of `DescriptorRow`.

```python
import numpy as np
from molpy.compute import DescriptorRow, Pca, KMeans

rng = np.random.default_rng(0)
rows_a = rng.normal(loc=[0, 0, 0, 0], scale=0.1, size=(20, 4))
rows_b = rng.normal(loc=[5, 5, 5, 5], scale=0.1, size=(20, 4))
rows = [DescriptorRow(r.astype(np.float64)) for r in np.vstack([rows_a, rows_b])]

pca = Pca()(rows)
pca.coords.shape  # (40, 2)
```

## `KMeans`

k-means over a `Pca` result.

```python
labels = KMeans(k=2, seed=42)(pca).labels
```

## See also

- [Shape](shape.md)
- [Cluster](cluster.md)
