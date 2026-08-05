# Shape

Textbook guide to **molecular shape descriptors**: gyration tensor, radius of
gyration, asphericity, and the inertia tensor.

!!! note "Conventions"
    - Length Å; $R_g$ in Å; tensors $3\times 3$.
    - Descriptors are **per cluster** — find aggregates first ([Cluster](cluster.md)).

---

## 1. Gyration tensor and $R_g$

$$
S = \frac{1}{N}\sum_{i=1}^N
(\mathbf{r}_i-\mathbf{r}_c)\otimes(\mathbf{r}_i-\mathbf{r}_c),
\qquad
R_g^2 = \operatorname{tr} S.
$$

Mass-weighted form uses COM and masses $\{m_i\}$. For polymers
$R_g\sim N^\nu$ with Flory exponent $\nu$ (swollen $\approx 3/5$, $\theta=1/2$,
globule $1/3$).

Eigenvalues $\lambda_1\le\lambda_2\le\lambda_3$ define asphericity
$b=\lambda_3-\tfrac12(\lambda_1+\lambda_2)$, acylindricity
$c=\lambda_2-\lambda_1$, and relative shape anisotropy

$$
\kappa^2 = (b^2 + \tfrac34 c^2)/R_g^4
$$

($0$ sphere → $1$ rod).

The **inertia tensor** is the mass-weighted cousin; its eigenvectors are
principal axes for molecular frames and orientational analyses.

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
from molpy.compute import (
    NeighborList, Cluster, ClusterCenters, CenterOfMass,
    RadiusOfGyration, GyrationTensor, InertiaTensor,
)

masses = np.full(len(xyz), 12.011)
nlist = NeighborList(cutoff=1.6)(frame)
clusters = Cluster(min_cluster_size=10)([frame], [nlist])
centers = ClusterCenters()([frame], clusters)
com = CenterOfMass(masses)([frame], clusters)
rg = RadiusOfGyration(masses)([frame], clusters, com)
S = GyrationTensor()([frame], clusters, centers)
I = InertiaTensor(masses)([frame], clusters, com)
```

Pass `masses=None` for geometric (unit-mass) descriptors.

---

## 3. Pitfalls

1. Molecule split across PBC → unwrap relative to cluster center.
2. Mass convention not reported.
3. Cluster cutoff wrong → garbage aggregates.

## See also

- [Cluster](cluster.md) · [Decomposition](decomposition.md) · [RDF](rdf.md)
- [API reference](../api/compute.md)
