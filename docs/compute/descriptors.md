# Molecular Shape, Clustering & Decomposition

This page is a self-contained, textbook-style introduction to MolPy's
**descriptor** operators: the shape tensors that summarize how a molecule or
aggregate is spread out in space, the clustering tools that find those aggregates
in the first place, and the PCA / k-means primitives for organizing large sets of
descriptors. The canonical applications are polymer coil/globule analysis,
micelle and aggregate detection, and reducing a trajectory to a handful of
interpretable structural coordinates.

As with every `compute` operator, the per-particle reductions run in Rust
(`molrs`); MolPy supplies frames, the cluster assignment, and per-particle masses,
and returns the native result. A key structural point: the shape descriptors are
defined **per cluster**, so the usual workflow is *find clusters → reduce each
cluster to a tensor*.

!!! note "Conventions used throughout"
    - Length is in Å; the gyration radius $R_g$ is in Å, the gyration tensor in Å².
    - A **cluster** is a group of particles — a single molecule, or a physically
      connected aggregate found by `Cluster`.
    - Tensors are $3\times3$; their eigenvalues are the principal values.

---

## 1. The gyration tensor measures molecular size

The size and shape of a group of particles are captured by the **gyration
tensor**, the mean outer product of positions about the group's center
$\mathbf r_\mathrm{c}$:

$$
S = \frac{1}{N}\sum_{i=1}^{N}\big(\mathbf r_i - \mathbf r_\mathrm{c}\big)\otimes\big(\mathbf r_i - \mathbf r_\mathrm{c}\big).
$$

Its trace is the squared **radius of gyration**, the most common single measure of
coil size:

$$
R_g^2 = \operatorname{tr} S = \frac{1}{N}\sum_{i=1}^{N}\big|\mathbf r_i - \mathbf r_\mathrm{c}\big|^2 .
$$

For a polymer chain $R_g$ scales as $N^\nu$ with the Flory exponent $\nu$, so its
dependence on chain length distinguishes a swollen coil ($\nu\approx 3/5$) from a
collapsed globule ($\nu = 1/3$).

---

## 2. Shape anisotropy comes from the gyration-tensor eigenvalues

Diagonalize $S$ to get principal values $\lambda_1\le\lambda_2\le\lambda_3$. Their
combinations are rotation-invariant shape descriptors:

$$
b = \lambda_3 - \tfrac{1}{2}(\lambda_1 + \lambda_2)\ \text{(asphericity)},\qquad
c = \lambda_2 - \lambda_1\ \text{(acylindricity)},
$$

$$
\kappa^2 = \frac{b^2 + \tfrac{3}{4}c^2}{R_g^4}\ \text{(relative shape anisotropy)} .
$$

$\kappa^2 = 0$ for a sphere (or a perfectly symmetric arrangement) and $\to 1$ for
a rod. These are how you tell a spherical micelle from a worm-like one without
ever looking at a picture.

---

## 3. The inertia tensor gives the principal axes

The mass-weighted cousin of the gyration tensor is the **moment-of-inertia
tensor**

$$
I = \sum_{i=1}^{N} m_i\big(|\mathbf r_i'|^2\,\mathbf{1} - \mathbf r_i'\otimes\mathbf r_i'\big),
\qquad \mathbf r_i' = \mathbf r_i - \mathbf r_\mathrm{cm},
$$

whose eigenvectors are the principal axes of the body and whose eigenvalues order
them from the long axis (smallest $I$) to the short. Use it to define a molecular
frame — for orientational analyses, or to align structures for averaging.

---

## 4. Computing shape descriptors

Shape operators consume a cluster assignment and a per-cluster center, so they
chain after a neighbor list and a clustering step. Each particle's molecule (or a
connected aggregate) is one cluster.

```python
from molpy.compute import (
    NeighborList, Cluster, CenterOfMass, RadiusOfGyration,
    GyrationTensor, InertiaTensor,
)

nlist = NeighborList(cutoff=1.6)(frame)
clusters = Cluster(min_cluster_size=10)([frame], [nlist])     # one ClusterResult / frame

com = CenterOfMass(masses)([frame], clusters)                 # mass-weighted centers
rg  = RadiusOfGyration(masses)([frame], clusters, com)        # R_g per cluster
S   = GyrationTensor()([frame], clusters, com)                # 3×3 tensor per cluster
I   = InertiaTensor(masses)([frame], clusters, com)           # inertia tensor per cluster
```

Pass `masses=None` to fall back to unit mass (geometric, unweighted descriptors).

---

## 5. Finding aggregates with `Cluster`

Before you can describe an aggregate you have to find it. **`Cluster`** builds a
connectivity graph from a `NeighborList` and returns the connected components
larger than `min_cluster_size` — micelles, droplets, percolating networks. The
companion **`ClusterProperties`** reduces each cluster to its size, center, mass,
gyration tensor, and $R_g$ in one call.

```python
from molpy.compute import ClusterProperties

clusters = Cluster(min_cluster_size=20)([frame], [nlist])
props = ClusterProperties()([frame], clusters)   # one dict of per-cluster properties / frame
```

The neighbor cutoff *is* the physical definition of "connected", so choose it from
the first minimum of $g(r)$ (see the [structural guide](structure.md)).

---

## 6. PCA reduces a descriptor set to its dominant variations

A trajectory analyzed with the operators above yields a high-dimensional table —
one row of descriptors per configuration. **Principal component analysis**
re-expresses that table in the orthogonal directions of greatest variance (the
eigenvectors of the covariance matrix), so that the first two components usually
capture the dominant structural motion. `Pca` projects a list of `DescriptorRow`
vectors onto its two leading components:

```python
from molpy.compute import Pca, DescriptorRow

rows = [DescriptorRow(r) for r in descriptor_matrix]   # each r is a 1-D float array
pca = Pca()(rows)                                       # 2-component projection
```

Scale or standardize the descriptors before PCA — otherwise a single
large-magnitude column dominates the variance and the components are meaningless.

---

## 7. K-means groups configurations into states

Given the reduced coordinates, **k-means** partitions them into $k$ clusters by
iteratively assigning each point to its nearest centroid and recomputing the
centroids (Lloyd's algorithm). It is the simplest way to turn a continuous PCA map
into discrete structural states — folded vs unfolded, paired vs free.

```python
from molpy.compute import KMeans

labels = KMeans(k=3, max_iter=100, seed=0)(pca)
```

$k$ is a modelling choice, not an output: try several and check that the clusters
are stable and physically interpretable.

---

## 8. Pitfalls checklist

1. **Mass weighting** → $R_g$ and the inertia tensor change with the mass
   convention; pass real masses for physical principal axes, `None` for geometric.
2. **Periodic images** → a molecule split across a box boundary has a nonsense
   $R_g$. Unwrap whole molecules (minimum-image relative to the cluster center)
   before computing shape.
3. **Cluster cutoff** → too large merges distinct aggregates, too small fragments
   one; read it off $g(r)$ and check the cluster-size distribution is stable.
4. **Unscaled features before PCA/k-means** → standardize columns; otherwise units
   dominate the result.
5. **Reading too much into $k$** → k-means always returns $k$ clusters even when
   the data have none; validate against a held-out metric or the PCA scatter.

---

## 9. References

- M. Rubinstein, R. H. Colby, *Polymer Physics*, Oxford (2003) — radius of
  gyration and chain-size scaling.
- D. N. Theodorou, U. W. Suter, *Macromolecules* **18**, 1206 (1985) — gyration
  tensor, asphericity, and relative shape anisotropy.
- I. T. Jolliffe, *Principal Component Analysis*, 2nd ed., Springer (2002).
- J. B. MacQueen, *Proc. 5th Berkeley Symp.* **1**, 281 (1967) — k-means.
- V. Ramasubramani et al., *Comput. Phys. Commun.* **254**, 107275 (2020) — the
  freud library, on which the cluster/shape kernels are modelled.

## See also

- [Structural Analysis](structure.md) — pick the cluster/neighbour cutoff from $g(r)$.
- [Bond-Orientational Order & Local Environment](order.md) — per-particle order to
  cluster on.
- [Compute overview](index.md) — the Compute → Result pattern.
- [API reference: Compute](../api/compute.md).
