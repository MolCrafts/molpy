# Shape

Once [Cluster](cluster.md) has told you *which* particles form an aggregate, the
next question is what that aggregate looks like. Is the micelle spherical? Is
the polymer coil swollen or collapsed? Is the protein elongated?

All of it comes from one object: the **gyration tensor**.

## One tensor, four numbers

Take a cluster of $N$ particles with centre $\mathbf{r}_c$ and build the
$3\times3$ matrix of squared displacements about that centre:

$$
S = \frac{1}{N}\sum_{i=1}^{N}
(\mathbf{r}_i-\mathbf{r}_c)\otimes(\mathbf{r}_i-\mathbf{r}_c).
$$

Its trace is the familiar radius of gyration,

$$
R_g^{2} = \operatorname{tr} S = \lambda_1+\lambda_2+\lambda_3,
$$

a single number for overall size. But throwing away everything except the trace
throws away the shape, which lives in the three eigenvalues
$\lambda_1\le\lambda_2\le\lambda_3$. They are the squared extents along the
cluster's own principal axes, so the object is a sphere when they are equal, a
rod when one dominates, and a disc when one is small.

Three conventional combinations name those cases:

$$
b = \lambda_3 - \tfrac12(\lambda_1+\lambda_2), \qquad
c = \lambda_2 - \lambda_1, \qquad
\kappa^{2} = \frac{b^{2} + \tfrac34 c^{2}}{R_g^{4}} .
$$

$b$ is the **asphericity** (zero for a sphere), $c$ the **acylindricity** (zero
for anything with an axis of symmetry), and $\kappa^2$ the **relative shape
anisotropy**, which is bounded: 0 for a sphere or any arrangement with
tetrahedral-or-higher symmetry, 1 for points on a line. A random-walk polymer
coil, despite everyone's mental picture of a fuzzy ball, sits around
$\kappa^2 \approx 0.4$ — it is distinctly aspherical on any given snapshot, and
only looks spherical after you average over orientations.

The **inertia tensor** is the mass-weighted sibling of $S$. Same eigenvectors
when all masses are equal; use it when you need real principal axes of rotation
rather than geometry.

## Testing it where the answer is known

An ideal random-walk chain of $N$ bonds of length $b$ has an exact result:

$$
\langle R_g^{2}\rangle = \frac{N b^{2}}{6},
$$

so $R_g \sim N^{\nu}$ with $\nu = 1/2$. That makes it the right thing to
compute first.

<figure id="fig-chain-rg" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
config:
  legend:
    orient: bottom
    direction: horizontal
    title: null
data: {$file: data/shape/ideal_chain_rg.json}
mark: {type: line, strokeWidth: 2.2, point: true}
encoding:
  x:
    field: n
    type: quantitative
    scale: {type: log}
    title: "chain length N"
  y:
    field: rg
    type: quantitative
    scale: {type: log}
    title: "R_g"
  color:
    field: series
    type: nominal
    title: null
```

</div>

**Figure 1.** Radius of gyration of ideal random-walk chains, 200 chains per
length. Three curves: $\sqrt{\langle R_g^2\rangle}$ and $\langle R_g\rangle$,
both measured with `RadiusOfGyration`, against the exact $\sqrt{Nb^2/6}$. The
fitted slope of the first is $\nu = 0.492$.
</figure>

The measured $\sqrt{\langle R_g^2\rangle}$ tracks the exact curve to within
2 %, and the fitted exponent is 0.492 against the true 0.5. That is the
validation.

Now look at the $\langle R_g\rangle$ curve, and at the fact that it sits
*systematically below* the other two by 2–4 %. That is not an error, and it is worth understanding
because it will bite you elsewhere:

$$
\langle R_g\rangle \;\ne\; \sqrt{\langle R_g^{2}\rangle}.
$$

The formula predicts the mean of the **square**. Averaging $R_g$ itself and
comparing against $\sqrt{Nb^2/6}$ compares two different quantities, and by
Jensen's inequality the mean of a square root is always the smaller one. Whenever
a textbook gives you $\langle X^2 \rangle$, average $X^2$ — not $X$.

For real polymers $\nu$ is the headline number: $\approx 0.588$ in a good
solvent (self-avoiding), $0.5$ at the theta point, $1/3$ for a collapsed
globule. Measuring it is how you tell which regime your simulation is in.

## Computing it

Every descriptor here is **per cluster**, so the pipeline always starts with
[Cluster](cluster.md), and the reference point matters: gyration takes the
geometric centre from `ClusterCenters`, while inertia and $R_g$ take the
mass-weighted `CenterOfMass`. They are not interchangeable arguments.

```python
import numpy as np
import molpy as mp
from molpy.compute import (
    CenterOfMass, Cluster, ClusterCenters, GyrationTensor,
    InertiaTensor, NeighborList, RadiusOfGyration,
)

rng = np.random.default_rng(0)
n_beads = 200
steps = rng.normal(0.0, 1.0 / np.sqrt(3.0), size=(n_beads, 3))
chain = np.cumsum(steps, axis=0) + 100.0

frame = mp.Frame()
frame["atoms"] = {"x": chain[:, 0], "y": chain[:, 1], "z": chain[:, 2]}
frame.box = mp.Box.cubic(200.0)
```

```python
masses = np.full(n_beads, 12.011)
clusters = Cluster(min_cluster_size=5)([frame], [NeighborList(cutoff=2.5)(frame)])
centers = ClusterCenters()([frame], clusters)
com = CenterOfMass(masses)([frame], clusters)

rg = np.asarray(RadiusOfGyration(masses)([frame], clusters, com)[0])
tensor = np.asarray(GyrationTensor()([frame], clusters, centers)[0])
print(round(float(rg[0]), 2), tensor.shape)     # -> 4.84 (1, 3, 3)
```

Predicted $\sqrt{200/6} = 5.77$; a single chain fluctuates by tens of percent
about that, which is exactly why Figure 1 averages 200 of them.

Now unpack the shape from the tensor rather than stopping at $R_g$:

```python
eigenvalues = np.sort(np.linalg.eigvalsh(tensor[0]))
b = eigenvalues[2] - 0.5 * (eigenvalues[0] + eigenvalues[1])
c = eigenvalues[1] - eigenvalues[0]
kappa2 = (b**2 + 0.75 * c**2) / eigenvalues.sum() ** 2
print(round(float(kappa2), 3))                  # -> 0.217
```

$\kappa^2 = 0.22$ for this particular chain: not spherical, and a long
way from a rod either. Single chains scatter widely around the
ensemble value of about 0.4, so quote $\kappa^2$ as an average. Pass `masses=None`
anywhere above for purely geometric (unit-mass) descriptors.

`InertiaTensor(masses)([frame], clusters, com)` returns the mass-weighted
version with the same shape, for principal axes.

## When it goes wrong

**$R_g$ is enormous — comparable to the box.**
The cluster straddles a periodic boundary and was not unwrapped. Cluster
*identification* works under minimum image, but the gyration tensor is computed
from raw coordinates about a centre, so half the molecule appears a box-length
away. This is the single most common failure on this page.

Fix: pick one atom of the cluster as a seed and re-express every other atom as
the nearest periodic image of that seed — `Box.diff_dr` applies the minimum-image
convention (including non-cubic cells):

```python
box = mp.Box.cubic(20.0)
split = np.array(
    [[0.5, 10.0, 10.0], [1.0, 10.0, 10.0], [19.5, 10.0, 10.0], [19.0, 10.0, 10.0]]
)
seed = split[0]
joined = seed + box.diff_dr(split - seed)
print(round(float(np.ptp(split[:, 0])), 2))   # -> 19.0  (PBC-split)
print(round(float(np.ptp(joined[:, 0])), 2))  # -> 2.0   (one molecule again)
```

Four atoms spanning a 20 Å box look 19 Å across; after minimum-image relative to
the seed they are the 2 Å cluster they actually are. Do this **per cluster**,
using the labels from [`Cluster`](cluster.md), before any shape descriptor.

**$\kappa^2 > 1$ or negative.**
An algebra slip: $\kappa^2$ is normalized by $R_g^4$, that is
$(\lambda_1+\lambda_2+\lambda_3)^2$, not by $R_g^2$.

**A sphere gives $\kappa^2$ well above 0.**
Check how many particles you have. Shape descriptors of small clusters are
dominated by noise; a handful of random points is never isotropic.

**Mass-weighted and geometric results differ and you cannot say which you have.**
`masses=None` means unit mass. Report which convention you used — this is
routinely omitted and makes published $R_g$ values incomparable.

**$R_g$ is fine but the eigenvectors jump between frames.**
Degenerate eigenvalues. For a near-spherical object the principal axes are
ill-defined, and no code can fix that.

## Check yourself

- Compute $R_g$ for points on a sphere of radius $R$: you should get
  $R\sqrt{3/5}$ for a filled ball and $R$ for a shell.
- Compute $\kappa^2$ for points on a straight line (should be 1) and for the
  vertices of a regular tetrahedron (should be 0).
- Average $R_g^2$ over many ideal chains, take the square root, and compare with
  $\sqrt{Nb^2/6}$. Then average $R_g$ instead and watch the answer drop.

## References

- D. N. Theodorou, U. W. Suter, *Macromolecules* **18**, 1206 (1985) — the
  gyration tensor, asphericity, and $\kappa^2$.
- M. Rubinstein, R. H. Colby, *Polymer Physics*, Oxford (2003), ch. 2 — ideal
  chains, $\nu$, and the scaling regimes.

## See also

- [Cluster](cluster.md) — produces the aggregates measured here
- [Decomposition](decomposition.md) — reducing many shape descriptors at once
- [Order](order.md) · [API reference](../api/compute.md)
