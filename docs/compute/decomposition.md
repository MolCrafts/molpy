# Decomposition

By the time you have run [Order](order.md), [Shape](shape.md), and
[Density](density.md) over a trajectory, every atom carries half a dozen
numbers. That is too many to plot and too few to be a model. You want the two or
three combinations that actually distinguish the configurations, and then a
label saying which state each configuration is in.

`Pca` finds the combinations. `KMeans` assigns the labels.

## PCA finds the directions that vary

Principal component analysis takes a table with one row per sample and one
column per descriptor, and rotates the coordinate axes so that the first new
axis lies along the direction of greatest variance, the second along the
greatest remaining variance, and so on. Keeping only the leading axes is what
turns the rotation into a reduction — and MolPy's `Pca` makes that choice for
you at two, as the API section below spells out.

The reason it works as *analysis* rather than mere compression: descriptors are
usually redundant. $q_4$, $q_6$, coordination number, and local density all rise
and fall together when a region crystallizes. PCA notices that they are one
underlying variable wearing four hats, and hands you that variable.

**Standardize the columns first.** This is not a refinement, it is a
prerequisite. PCA maximizes variance, and variance has units. A coordination
number of order 12 and a $q_6$ of order 0.5 differ by a factor of 24, so on raw
columns the first principal component is "coordination number" no matter what
the physics is. Subtract the mean and divide by the standard deviation of each
column, and PCA is comparing shapes of distributions instead of choices of unit.

## Does it work? Crystal against liquid

Here is the honest test. Take a perfect FCC crystal and liquid argon, compute
$q_4$, $q_6$, and coordination for every atom, standardize, and project onto the
first two components — **without ever telling the algorithm which is which**.

<figure id="fig-phase-map" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
config:
  legend:
    orient: bottom
    direction: horizontal
    title: null
data: {$file: data/decomposition/phase_map.json}
mark: {type: point, size: 18, filled: true, opacity: 0.45}
encoding:
  x:
    field: pc1
    type: quantitative
    title: "PC1"
  y:
    field: pc2
    type: quantitative
    title: "PC2"
  color:
    field: phase
    type: nominal
    title: null
```

</div>

**Figure 1.** First two principal components of per-atom order descriptors for
400 FCC atoms and 400 liquid-argon atoms. Colour is the true phase, which the
decomposition never saw. The two clouds separate cleanly along PC1.
</figure>

The two phases land in disjoint regions along PC1, which carries a variance of
2.44 against 0.49 for PC2 — so a single coordinate holds most of the
information. That is the useful output: you started with three descriptors and
found that one combination of them is the order parameter.

Running k-means with $k=2$ on the same data and comparing its labels with the
truth gives **99.75 % agreement**. The algorithm recovered the phases from
geometry alone.

Be careful about what that demonstrates. FCC and a liquid at the triple point
are about as separable as two states ever get. The honest lesson is the method,
not the score — and the corresponding warning is that **k-means always returns
$k$ clusters**, including on data with no clusters at all. It will happily
partition a single Gaussian blob into two halves and report tidy labels. The
number $k$ is a hypothesis you are imposing; test it by varying $k$ and checking
whether the assignment is stable, and by looking at the projection before you
believe the labels.

## Computing it

Descriptors go in as `DescriptorRow` objects, one per sample:

```python
import numpy as np
from molpy.compute import DescriptorRow, KMeans, Pca

rng = np.random.default_rng(0)
group_a = rng.normal(0.0, 1.0, size=(60, 5))
group_b = rng.normal(3.0, 1.0, size=(60, 5))
table = np.vstack([group_a, group_b])

table = (table - table.mean(axis=0)) / table.std(axis=0)   # standardize first
projected = Pca()([DescriptorRow(row) for row in table])

coords = np.asarray(projected.coords)
variance = np.asarray(projected.variance)
print(coords.shape, np.round(variance, 2).tolist())
# -> (120, 2) [3.81, 0.35]
```

Note the shape. `Pca` here is specifically a **two-component** PCA — it returns
the first two axes and nothing else, so there is no "keep the first $k$"
decision to make and no way to inspect components 3 and beyond. If you need the
full spectrum of explained variance to decide how many components matter, this
is not the tool.

`variance` says how much each of the two earned: 3.81 against 0.35, so the two
groups differ along essentially one direction — correct, because that is how
they were built.

`KMeans` consumes the PCA result directly:

```python
labels = np.asarray(KMeans(k=2, max_iter=100, seed=0)(projected).labels)
truth = np.array([0] * 60 + [1] * 60)
agreement = max((labels == truth).mean(), (labels != truth).mean())
print(round(float(agreement), 3))               # -> 1.0
```

The `max` over both comparisons is not a trick: cluster labels are arbitrary
names, so "all 0s and 1s swapped" is the same clustering. Any time you score
k-means against known classes you have to allow for that permutation.

`seed` makes the initialization reproducible. Change it and re-run before
trusting a marginal result — k-means finds a local optimum, not the global one.

## When it goes wrong

**PC1 is just your largest-magnitude column.**
You did not standardize.

**The projection looks like one blob but k-means reports clean clusters.**
Believe the projection. This is k-means doing what it always does.

**Component signs flip between runs.**
Eigenvectors are defined up to sign. A flipped PC1 is the same decomposition;
do not interpret the sign.

**Adding a descriptor changes the answer completely.**
Your descriptors are correlated and you have added another copy of one that was
already there, which up-weights that direction. Prefer a small set of
independent descriptors.

**PC1 and PC2 have comparable variance.**
There may be no low-dimensional structure at all — a real result worth
reporting rather than plotting around. Because only two components are
returned, you cannot check how much is left in components 3+; do that with a
general-purpose PCA if it matters.

## Check yourself

- Run PCA on isotropic random data. The two variances should come out
  comparable; if one dominates, you have a bug or an unstandardized column.
- Cluster that same structureless data with $k=3$. You will get three neat
  clusters. Sit with that.
- Duplicate one descriptor column and re-run. Watch PC1 rotate toward it.

## References

- I. T. Jolliffe, *Principal Component Analysis*, 2nd ed., Springer (2002).
- S. Lloyd, *IEEE Trans. Inf. Theory* **28**, 129 (1982) — the k-means
  algorithm.
- W. Lechner, C. Dellago, *J. Chem. Phys.* **129**, 114707 (2008) — order
  descriptors of the kind fed in here.

## See also

- [Order](order.md) · [Shape](shape.md) — where the descriptors come from
- [Cluster](cluster.md) — clustering in *space* rather than in descriptor space
- [API reference](../api/compute.md)
