# Decomposition

By the time you have run [Order](order.md), [Shape](shape.md), and
[Density](density.md) over a trajectory, every atom carries half a dozen
numbers. That is too many to plot and too few to be a model. You want the two or
three combinations that actually distinguish the configurations, and then a
label saying which state each configuration is in.

`Pca` finds the combinations. `Kmeans` assigns the labels.

## PCA finds the directions that vary

Principal component analysis takes a table with one row per sample and one
column per descriptor, and rotates the coordinate axes so that the first new
axis lies along the direction of greatest variance, the second along the
greatest remaining variance, and so on. Keeping only the leading axes is what
turns the rotation into a reduction, and how many to keep is the one thing you
configure: `Pca(n_components)`.

The reason it works as *analysis* rather than mere compression: descriptors are
usually redundant. $q_4$, $q_6$, coordination number, and local density all rise
and fall together when a region crystallizes. PCA notices that they are one
underlying variable wearing four hats, and hands you that variable.

**The columns must be standardized.** This is not a refinement, it is a
prerequisite. PCA maximizes variance, and variance has units. A coordination
number of order 12 and a $q_6$ of order 0.5 differ by a factor of 24, so on raw
columns the first principal component is "coordination number" no matter what
the physics is. Subtract the mean and divide by the standard deviation of each
column, and PCA is comparing shapes of distributions instead of choices of unit.
`Pca` does exactly this before it decomposes anything, so the axes it reports
are those of the correlation matrix, and the variances of all components add
up to the number of columns. A column with no variance at all carries no
direction and is refused rather than divided by zero.

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
data: {$file: series/decomposition/phase_map.json}
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
2.51 against 0.43 for PC2 — so a single coordinate holds most of the
information. That is the useful output: you started with three descriptors and
found that one combination of them is the order parameter.

Running k-means with $k=2$ on the same data and comparing its labels with the
truth gives **100 % agreement**. The algorithm recovered the phases from
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

Both computes consume a **feature matrix**: one row per sample, one column per
descriptor. A plain 2-D NumPy array is one. So is the result of a descriptor
compute — `Steinhardt`'s per-atom $q_\ell$, `LocalDescriptors` — and so is a
`Pca` result, which is why the three chain without any conversion in between.
How many components to keep is configuration and goes into the constructor;
the table is data and goes into `compute`.

Start with a table whose answer is known: two groups of 60 samples in five
dimensions, offset along every axis at once.

```python
import numpy as np
from molpy.compute import Kmeans, Pca

rng = np.random.default_rng(0)
group_a = rng.normal(0.0, 1.0, size=(60, 5))
group_b = rng.normal(3.0, 1.0, size=(60, 5))
table = np.vstack([group_a, group_b])

projected = Pca(n_components=2).compute(features=table)
print(projected.coords.shape, np.round(projected.variance, 2).tolist())
# -> (120, 2) [3.81, 0.35]
```

`variance` says how much each component earned: 3.81 against 0.35, so the two
groups differ along essentially one direction — correct, because that is how
they were built. To see the whole spectrum before deciding how many components
matter, ask for all of them; they add up to the five columns:

```python
spectrum = Pca(n_components=5).compute(features=table).variance
print(np.round(spectrum, 2).tolist(), round(float(spectrum.sum()), 2))
# -> [3.81, 0.35, 0.31, 0.27, 0.25] 5.0
```

`Kmeans` consumes the PCA result directly:

```python
labels = Kmeans(k=2, max_iter=100, seed=0).compute(features=projected).labels
truth = np.array([0] * 60 + [1] * 60)
agreement = max((labels == truth).mean(), (labels != truth).mean())
print(round(float(agreement), 3))               # -> 1.0
```

The `max` over both comparisons is not a trick: cluster labels are arbitrary
names, so "all 0s and 1s swapped" is the same clustering. Any time you score
k-means against known classes you have to allow for that permutation.

`seed` makes the initialization reproducible. Change it and re-run before
trusting a marginal result — k-means finds a local optimum, not the global one.

### From atoms to phases in three computes

The same chain runs on per-atom order descriptors. Melt half of an FCC crystal
by displacing its atoms, compute $q_4$ and $q_6$ for every atom, and hand the
Steinhardt result to `Pca` and the projection to `Kmeans`:

```python
import molpy as mp
from molpy.compute import Steinhardt

a = 5.26
basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
xyz = np.array(
    [(np.array([i, j, k]) + b) * a
     for i in range(6) for j in range(6) for k in range(6) for b in basis]
)
melted = xyz[:, 0] < 3 * a
noise = np.random.default_rng(0)
xyz[melted] += noise.normal(0.0, 0.7, size=(int(melted.sum()), 3))

nl = mp.core.NeighborList(4.5)
nl.build(xyz, mp.Box(mp.core.Cell.lengths([6 * a] * 3)))
order = Steinhardt(l=[4, 6]).compute(neighbors=nl.neighbors())
phases = Pca(n_components=2).compute(features=order)
labels = Kmeans(k=2, seed=0).compute(features=phases).labels

truth = melted.astype(int)
agreement = max((labels == truth).mean(), (labels != truth).mean())
print(round(float(agreement), 3))               # -> 0.925
```

92.5 % of 864 atoms land in their true phase, from two descriptors and no
labels. The misassigned ones sit mostly at the interface, where an atom's
neighbourhood is genuinely half crystal and half not.

The k-means labels are themselves a **partition** — one label per atom — so
they can be handed on again, as the `partition` of `CenterOfMass` or
`GyrationTensor` (see [Shape](shape.md)), to locate and measure each phase.

## When it goes wrong

**k-means splits along your largest-magnitude column.**
`Kmeans` measures plain Euclidean distance on the features it is given, and it
does not standardize them. Feed it the `Pca` result, which is standardized, or
standardize the columns yourself.

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
reporting rather than plotting around. Ask for every component and check how
much is left beyond the first two before you plot only those.

**`ValueError` about a column out of range or a constant column.**
You asked for more components than there are columns, or a descriptor column
never varies — the $\ell = 0$ columns of `LocalDescriptors` are the usual
culprit. Drop the column, or cluster the raw descriptors with `Kmeans`.

## Check yourself

- Run PCA on isotropic random data. The two variances should come out
  comparable; if one dominates, you have a bug.
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
