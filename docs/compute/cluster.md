# Clustering

Micelles, droplets, aggregates, percolating gel networks — all of them are the
same computational question: which particles are connected to which?

`Cluster` answers it by building a graph. Every pair inside the
[neighbour cutoff](neighborlist.md) becomes an edge, and each connected
component of that graph is a cluster. The algorithm is standard and not the
interesting part.

**The cutoff is the interesting part**, because the cutoff *is* your definition
of an aggregate. Nothing in the physics tells you where one droplet ends and
another begins; you decide, by choosing a number, and the answer follows from
that choice far more strongly than most people expect.

## How strongly: a 0.15 Å window

Here is the same 500-atom argon liquid, clustered at a range of cutoffs.

<figure id="fig-percolation" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
config:
  legend:
    orient: bottom
    direction: horizontal
    title: null
data: {$file: data/cluster/argon_percolation.json}
mark: {type: line, strokeWidth: 2.4, interpolate: monotone}
encoding:
  x:
    field: cutoff
    type: quantitative
    title: "cutoff (Å)"
  y:
    field: value
    type: quantitative
    title: null
  color:
    field: series
    type: nominal
    title: null
```

</div>

**Figure 1.** Cluster count and largest-cluster size against the neighbour
cutoff, for liquid argon at 85 K. Nothing about the configuration changes across
this plot — only the definition of "connected".
</figure>

Read the numbers, all with `min_cluster_size=2` so that lone atoms are not
counted as clusters:

| cutoff (Å) | clusters (size ≥ 2) | largest cluster |
|---|---|---|
| 3.3 | 11 | 0.4 % (2 atoms) |
| 3.5 | 89 | 2.9 % |
| 3.6 | 49 | 34 % |
| 3.65 | 19 | 79 % |
| 3.7 | 6 | 93 % |
| 3.8 | 2 | 98.8 % |
| 4.0 | 1 | 100 % |

The cluster count rises and then falls, which is not a mistake: at 3.3 Å almost
nothing is within range, so there are a handful of isolated pairs; by 3.5 Å most
atoms have a partner and there are 89 small fragments; past that the fragments
start merging into each other, so the count collapses while the largest cluster
grows. Nearly all the atoms are in no cluster at all at the small cutoffs, which
is why 11 clusters and "largest = 2 atoms" are consistent.

Between 3.55 and 3.70 Å the system goes from dozens of small aggregates to one
network containing essentially every atom. That is a **percolation transition**,
and it happens inside a 0.15 Å window. For scale, the first peak of
[$g(r)$](rdf.md) is centred at 3.68 Å and is roughly 1 Å wide at half height —
so the entire transition fits inside the leading edge of the first coordination
shell.

So a colleague who used 3.6 Å and a colleague who used 3.7 Å on the same
trajectory would publish qualitatively different conclusions, and neither would
be wrong about the arithmetic. This is why the cutoff must be reported with any
cluster result, and why it should be justified physically rather than picked.

The defensible choice for a *contact* definition is the **first minimum of
$g(r)$** — 5.4 Å for this argon, deep in the "everything is one cluster" regime,
which is the correct answer: a dense liquid *is* one connected blob at contact
distance. If you want droplets, you need a system that actually has droplets,
not a smaller cutoff on a homogeneous one.

## Computing it

```python
import numpy as np
import molpy as mp
from molpy.compute import NeighborList, Cluster

rng = np.random.default_rng(0)
# Three well-separated blobs of 40 points each.
centres = np.array([[5.0, 5.0, 5.0], [25.0, 5.0, 5.0], [5.0, 25.0, 5.0]])
xyz = np.concatenate([c + rng.normal(0.0, 0.8, size=(40, 3)) for c in centres])

frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box.cubic(40.0)

nlist = NeighborList(cutoff=2.0)(frame)
result, = Cluster(min_cluster_size=5)([frame], [nlist])
print(result.num_clusters)                      # -> 3
print(sorted(np.asarray(result.cluster_sizes).tolist()))    # -> [40, 40, 40]
```

Three blobs in, three clusters out. The result also carries `cluster_idx`, the
per-particle cluster label you need in order to colour a rendering or select one
aggregate:

```python
labels = np.asarray(result.cluster_idx)
print(labels.shape, len(set(labels.tolist())))  # -> (120,) 3
```

`min_cluster_size` discards components smaller than the threshold. Use it to
drop monomers, but be aware it changes `num_clusters` — it is a reporting
filter, not part of the physics.

### Per-cluster properties in one call

`ClusterProperties` reduces every cluster at once rather than making you loop:

```python
from molpy.compute import ClusterProperties

props, = ClusterProperties()([frame], [result])
print(sorted(props))
# -> ['centers', 'centers_of_mass', 'cluster_masses', 'gyration_tensors',
#     'radii_of_gyration', 'sizes']
print(np.round(np.asarray(props["radii_of_gyration"]), 2).tolist())
# -> [1.29, 1.45, 1.39]
```

`ClusterProperties` takes no mass argument, so its `radii_of_gyration` and
`gyration_tensors` are **geometric** — every particle counted with unit mass,
and `cluster_masses` simply the particle count. When you need mass weighting,
use the explicit `CenterOfMass` / `RadiusOfGyration` route on
[Shape](shape.md), which makes the convention visible in the call.

Each blob was drawn from a Gaussian of width 0.8 Å per direction, so its radius
of gyration should be close to $\sqrt{3}\times 0.8 = 1.39$ Å. The three come out
at 1.29, 1.45 and 1.39, averaging 1.38 — a case where you can predict the output
before running it, and the spread tells you the sampling noise on 40 points. See [Shape](shape.md) for
what the gyration tensor tells you beyond its trace.

## When it goes wrong

**Everything is one cluster.**
Usually correct for a dense liquid at a contact cutoff. If you expected
droplets, either the cutoff is too generous or the system is genuinely
homogeneous. Check the largest-cluster fraction against cutoff, as in Figure 1,
before assuming a bug.

**The cluster count changes wildly between frames.**
You are sitting on the percolation transition. Move the cutoff away from it, or
report the transition itself rather than a number from inside it.

**A single molecule is reported as two clusters.**
It straddles a periodic boundary and you unwrapped it wrongly — or not at all.
Cluster *identification* is fine under minimum image, so `num_clusters` is
right; it is the per-cluster *shape* that breaks. [Shape](shape.md) gives the
`unwrap_cluster` recipe — fold each cluster about one of its own atoms before
measuring anything geometric.

**`num_clusters` disagrees with the number of components you expected.**
`min_cluster_size` is filtering. Set it to 1 to see everything.

## Check yourself

- Build well-separated blobs, as above, and confirm you get exactly as many
  clusters as blobs. Then move two blobs together and watch them merge.
- Scan the cutoff on your own system and plot the largest-cluster fraction. If
  your chosen cutoff sits on the steep part, no conclusion drawn from it is
  robust.
- Compare your cutoff with the first minimum of $g(r)$. If they differ, be able
  to say why.

## References

- V. Ramasubramani et al., *Comput. Phys. Commun.* **254**, 107275 (2020) — the
  freud `cluster` module this mirrors.
- D. Stauffer, A. Aharony, *Introduction to Percolation Theory*, 2nd ed. (1994)
  — why the transition in Figure 1 is sharp.

## See also

- [NeighborList](neighborlist.md) — where the cutoff lives
- [RDF](rdf.md) — where a defensible cutoff comes from
- [Shape](shape.md) — what to measure once you have clusters
- [API reference](../api/compute.md)
