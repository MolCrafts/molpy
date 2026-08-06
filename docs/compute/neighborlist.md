# Neighbor list

Almost every structural question is really a question about neighbours. Which
atoms are in the first shell? Which molecules touch? Which particles belong to
the same cluster? All of them start by finding, for each atom, the other atoms
within some distance $r_c$.

Doing that from scratch means comparing every atom with every other atom: for
$N = 10^5$ atoms that is $5\times 10^9$ distance evaluations, and you would pay
it again for each analysis. So MolPy separates the search from the analysis. A
`NeighborList` finds the pairs once; [RDF](rdf.md), [LocalDensity](density.md),
[Steinhardt](order.md), [Cluster](cluster.md), and [PMFTXY](pmft.md) all consume
the same object.

## What a pair means under periodic boundaries

A simulation box has no walls. An atom near the right face is a close neighbour
of an atom near the left face, because the box repeats forever in all
directions. The **minimum-image convention** handles this: the distance between
two atoms is the distance to the *nearest periodic copy* of the second one.

That convention has a hard limit built into it. In a cubic box of edge $L$, once
you look further than $L/2$ you can reach two different images of the same atom,
and "the nearest copy" stops being a well-defined single partner. So:

$$
r_c \le \frac{L}{2} \quad\text{(or half the shortest box edge)}.
$$

Above that, results are not merely inaccurate — they are counting the same atom
twice. The 28.9 Å argon box used throughout these pages allows $r_c \le 14.4$ Å.

## The cost of a larger cutoff

The work is set by how many atoms fit in a sphere of radius $r_c$. If atoms were
spread at random with number density $\rho$, each atom would have

$$
\frac{\text{neighbours}}{\text{atom}} = \frac{4}{3}\pi\rho\, r_c^{3}
$$

inside that sphere. The cubic dependence is the thing to remember: **doubling
the cutoff costs eight times as much**.

Be careful which of two quantities you are counting, because they differ by a
factor of two. The formula above counts **neighbours** — for each atom, how many
others are nearby. The list stores **pairs**, and it stores each pair once, so

$$
\frac{\text{neighbours}}{\text{atom}} = \frac{2\,n_\text{pairs}}{N}.
$$

Mixing these up is the most common way to get a coordination number that is
half or double the right answer.

<figure id="fig-nlist-cost" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
config:
  legend:
    orient: bottom
    direction: horizontal
    title: null
data: {$file: data/neighborlist/pair_scaling.json}
mark: {type: line, strokeWidth: 2.4, interpolate: monotone}
encoding:
  x:
    field: cutoff
    type: quantitative
    title: "cutoff r_c (Å)"
  y:
    field: neighbours
    type: quantitative
    title: "neighbours / atom"
  color:
    field: series
    type: nominal
    title: null
```

</div>

**Figure 1.** Neighbours per atom as a function of cutoff in liquid argon
(500 atoms, $\rho = 0.0207$ Å⁻³), against the ideal-gas estimate
$\tfrac{4}{3}\pi\rho r_c^3$. The measured curve is not a smooth approach to the
estimate — it tracks the coordination shells.
</figure>

The gap between the two curves is the structure of the liquid, and it is worth
reading rather than skipping:

| $r_c$ (Å) | measured | ideal gas | ratio |
|---|---|---|---|
| 3.0 | 0.00 | 2.34 | 0 |
| 4.5 | 8.96 | 7.91 | 1.13 |
| 6.0 | 16.36 | 18.74 | 0.87 |
| 8.5 | 52.88 | 53.28 | 0.99 |
| 14.0 | 237.66 | 238.08 | 1.00 |

Below 3.2 Å the list finds **nothing at all**: no two argon atoms are ever that
close, which is the excluded core of [$g(r)$](rdf.md) seen from a different
angle. At 4.5 Å the count runs 13 % *above* random because the first
coordination shell is packed more densely than chance. At 6 Å, just past the
first minimum, it falls 13 % *below*. By 8.5 Å the shells have averaged out and
the ideal-gas estimate is good to a percent.

There is a useful cross-check hiding in that table. At $r_c = 5$ Å the list
gives 11.10 neighbours per atom; integrating $g(r)$ on the [RDF](rdf.md) page
gives a coordination number $n(5.0) = 11.06$. Two different calculations,
agreeing to 0.4 %.

For planning work: 11 neighbours per atom at 5 Å, 53 at 8.5 Å, 238 at 14 Å.
Building a 14 Å list to histogram $g(r)$ out to 6 Å does about twenty times the
necessary work.

Choosing the cutoff is therefore a physical decision, not a safety margin:

| Consumer | Sensible cutoff |
|---|---|
| [`RDF(r_max=…)`](rdf.md) | exactly `r_max` — no more |
| [`Cluster`](cluster.md), [`Steinhardt`](order.md) | first minimum of $g(r)$ (5.4 Å for argon) |
| [`LocalDensity`](density.md) | the length scale you want to smooth over |
| [`PMFTXY`](pmft.md) | far enough to cover the free-energy well of interest |

## Building one

```python
import numpy as np
import molpy as mp
from molpy.compute import NeighborList

xyz = np.array([[0.0, 0, 0], [2.0, 0, 0], [0, 2.0, 0], [8.0, 8, 8]])
frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box.cubic(20.0)

nlist = NeighborList(cutoff=3.0)(frame)
print(nlist.n_pairs)   # -> 3
```

Four atoms; the first three are mutually within 3 Å and the fourth is far away,
so there are three pairs. Note that a pair is stored **once**, not twice:

```python
print(nlist.pairs().tolist())               # -> [[0, 1], [0, 2], [1, 2]]
print(np.round(nlist.distances, 3).tolist())  # -> [2.0, 2.0, 2.828]
```

`pairs()` is a **method**, not an attribute — it materializes an `(n_pairs, 2)`
array each time you call it. The same two columns are available without that
copy as `query_point_indices` (column 0) and `point_indices` (column 1). Those
names come from the general case of querying one set of points against a
different set; the list built here is a *self*-query, where both sets are the
atoms of this frame, so both arrays are simply atom indices. `dist_sq` gives
squared distances if you want to skip the square root.

Because each pair appears once, the mean number of neighbours per atom is
`2 * n_pairs / n_atoms`. That identity is the quickest sanity check you can run:

```python
n_atoms = frame["atoms"].nrows
print(2 * nlist.n_pairs / n_atoms)     # -> 1.5
```

Four atoms, three of them mutually close, gives $2\times3/4 = 1.5$ neighbours
per atom on average. For liquid argon with a 5.4 Å cutoff the same expression
gives about 13 — the coordination number [RDF](rdf.md) obtains by integrating
$g(r)$, reached without any histogram at all.

## When it goes wrong

**`ValueError: frame.box is required for spatial neighbor search`.**
The frame has no periodic box, or the box is free. Neighbour search needs to
know how space wraps. Set `frame.box = mp.Box.cubic(L)`.

**Histograms are truncated at some radius.**
The cutoff is smaller than the consumer's `r_max`. `RDF` and friends can only
see pairs the list contains; they do not go back for more.

**Coordination numbers come out roughly double what you expect.**
You are treating the list as containing both $(i,j)$ and $(j,i)$. It does not.

**Results change after you move atoms.**
The list is a snapshot. Coordinates that change invalidate it; build a new list
per frame. This is why the trajectory-averaging APIs take parallel lists of
frames and neighbor lists.

**Memory blows up on a big system.**
Pairs scale as $N\rho r_c^3$. Shrink the cutoff before anything else.

## Check yourself

- Put two atoms 1 Å apart across a periodic face of a 20 Å box (say at $x = 0.5$
  and $x = 19.5$). Confirm the list finds them at 1 Å, not 19 Å.
- Take any frame, build lists at $r_c$ and $2r_c$, and compare `n_pairs`. The
  ratio should be near 8 once $r_c$ is past the first few shells.
- Compute `2 * n_pairs / n_atoms` at the first minimum of $g(r)$ and check it
  against the coordination number from [RDF](rdf.md).

## References

- M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.,
  Oxford (2017) — minimum image, cell lists, and Verlet lists.
- V. Ramasubramani et al., *Comput. Phys. Commun.* **254**, 107275 (2020) — the
  freud neighbor-query model this API mirrors.

## See also

- [RDF](rdf.md) · [Density](density.md) · [PMFT](pmft.md) · [Order](order.md)
- [Compute overview](index.md) · [API reference](../api/compute.md)
