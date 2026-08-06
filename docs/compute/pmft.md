# PMFT

[$g(r)$](rdf.md) tells you that neighbours prefer to sit 3.7 Å away. It cannot
tell you *where* — whether they gather at the ends of a rod-shaped molecule or
along its flat side, whether two discs prefer to stack face-to-face or meet
edge-to-edge. Averaging over all directions destroys exactly the information
that matters for anisotropic particles.

The **potential of mean force and torque** keeps it.

## Structure is already a free energy

Start from a fact that is easy to state and easy to under-appreciate. In
equilibrium, the probability of finding a configuration is a Boltzmann factor.
Since $g(r)$ *is* the relative probability of finding a neighbour at $r$,

$$
w(r) = -k_B T \ln g(r).
$$

That is not an approximation or a fit. It is the definition of a free energy
read backwards. Wherever $g > 1$, neighbours accumulate, and $w < 0$: a
favourable separation. Wherever $g < 1$, $w > 0$: a barrier.

For argon, $g = 2.95$ at the first peak gives $w = -1.08\,k_BT$, and the
minimum at $g = 0.60$ gives $w = +0.50\,k_BT$ — the desolvation barrier a pair
must cross to move from the first shell to the second. Those two numbers are the
free-energy landscape of the liquid, obtained from nothing but a histogram of
distances.

`PMFTXY` does the same thing in two dimensions. Instead of binning neighbours by
distance alone, it bins them by their $(x, y)$ position relative to the
reference particle, and takes $-\ln$ of the result:

$$
w(x, y) = -k_B T \ln \frac{\rho(x,y)}{\langle\rho\rangle}.
$$

## The body frame is the point

Binning in the *lab* frame is not yet useful. If particle A is lying along $x$
and particle B along $y$, "a neighbour at $(+4, 0)$ Å" means the end of one and
the side of the other; averaging them together erases the distinction all over
again.

So each reference particle gets its own coordinate frame, aligned with its own
orientation. Supply an `orientations` topology block — one `(head, tail)` atom
pair per particle — and every neighbour is rotated into that particle's body
frame before binning. The per-particle angle is `atan2` of the `head - tail`
axis, recomputed from each frame's positions, so it follows the particle as it
tumbles.

With that block, a lobe at the "front" of the map means neighbours genuinely
prefer the front. Without it, the compute still runs, but in the lab frame, and
the result is a rotationally averaged smear that tells you nothing $g(r)$ did not
already.

That is the single most important thing to get right on this page: **`PMFTXY`
without an `orientations` block is not a PMFT.**

One further property to expect: `PMFTXY` bins $x$ and $y$ but integrates over
$z$. A three-dimensional coordination shell therefore appears as a fairly shallow
two-dimensional ring, because the bins near the origin also collect distant pairs
that happen to lie almost along $z$. Contrast in the map is weaker than in
$w(r) = -\ln g(r)$, and that is geometry, not a bug.

!!! note "No figure on this page yet — TODO"
    The figure worth showing is the lobed, orientation-resolved map, and that
    needs an anisotropic system — rods, discs, or a molecular liquid — carrying
    an `orientations` block. The reference trajectory behind the other compute
    pages is monatomic argon, whose map is a featureless ring with a measured
    contrast of only about $0.3\,k_BT$. Showing that would suggest PMFT maps are
    uninformative, which is the opposite of the truth, so this page has no
    figure until an anisotropic trajectory exists under `scripts/docs_data/`.

## Computing it

```python
import numpy as np
import molpy as mp

rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 20.0, size=(400, 3))
frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box.cubic(20.0)
```

Pick the neighbour cutoff from the *corner* of the map, not its edge. A
$\pm 6$ Å window reaches $6\sqrt{2} = 8.49$ Å at its corners, so anything
shorter leaves those corners permanently unvisited:

```python
from molpy.compute import NeighborList, PMFTXY

analyzer = PMFTXY(x_max=6.0, y_max=6.0, n_x=40, n_y=40)
(counts, density, pmf), = analyzer([frame], [NeighborList(cutoff=8.5)(frame)])

print(counts.shape, pmf.shape)          # -> (40, 40) (40, 40)
print(int(counts.sum()))                # -> 39708
```

`PMFTXY` returns, **per frame**, a tuple of three `(n_x, n_y)` arrays. Keep them
apart in your head: `counts` is the raw histogram, `density` is that histogram
normalized, and `pmf` is the free energy $-\ln(\text{density})$.

Bins that no neighbour ever visited have no free energy, since $-\ln 0$ is
infinite. With a correctly sized cutoff this frame has none:

```python
print(int((~np.isfinite(pmf)).sum()))   # -> 0
```

Shorten the cutoff below the corner distance and they appear, in exactly the
places geometry predicts:

```python
short = NeighborList(cutoff=8.0)(frame)
(_, _, clipped), = analyzer([frame], [short])
print(int((~np.isfinite(clipped)).sum()))   # -> 8
```

Eight bins, all in the corners where $\sqrt{x^2+y^2} > 8$ Å. If you see empty
bins, check this cause before blaming statistics — a clipped map is a geometry
error, and no amount of extra sampling will fill it.

Genuine sampling holes look different: scattered through the map rather than
banded at the corners, and they retreat as you add frames or coarsen the grid. A
2-D histogram is hungry — doubling the resolution in each direction quarters the
counts per bin.

**Averaging over frames.** Sum the raw counts and take the logarithm once, at
the end:

```python
nlist = NeighborList(cutoff=8.5)(frame)
per_frame = analyzer([frame, frame], [nlist, nlist])
total = np.sum([raw for raw, _, _ in per_frame], axis=0)

occupied = total > 0
w = np.full(total.shape, np.nan)
w[occupied] = -np.log(total[occupied] / total[occupied].mean())
print(int(np.isnan(w).sum()))           # -> 0 bins left undefined
```

Masking with `occupied` rather than dividing straight through is not
fastidiousness: `-np.log(0)` is an infinity that will propagate into every
subsequent mean, contour level, and colour scale you compute.

Note that this hand-rolled `w` is normalized by the mean of the occupied bins,
which is a choice — it sets the zero of free energy to "an average bin". The
`pmf` array returned per frame uses the compute's own normalization, so the two
differ by an additive constant. That is harmless for reading barriers and well
depths, which are differences, but do not mix the two on one colour scale.

## Supplying orientations

Everything above ran in the lab frame, which — as the section before last
insisted — is not yet a PMFT. Here is the part that makes it one.

The body frame comes from an `orientations` block on the frame. It uses the same
schema as `bonds`: two integer columns, `atomi` and `atomj`, naming the head and
tail atoms of an axis. The director for that row is `pos[atomi] - pos[atomj]`,
and its 2-D angle is `atan2(dy, dx)`.

One rule catches everyone, and it is not what you would guess from "one pair per
particle": the block is indexed by **atom index**, so it needs **one row per
atom in the frame**, not one row per rigid body. Every atom that can appear in a
neighbour pair must have its own row saying which way its body is pointing.
Atoms belonging to the same body simply repeat the same `(head, tail)`.

Build a system of 200 two-atom rods, each with a random in-plane orientation:

```python
rng = np.random.default_rng(0)
n_rods = 200
centres = rng.uniform(0.0, 20.0, size=(n_rods, 3))
angle = rng.uniform(0.0, 2 * np.pi, n_rods)
axis = np.stack([np.cos(angle), np.sin(angle), np.zeros(n_rods)], axis=1)

xyz = np.concatenate([centres + 0.5 * axis, centres - 0.5 * axis])
rods = mp.Frame()
rods["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
rods.box = mp.Box.cubic(20.0)
```

Atoms `0 … 199` are the heads and `200 … 399` the matching tails. Now give
*every* atom an orientation row, pointing along its own rod:

```python
head = np.concatenate([np.arange(n_rods), np.arange(n_rods)])
tail = np.concatenate([np.arange(n_rods, 2 * n_rods), np.arange(n_rods, 2 * n_rods)])
rods["orientations"] = {"atomi": head, "atomj": tail}

(body_counts, _, _), = analyzer([rods], [NeighborList(cutoff=8.5)(rods)])
print(rods["orientations"].nrows, rods["atoms"].nrows)   # -> 400 400
```

Get that length wrong and the error is unhelpful — a
`ValueError: PMFTXY orientations length dimension mismatch: expected 207,
got 200`, where 207 is simply the first atom index the kernel reached that had
no row. If you see it, count rows against atoms.

The map is now genuinely different from the lab-frame one, because each
neighbour has been rotated into its reference rod's frame before binning:

```python
plain = mp.Frame()
plain["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
plain.box = mp.Box.cubic(20.0)
(lab_counts, _, _), = analyzer([plain], [NeighborList(cutoff=8.5)(plain)])

print(np.allclose(body_counts, lab_counts))   # -> False
```

For randomly oriented rods the difference is modest, since there is no
correlation between orientation and position to reveal. In a system that *does*
have one — a liquid crystal, a stack of discs, water around a solute — this is
the difference between seeing lobes and seeing a ring.

Averaging per-frame `pmf` arrays instead is wrong: the mean of logarithms is not
the logarithm of the mean, and empty bins in individual frames contribute
infinities that bias the result.

## When it goes wrong

**The map is a smooth ring with no angular structure.**
Either the particles really are isotropic, or — far more likely — there is no
`orientations` block and you are looking at the lab frame.

**The lobes point the wrong way, or rotate between frames.**
The `(head, tail)` endpoints are swapped or inconsistent across particles. The
body-frame angle comes from `head - tail`; reverse it and every map flips.

**The corners of the map are empty.**
Neighbour cutoff smaller than the diagonal of the window, $\sqrt{x_\max^2 +
y_\max^2}$.

**Most of the map is `inf` or `nan`.**
Empty bins. Either the grid is too fine for the amount of data, or the window
extends past where any neighbour is found. A 2-D histogram needs far more
samples than a 1-D one: doubling the resolution in each direction quarters the
counts per bin.

**The free energies look implausibly large.**
Check the normalization you divided by, and check you have not averaged
logarithms.

## Check yourself

- Compute $w(r) = -\ln g(r)$ from the [RDF](rdf.md) page's argon data by hand and
  confirm the first-shell minimum is about $-1.1\,k_BT$.
- Run `PMFTXY` on random points. The map should be flat to within noise, since
  uncorrelated particles have no free-energy landscape.
- Halve `n_x` and `n_y` and watch how many bins stop being empty. That trade
  between resolution and statistics is the whole art of 2-D histogramming.

## References

- G. van Anders, D. Klotsa, N. K. Ahmed, M. Engel, S. C. Glotzer, *ACS Nano*
  **8**, 931 (2014) — potential of mean force and torque for anisotropic
  particles. DOI: 10.1021/nn4057353
- V. Ramasubramani et al., *Comput. Phys. Commun.* **254**, 107275 (2020) — the
  freud PMFT implementation this mirrors.

## See also

- [RDF](rdf.md) — the 1-D version of the same idea
- [Spatial](spatial.md) — full 3-D body-fixed distribution
- [NeighborList](neighborlist.md) · [API reference](../api/compute.md)
