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

For argon, $g = 2.91$ at the first peak gives $w = -1.07\,k_BT$, and the
minimum at $g = 0.61$ gives $w = +0.50\,k_BT$ — the desolvation barrier a pair
must cross to move from the first shell to the second. Those two numbers are the
free-energy landscape of the liquid, obtained from nothing but a histogram of
distances.

`PmftXy` does the same thing in two dimensions. Instead of binning neighbours by
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
orientation. Hand `compute` one orientation per particle — for `PmftXy` a
planar angle in radians — and every neighbour is rotated into that particle's
body frame before binning. The angle is data, read from the snapshot being
analysed, so it follows the particle as it tumbles: typically `atan2` of the
particle's own `head - tail` axis in that frame.

With orientations, a lobe at the "front" of the map means neighbours genuinely
prefer the front. Without them, the compute still runs, but in the lab frame, and
the result is a rotationally averaged smear that tells you nothing $g(r)$ did not
already.

That is the single most important thing to get right on this page: **`PmftXy`
without orientations is not a PMFT.**

One further property to expect: `PmftXy` bins $x$ and $y$ but integrates over
$z$. A three-dimensional coordination shell therefore appears as a fairly shallow
two-dimensional ring, because the bins near the origin also collect distant pairs
that happen to lie almost along $z$. Contrast in the map is weaker than in
$w(r) = -\ln g(r)$, and that is geometry, not a bug.

!!! note "No figure on this page yet — TODO"
    The figure worth showing is the lobed, orientation-resolved map, and that
    needs an anisotropic system — rods, discs, or a molecular liquid — carrying
    per-particle orientations. The reference trajectory behind the other compute
    pages is monatomic argon, whose map is a featureless ring with a measured
    contrast of only about $0.3\,k_BT$. Showing that would suggest PMFT maps are
    uninformative, which is the opposite of the truth, so this page has no
    figure until an anisotropic trajectory exists under `molpy/scripts/docs_series/`.

## Computing it

```python
import numpy as np
import molpy as mp

rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 20.0, size=(400, 3))
frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box(mp.core.Cell.lengths([20.0] * 3))
```

The constructor holds the window and its resolution — $\pm x_\max \times \pm
y_\max$ in `n_x` × `n_y` bins. `compute` takes one snapshot's neighbour table
(it needs the `disp` column, which a `NeighborList` table carries, and reads the
box from the table) and, optionally, the orientations.

Pick the neighbour cutoff from the *corner* of the map, not its edge. A
$\pm 6$ Å window reaches $6\sqrt{2} = 8.49$ Å at its corners, so anything
shorter leaves those corners permanently unvisited:

```python
from molpy.compute import PmftXy

analyzer = PmftXy(x_max=6.0, y_max=6.0, n_x=40, n_y=40)
nl = mp.core.NeighborList(8.5)
nl.build(frame.coords, frame.box)
result = analyzer.compute(pairs=nl.neighbors())

print(result.raw_counts.shape, result.pmf.shape)   # -> (40, 40) (40, 40)
print(int(result.raw_counts.sum()))                # -> 39708
```

The result holds three `(n_x, n_y)` arrays and the bin `edges` of each axis.
Keep the arrays apart in your head: `raw_counts` is the raw histogram,
`density` is that histogram over each bin's area, and `pmf` is the free energy
$-\ln(\rho/\rho_\text{ref})$, with $\rho_\text{ref}$ the density of an
uncorrelated system.

Bins that no neighbour ever visited have no free energy, since $-\ln 0$ is
infinite, and `pmf` reports them as `+inf`. With a correctly sized cutoff this
frame has none:

```python
print(int((~np.isfinite(result.pmf)).sum()))   # -> 0
```

Shorten the cutoff below the corner distance and they appear, in exactly the
places geometry predicts:

```python
nl = mp.core.NeighborList(8.0)
nl.build(frame.coords, frame.box)
clipped = analyzer.compute(pairs=nl.neighbors())
print(int((~np.isfinite(clipped.pmf)).sum()))   # -> 8
```

Eight bins, all in the corners where $\sqrt{x^2+y^2} > 8$ Å. If you see empty
bins, check this cause before blaming statistics — a clipped map is a geometry
error, and no amount of extra sampling will fill it.

Genuine sampling holes look different: scattered through the map rather than
banded at the corners, and they retreat as you add frames or coarsen the grid. A
2-D histogram is hungry — doubling the resolution in each direction quarters the
counts per bin.

**Averaging over frames.** `PmftXy` analyses one snapshot; a trajectory is a
loop over snapshots. Sum the raw counts and take the logarithm once, at the
end:

```python
nl = mp.core.NeighborList(8.5)
frames = [frame, frame]                    # in practice, your trajectory
total = np.zeros((40, 40))
for f in frames:
    nl.build(f.coords, f.box)
    total += analyzer.compute(pairs=nl.neighbors()).raw_counts

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
`pmf` array of each result uses the compute's own reference density, so the two
differ by an additive constant. That is harmless for reading barriers and well
depths, which are differences, but do not mix the two on one colour scale.

Averaging per-frame `pmf` arrays instead is wrong: the mean of logarithms is not
the logarithm of the mean, and empty bins in individual frames contribute
infinities that bias the result.

## Orientations make it a PMFT

Everything above ran in the lab frame, which — as the section before last
insisted — is not yet a PMFT. Here is the part that makes it one.

`orientations` is one planar angle per particle, and "particle" means **row of
the neighbour table's point set**: the same array is indexed by both $i$ and
$j$ of every pair, so it needs **one entry per atom in the frame**, not one per
rigid body. Atoms belonging to the same body simply repeat its angle.

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
rods.box = mp.Box(mp.core.Cell.lengths([20.0] * 3))
```

Atoms `0 … 199` are the heads and `200 … 399` the matching tails. Reading the
orientations is your step: for every atom, take the axis of the rod it belongs
to from this frame's positions and convert it to an angle.

```python
head = np.concatenate([np.arange(n_rods), np.arange(n_rods)])
tail = np.concatenate([np.arange(n_rods, 2 * n_rods), np.arange(n_rods, 2 * n_rods)])
positions = rods.coords
director = positions[head] - positions[tail]
theta = np.arctan2(director[:, 1], director[:, 0])

nl = mp.core.NeighborList(8.5)
nl.build(rods.coords, rods.box)
table = nl.neighbors()
body = analyzer.compute(pairs=table, orientations=theta)
print(theta.shape, rods["atoms"].n_rows)   # -> (400,) 400
```

Give one angle per rod instead and the compute refuses with
`ValueError: orientations vs query points dimension mismatch: expected 400,
got 200`. If you see it, count angles against atoms.

The map is now genuinely different from the lab-frame one, because each
neighbour has been rotated into its reference rod's frame before binning:

```python
lab = analyzer.compute(pairs=table)
print(np.allclose(body.raw_counts, lab.raw_counts))   # -> False
```

For randomly oriented rods the difference is modest, since there is no
correlation between orientation and position to reveal. In a system that *does*
have one — a liquid crystal, a stack of discs, water around a solute — this is
the difference between seeing lobes and seeing a ring.

## Three more coordinate systems share the engine

`PmftXy` is one of four. All bin the same pair bonds; they differ only in which
coordinates a bond has in the reference frame, and so in what orientations they
need:

| Compute | Bins a bond as | Orientations |
|---|---|---|
| `PmftXy(x_max, y_max, n_x, n_y)` | $(x, y)$ in the query particle's frame (2-D) | planar angles, or none (lab frame) |
| `PmftXyt(x_max, y_max, n_x, n_y, n_t)` | $(x, y)$ plus the relative orientation $\theta$ | planar angles, required |
| `PmftR12(r_max, n_r, n_t1, n_t2)` | distance and the bond angle in each particle's frame | planar angles, required |
| `PmftXyz(x_max, y_max, z_max, n_x, n_y, n_z)` | $(x, y, z)$ in the query particle's frame (3-D) | `(w, x, y, z)` quaternions, or none |

The rod system above already has everything `PmftXyt` needs:

```python
from molpy.compute import PmftXyt

torque = PmftXyt(x_max=6.0, y_max=6.0, n_x=20, n_y=20, n_t=12).compute(
    pairs=table, orientations=theta
)
print(torque.raw_counts.shape, len(torque.edges))   # -> (20, 20, 12) 3
```

## When it goes wrong

**The map is a smooth ring with no angular structure.**
Either the particles really are isotropic, or — far more likely — no
`orientations` were passed and you are looking at the lab frame.

**The lobes point the wrong way, or rotate between frames.**
The `(head, tail)` endpoints you built the angles from are swapped or
inconsistent across particles. Reverse the axis and every map flips; mix the
two conventions and the lobes blur into each other.

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
- Run `PmftXy` on random points. The map should be flat to within noise, since
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
