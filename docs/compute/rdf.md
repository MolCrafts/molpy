# Radial distribution function

You have a box of argon that has been running for 30 ps. Is it a liquid?

Nothing in the coordinate file answers that. A gas, a liquid, and a crystal all
look like a list of numbers. What separates them is not where any single atom
is, but how atoms sit *relative to each other* — and the radial distribution
function $g(r)$ is the standard way to measure that.

## Counting neighbours in a shell

Forget the formula for a moment and imagine doing the measurement by hand.

Stand on one atom. Draw a thin spherical shell around yourself at distance $r$,
with thickness $\mathrm{d}r$, and count how many other atoms fall inside it.
Move to the next atom and repeat. Average over every atom and every frame. Call
that average count $N_\text{shell}(r)$.

That number is not yet useful, because it grows with $r$ for a boring reason: a
shell far away is simply bigger. A shell at 10 Å has a hundred times the volume
of one at 1 Å, so it catches more atoms even if the atoms are sprinkled
completely at random.

So divide out the boring part. If the atoms ignored each other entirely — an
ideal gas at the same number density $\rho = N/V$ — that shell would hold
$\rho \cdot 4\pi r^2\,\mathrm{d}r$ atoms. The ratio of what you counted to what
random chance would give is $g(r)$:

$$
g(r) = \frac{N_\text{shell}(r)}{4\pi r^{2}\rho\,\mathrm{d}r}.
$$

Written as an ensemble average over pairs, the same quantity is

$$
g(r) = \frac{V}{N^{2}}
\Big\langle \sum_{i}\sum_{j\neq i}\delta(r - r_{ij}) \Big\rangle
\Big/ 4\pi r^{2}.
$$

Both lines say the same thing, and the first is the one to keep in your head.
Read $g(r)$ as a **local enrichment factor**:

| $g(r)$ | meaning at that distance |
|---|---|
| $=1$ | exactly as many neighbours as random chance — no correlation |
| $>1$ | atoms pile up here; a preferred separation |
| $<1$ | atoms avoid this separation |
| $=0$ | never happens |

The $4\pi r^2$ in the denominator is the single most misread part of the
definition. It is not physics; it is the price of using a spherical shell.

## Reading a real curve

Here is $g(r)$ for 500 argon atoms at 85 K and 1.374 g cm⁻³, averaged over 600
configurations from a 30 ps constant-energy trajectory.

<figure id="fig-rdf-argon" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
data: {$file: data/rdf/argon_gr.json}
mark: {type: line, strokeWidth: 2.4, interpolate: monotone}
encoding:
  x:
    field: r
    type: quantitative
    title: "r (Å)"
    scale: {domain: [0, 14]}
  y:
    field: g
    type: quantitative
    title: "g(r)"
    scale: {domain: [0, 3.2]}
```

</div>

**Figure 1.** $g(r)$ of liquid argon at 85 K. The excluded core, the first
coordination shell at 3.68 Å, and the decay to $g = 1$ are all visible.
</figure>

Walk across it from left to right.

**Below about 3 Å, $g(r)$ is flat zero.** No pair of argon atoms is ever found
closer than that. This is the repulsive wall of the interatomic potential, and
it is the reason liquids are nearly incompressible. Every atom carries an
exclusion zone that nothing else enters.

**A sharp peak at 3.68 Å, reaching $g = 2.95$.** Almost three times as many
neighbours sit at this distance as chance would put there. This is the first
coordination shell — the atoms in direct contact. Its position is not arbitrary.
Argon is modelled here as a Lennard-Jones fluid with collision diameter
$\sigma = 3.405$ Å, and that potential has its energy minimum at
$2^{1/6}\sigma = 3.82$ Å. The peak sits just inside the bottom of the pair
potential well: atoms settle where the energy is lowest, pushed slightly closer
by the pressure of the surrounding fluid.

**A dip to $g = 0.60$ at 5.43 Å.** Depleted, not empty. Atoms in the first shell
are in the way, so this separation is awkward to occupy.

**A broad second peak at 7.03 Å, only $g = 1.26$.** The second shell, roughly
twice the first-shell distance and already much weaker.

**Beyond about 9 Å, $g(r)$ wanders around 1.** Correlation has died out. An atom
here knows nothing about the one at the origin. The residual few-percent
wobble is sampling noise, not structure — worth remembering as the honest
precision of a 500-atom, 600-frame average.

That shape — a strong first shell, a weak second one, then nothing — *is* the
signature of a liquid. Compare the three states:

| System | $g(r)$ looks like |
|---|---|
| Dilute gas | $\approx 1$ everywhere except the excluded core |
| Liquid | 2–3 decaying shells, then flat 1 |
| Crystal | sharp peaks at lattice distances, out to the edge of the box |

## Turning the curve into a number: coordination

The most useful thing to extract from $g(r)$ is *how many* neighbours are in the
first shell. Undo the normalization: multiply back by the shell volume and
integrate.

$$
n(R) = 4\pi\rho\int_{0}^{R} r^{2} g(r)\,\mathrm{d}r
$$

$n(R)$ is the average number of atoms within radius $R$. Evaluate it at the
first minimum of $g(r)$ — the natural boundary of the first shell — and you get
the coordination number. There is no `coordination` field on the result object;
you build this integral yourself, and the recipe is
[four lines, below](#computing-it).

<figure id="fig-rdf-coordination" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
data: {$file: data/rdf/argon_coordination.json}
mark: {type: line, strokeWidth: 2.4, interpolate: monotone}
encoding:
  x:
    field: r
    type: quantitative
    title: "R (Å)"
    scale: {domain: [0, 8]}
  y:
    field: n
    type: quantitative
    title: "n(R)"
    scale: {domain: [0, 40]}
```

</div>

**Figure 2.** Running coordination number for the same trajectory. The plateau
between the first peak and the first minimum of $g(r)$ is the first
coordination shell.
</figure>

For this argon, $n$ at the first minimum is **12.9**. A close-packed crystal has
exactly 12 nearest neighbours. So each atom in the liquid is packed almost as
tightly as it would be in a solid — it has kept its neighbours and lost only the
long-range order. That one number is the physical content of the whole curve,
and it is why liquids are dense but flow.

The first minimum is also the cutoff you should reuse elsewhere: it is the
defensible definition of "in contact" for [Cluster](cluster.md),
[Order](order.md), and [Persist](persist.md).

## Computing it

`RDF` histograms distances that a [`NeighborList`](neighborlist.md) has already
found, so the two are always used together. Build the frame first:

```python
import numpy as np
import molpy as mp

a = 5.26  # argon FCC lattice constant, Å
basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
xyz = np.array(
    [(np.array([i, j, k]) + b) * a
     for i in range(4) for j in range(4) for k in range(4) for b in basis]
)
frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box.cubic(4 * a)
```

That is a perfect FCC crystal, which makes a good first test: you know the
answer in advance. Now histogram it.

```python
from molpy.compute import NeighborList, RDF

nlist = NeighborList(cutoff=8.0)(frame)
result = RDF(n_bins=160, r_max=8.0)([frame], [nlist])

r, g = result.bin_centers, result.rdf
peaks = r[g > 0.1]
print(np.round(peaks, 2))   # -> [3.72 5.28 6.43 7.43]
```

Check those against the lattice by hand: FCC neighbour distances are
$a/\sqrt{2} = 3.72$, $a = 5.26$, $a\sqrt{3/2} = 6.44$, and $a\sqrt{2} = 7.44$ Å.
The peaks land on them. A crystal gives sharp spikes at the lattice distances
and never decays to 1 — exactly the third row of the table above.

`RDF` takes **lists** of frames and neighbor lists, and averages over them, so a
trajectory is the same call with longer lists:

```python
frames = [frame, frame]                        # in practice, your trajectory
nlists = [NeighborList(cutoff=8.0)(f) for f in frames]
averaged = RDF(n_bins=160, r_max=8.0)(frames, nlists)
print(averaged.n_frames)                       # -> 2
```

The result object carries `bin_centers`, `bin_edges`, `rdf`, the raw per-bin
counts `n_r`, and the bookkeeping (`n_frames`, `n_points`, `volume`) needed to
normalize anything yourself.

### Coordination number, in four lines

Nothing on the result gives you $n(R)$ directly, so build the integral
$n(R) = 4\pi\rho\int_0^R r^2 g(r)\,\mathrm{d}r$ from the returned arrays. The
density you need is already there, as `n_points / volume`:

```python
r, g = result.bin_centers, result.rdf
rho = result.n_points / result.volume
shell = 4 * np.pi * rho * r**2 * g
n_of_r = np.cumsum(shell * (r[1] - r[0]))

print(round(float(np.interp(4.5, r, n_of_r)), 1))   # -> 12.0
print(round(float(np.interp(6.0, r, n_of_r)), 1))   # -> 18.0
print(round(float(np.interp(7.0, r, n_of_r)), 1))   # -> 42.0
```

Those are the FCC shells exactly: 12 nearest neighbours, then 6 more, then 24
more — 12, 18, 42. Because the crystal's $g(r)$ is a set of isolated spikes,
$n(R)$ is a staircase, and any $R$ in the flat region between two shells gives
the same integer. Run the identical four lines on the argon liquid and you get
the 12.9 quoted above; the only difference is that the liquid's steps are
rounded, so *where* you evaluate matters.

The argon curve in Figure 1 is the same calculation on a longer trajectory:

```python
# docs: skip — runs a 30 ps MD trajectory (minutes, not seconds)
from docs_data.run import argon_trajectory
from docs_data.structure import radial_distribution

radial_distribution(argon_trajectory())
```

## When the curve looks wrong

**$g(r)$ never settles to 1, it drifts up or down at large $r$.**
Your density is inconsistent with the box, or you are reading past $L/2$. The
normalization uses $\rho = N/V$ from the frame's box; if the box is wrong, every
value is scaled wrong.

**$g(r)$ rises steeply near $r_\max$ and $r_\max > L/2$.**
Under the minimum-image convention, a distance beyond half the shortest box edge
is not a well-defined separation — you start counting periodic images of atoms
you have already counted. Keep $r_\max \le L/2$. For the 28.9 Å argon box that
means 14 Å, which is why Figure 1 stops there.

**The curve is truncated or drops to zero early.**
The neighbor list cutoff is smaller than `r_max`. `RDF` can only histogram pairs
the neighbor list found. Set the cutoff at least equal to `r_max`.

**The first peak is low and broad, and the coordination number is too small.**
Too few bins. Each bin averages the structure across its width; a sharp shell
smeared over a 0.5 Å bin loses height. Use bins of roughly 0.02–0.05 Å.

**The curve is noisy and jumps between runs.**
$g(r)$ is an ensemble average. One frame of 500 atoms gives about 12 neighbours
per atom in the first shell — a few thousand counts spread over all bins. Average
hundreds of frames.

## Check yourself

- Predict $g(r)$ for an ideal gas, then compute it: place 200 points uniformly at
  random in a cubic box and histogram. You should get 1 everywhere, with no
  excluded core, and the scatter tells you your sampling noise.
- Move the evaluation point in the coordination snippet from 4.5 Å to 4.0 Å and
  to 5.0 Å. For the crystal nothing changes — you are still between shells. Try
  the same on a liquid and watch the answer drift.
- Halve the number of bins and watch the first peak height fall. Nothing physical
  changed; you only changed the resolution.

## References

- J.-P. Hansen, I. R. McDonald, *Theory of Simple Liquids*, 4th ed., Academic
  Press (2013) — chapters 2 and 4 for $g(r)$ and its relation to thermodynamics.
- M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.,
  Oxford (2017) — the histogram algorithm and its normalization.
- A. Rahman, *Phys. Rev.* **136**, A405 (1964) — the original MD study of liquid
  argon, the state point used here. DOI: 10.1103/PhysRev.136.A405

## See also

- [NeighborList](neighborlist.md) — the pair search $g(r)$ consumes
- [Diffraction](diffraction.md) — the same information in reciprocal space
- [Density](density.md) — where matter sits, without averaging over directions
- [API reference](../api/compute.md)
