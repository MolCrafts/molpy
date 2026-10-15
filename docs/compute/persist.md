# Pair survival

[RDF](rdf.md) tells you that an argon atom has about 13 neighbours. It cannot
tell you whether they are the *same* 13 neighbours a picosecond later.

That distinction matters. A crystal and a liquid can have similar coordination
numbers and completely different lifetimes. An ion pair that survives
nanoseconds behaves like a neutral molecule; one that survives femtoseconds does
not. Structure is a snapshot; **persistence is the movie**.

## Turning "still together" into a correlation function

Define an indicator $h_{ij}(t)$ that is 1 when atoms $i$ and $j$ count as paired
at time $t$ and 0 otherwise. Then correlate it with itself:

$$
C(\tau) = \Big\langle \sum_j h_{ij}(t)\,h_{ij}(t+\tau)\Big\rangle_{i,\,t}.
$$

Because $h$ is an indicator rather than a fluctuation, $C(0)$ is not 1 — it is
the **mean number of partners**, which is the coordination number. That makes
the zero-lag value a free cross-check against [RDF](rdf.md), and it is the first
thing to look at. $C(\tau)/C(0)$ then decays from 1 as pairs break, and its
decay time is the residence time you were after.

## Continuous or intermittent: two different questions

The subtlety is what happens when a pair breaks and later re-forms. There are
two defensible answers, and they measure different physics.

**Continuous** requires $h$ to have been 1 at *every* intervening step. It
measures how long a bond survives without interruption: once broken, always
broken. This is the definition of a lifetime.

**Intermittent** looks only at the endpoints. A pair counts as together at
$\tau$ even if it separated and came back. This measures how long two partners
stay in each other's *neighbourhood*, and it is the right choice for structural
relaxation, being insensitive to brief rattling.

Continuous always decays at least as fast as intermittent. The gap between them
is a direct measure of how much re-formation is happening.

There is a second knob for the same problem. With a single distance threshold, a
pair sitting right at the cutoff flickers on and off many times per picosecond
and the continuous lifetime collapses to noise. So `PairPresence`, which turns
distances into the indicator $h_{ij}(t)$, takes **two** radii: a pair becomes
bonded inside $r_0$ and is only considered broken once it leaves $r_1 > r_0$. Take $r_0$ at the first minimum of $g(r)$ and $r_1$ a little beyond.

## Reading a real curve

Below is first-shell survival in liquid argon, with $r_0 = 5.4$ Å (the first
minimum of $g(r)$) and $r_1 = 5.9$ Å.

<figure id="fig-persist-argon" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
config:
  legend:
    orient: bottom
    direction: horizontal
    title: null
data: {$file: series/persist/argon_survival.json}
mark: {type: line, strokeWidth: 2.4, interpolate: monotone}
encoding:
  x:
    field: t
    type: quantitative
    title: "lag τ (fs)"
  y:
    field: c
    type: quantitative
    title: "C(τ)/C(0)"
    scale: {domain: [0.45, 1.0]}
  color:
    field: series
    type: nominal
    title: null
```

</div>

**Figure 1.** First-shell pair survival in liquid argon at 85 K, by the
continuous and intermittent definitions. They separate as soon as pairs begin
breaking and re-forming.
</figure>

Normalised per reference atom and counted from the inner radius (the two
conventions described under "Computing it" below), $C(0) = 12.88$, against the
coordination number 12.89 that [RDF](rdf.md) gets by integrating $g(r)$. To be clear about what that check is worth: these are the
same physical quantity — the mean number of partners within $r_0$ — reached by
two different code paths, one counting indicator functions directly and the
other histogramming distances and integrating. Agreement confirms that your
$r_0$ matches the radius you integrated to and that `exclude_self` is set
correctly. It is an implementation check, not new physics, and it is the fastest
way to catch the two mistakes that most often silently wreck this analysis.

The decay is slow. The continuous curve falls to 0.90 at 1 ps, 0.73 at 3 ps and
0.55 at 6 ps. Fitting the tail gives a continuous residence time of about
10.0 ps — an extrapolation, since the curve has not yet reached half its
initial value inside the window. The intermittent curve decays more slowly
still and has not come close to $1/e$ by 6 ps, so this trajectory cannot pin its
lifetime down; treat it as "longer than 6 ps" and lengthen the run if you need
the number. That restraint is the same one the troubleshooting section below
asks of you.

The physical picture is clear even so. An argon atom keeps most of its
neighbours for many picoseconds while itself moving very little: [MSD](msd.md)
gives 7.4 Å², an rms displacement of 2.7 Å, over that same 6 ps — less than one
atomic diameter. So the shell is not being left behind; it travels *with* the
atom. That is what "cage" means quantitatively, and it is the same cage
[VACF](vacf.md) sees as a negative lobe at 440 fs.

## Computing it

Survival is two computes in a row, because it answers two questions in turn:
*which pairs are together at each frame*, and *how long does togetherness
last*.

The first question is geometry, so it starts from neighbour tables — one per
frame, from the core neighbour search, searched out to at least $r_1$. Here are
five sites of one species and five of another, jittering around a fixed
3.2 Å separation; each frame's table is a cross-query of the second set
against the first:

```python
import numpy as np
import molpy as mp

rng = np.random.default_rng(0)
n_frames = 300
coords_i = rng.normal(0.0, 0.3, size=(n_frames, 5, 3))
coords_j = rng.normal(0.0, 0.3, size=(n_frames, 5, 3)) + np.array([3.2, 0.0, 0.0])
box = mp.Box.cube(30.0)

tables = [
    mp.core.NeighborQuery(box, coords_j[t], 4.0).query(coords_i[t])
    for t in range(n_frames)
]
```

On a real trajectory the coordinates come out of each frame through the
Frame/Block API, and a same-species shell is one `NeighborList` per frame
instead.

`PairPresence` is configured with the two radii and with `exclude_self`, which
has no default on purpose: set it to `True` when both point sets are the same
atoms, or every atom is found paired with itself at distance zero forever. It
returns a `Presence` — `values[t, k]` is 1 while pair `pairs[k]` is bonded —
or `None` when no pair ever came within $r_0$:

```python
from molpy.compute import PairPresence

presence = PairPresence(3.5, False, r1=4.0).compute(frames=tables)
print(presence.values.shape, presence.pairs[1])   # -> (300, 25) [0 1]
```

The second question is a time correlation, and `Survival` is the
configuration of the correlation engine that answers it. Its `kind` is the
definition — `"continuous"` or `"intermittent"` — and it takes the presence
series and the frame spacing:

```python
from molpy.compute import Survival

continuous = Survival("continuous", 40).compute(presence=presence.values, dt=10.0)
print(continuous.n_lags, round(float(continuous.values[0, 0]), 3))   # -> 41 0.944
```

By default the curve is averaged over the tracked **pairs**, so $C(0)$ is the
fraction of pair-frames that are bonded — 0.944 here — rather than a count.
The coordination-number convention of the equation at the top of the page is
one option away. `normalization="per_reference"` rescales to partners per
reference atom, with the number of reference atoms given as data
(`n_reference`). And by default a time origin counts whenever a pair is in its
bonded state; `origin="inner"` counts it only when the pair sits inside $r_0$
at that frame — the stable-states initial condition — given the presence at
the inner radius alone, a second `PairPresence` with $r_1 = r_0$:

```python
inner = PairPresence(3.5, False).compute(frames=tables)
per_atom = Survival(
    "continuous", 40, normalization="per_reference", origin="inner"
).compute(presence=presence.values, dt=10.0, n_reference=5, inner=inner.values)
print(round(float(per_atom.values[0, 0]), 2))     # -> 3.48
```

Now compare the two definitions on identical data:

```python
curves = {}
for kind in ("continuous", "intermittent"):
    out = Survival(kind, 40).compute(presence=presence.values, dt=10.0)
    curves[kind] = out.values[:, 0] / out.values[0, 0]

print(round(float(curves["continuous"][20]), 2))     # -> 0.41
print(round(float(curves["intermittent"][20]), 2))   # -> 0.95
```

At the same lag, intermittent has barely moved while continuous has more than
halved. These sites never actually leave each other's neighbourhood; they only
rattle across the threshold. That is precisely the situation the two definitions
exist to tell apart, and it is why quoting "the lifetime" without saying which
one you used is meaningless.

A lifetime is then the area under the normalised curve: hand the survival
curve to `CumulativeTrapezoid` and read where its running integral levels off,
exactly as the [VACF](vacf.md) page does for diffusion.

## When it goes wrong

**$C(0)$ does not match the coordination number from [RDF](rdf.md).**
Check the conventions first: only `normalization="per_reference"` with
`origin="inner"` counts partners per atom. Then either $r_0$ differs from the
radius you integrated to, or `exclude_self` is wrong for a same-species
calculation.

**The continuous lifetime is absurdly short.**
$r_0$ and $r_1$ are too close, so boundary rattling breaks pairs. Widen the
buffer.

**Continuous and intermittent lie on top of each other.**
Nothing is re-forming — plausible in a dilute gas, suspicious in a dense liquid.
Check that the buffer is not so wide that pairs can never break.

**The curve is still near 1 at the longest lag.**
`max_lag` is too short to see the decay. You cannot fit a residence
time you have not observed; extend the window, or the trajectory.

**It takes a long time, or a lot of memory.**
The presence series has one column per pair that was ever bonded, for every
frame, so cost grows with frames, particles and lag window together. Reduce
`max_lag` first, then the number of frames.

**`PairPresence` returned `None`.**
No pair ever came within $r_0$ — usually a cutoff in the wrong unit, or a
neighbour search with a smaller cutoff than $r_1$.

## Check yourself

- Confirm $C(0)$ in the per-reference, inner-origin convention equals your
  coordination number. If it does not, nothing else on the page is
  trustworthy.
- Run both methods. Continuous must decay at least as fast as intermittent at
  every lag; if it does not, something is wrong.
- Set $r_1 = r_0$ and watch the continuous lifetime collapse. That is the
  flicker problem, and it is why the buffer exists.

## References

- F. H. Stillinger, A. Rahman, *J. Chem. Phys.* **60**, 1545 (1974) — the
  two-threshold bonded criterion.
- D. C. Rapaport, *Mol. Phys.* **50**, 1151 (1983) — continuous and intermittent
  correlation functions.
- A. Luzar, D. Chandler, *Nature* **379**, 55 (1996) — the reactive-flux view of
  bond lifetimes.

## See also

- [RDF](rdf.md) — where $r_0$ and the $C(0)$ cross-check come from
- [HBond](hbond.md) — the same idea with an angle-aware geometric criterion
- [VACF](vacf.md) — the cage, seen in the velocity correlation
- [Onsager](onsager.md) — whether pairing shows up in transport
- [API reference](../api/compute.md)
