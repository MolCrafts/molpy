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
and the continuous lifetime collapses to noise. So `Persist` takes **two** radii:
a pair becomes bonded inside $r_0$ and is only considered broken once it leaves
$r_1 > r_0$. Take $r_0$ at the first minimum of $g(r)$ and $r_1$ a little beyond.

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
data: {$file: data/persist/argon_survival.json}
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

$C(0) = 12.88$, against the coordination number 12.89 that [RDF](rdf.md) gets by
integrating $g(r)$. To be clear about what that check is worth: these are the
same physical quantity — the mean number of partners within $r_0$ — reached by
two different code paths, one counting indicator functions directly and the
other histogramming distances and integrating. Agreement confirms that your
$r_0$ matches the radius you integrated to and that `exclude_self` is set
correctly. It is an implementation check, not new physics, and it is the fastest
way to catch the two mistakes that most often silently wreck this analysis.

The decay is slow. The continuous curve falls to 0.91 at 1 ps, 0.73 at 3 ps and
0.54 at 6 ps. Fitting the tail gives a continuous residence time of about
9.7 ps — already a mild extrapolation, since the curve only just reaches half
its initial value inside the window. The intermittent curve decays more slowly
still and has not come close to $1/e$ by 6 ps, so this trajectory cannot pin its
lifetime down; treat it as "longer than 6 ps" and lengthen the run if you need
the number. That restraint is the same one the troubleshooting section below
asks of you.

The physical picture is clear even so. An argon atom keeps most of its
neighbours for many picoseconds while itself moving very little: [MSD](msd.md)
gives 8.0 Å², an rms displacement of 2.8 Å, over that same 6 ps — less than one
atomic diameter. So the shell is not being left behind; it travels *with* the
atom. That is what "cage" means quantitatively, and it is the same cage
[VACF](vacf.md) sees as a negative lobe at 440 fs.

## Computing it

`Persist.pair_survival_tcf` is a static method over two coordinate arrays. Watch
the shapes — the argument that catches everyone is `box_lengths`, which is
**per frame**, shape `(n_frames, 3)`, not a single box vector:

```python
import numpy as np
from molpy.compute import Persist

rng = np.random.default_rng(0)
n_frames = 300
# Five sites of each species, jittering around a fixed 3.2 Å separation.
coords_i = np.ascontiguousarray(rng.normal(0.0, 0.3, size=(n_frames, 5, 3)))
coords_j = np.ascontiguousarray(
    rng.normal(0.0, 0.3, size=(n_frames, 5, 3)) + np.array([3.2, 0.0, 0.0])
)
box = np.tile(np.array([[30.0, 30.0, 30.0]]), (n_frames, 1))

result = Persist.pair_survival_tcf(
    coords_i, coords_j, box, 3.5, 4.0, "continuous", 10.0, 40
)
print(sorted(result))                       # -> ['correlation', 'lag_times']
print(round(float(result["correlation"][0]), 2))   # -> 3.48
```

Pass a plain `(3,)` box and you get
`TypeError: argument 'box_lengths': 'ndarray' object is not an instance of
'ndarray'` — a thoroughly unhelpful message for what is really a shape error. If
you see it, tile your box to one row per frame.

The positional arguments in order are `coords_i`, `coords_j`, `box_lengths`,
`r0`, `r1`, `method`, `dt`, `max_correlation_time`, and optionally
`exclude_self`. Set `exclude_self=True` whenever the two coordinate arrays are
the same set of atoms, or every atom is found paired with itself at distance
zero forever.

Now compare the two definitions on identical data:

```python
curves = {}
for method in ("continuous", "intermittent"):
    out = Persist.pair_survival_tcf(
        coords_i, coords_j, box, 3.5, 4.0, method, 10.0, 40
    )
    curve = np.asarray(out["correlation"])
    curves[method] = curve / curve[0]

print(round(float(curves["continuous"][20]), 2))     # -> 0.41
print(round(float(curves["intermittent"][20]), 2))   # -> 0.96
```

At the same lag, intermittent has barely moved while continuous has more than
halved. These sites never actually leave each other's neighbourhood; they only
rattle across the threshold. That is precisely the situation the two definitions
exist to tell apart, and it is why quoting "the lifetime" without saying which
one you used is meaningless.

## When it goes wrong

**$C(0)$ does not match the coordination number from [RDF](rdf.md).**
Either $r_0$ differs from the radius you integrated to, or `exclude_self` is
wrong for a same-species calculation.

**The continuous lifetime is absurdly short.**
$r_0$ and $r_1$ are too close, so boundary rattling breaks pairs. Widen the
buffer.

**Continuous and intermittent lie on top of each other.**
Nothing is re-forming — plausible in a dilute gas, suspicious in a dense liquid.
Check that the buffer is not so wide that pairs can never break.

**The curve is still near 1 at the longest lag.**
`max_correlation_time` is too short to see the decay. You cannot fit a residence
time you have not observed; extend the window, or the trajectory.

**It takes minutes.**
It does. The kernel examines pairs at every lag, so cost grows with frames,
particles, and lag window together. The argon figure above takes several minutes
to generate. Reduce `max_correlation_time` first.

## Check yourself

- Confirm $C(0)$ equals your coordination number. If it does not, nothing else
  on the page is trustworthy.
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
