# Mean squared displacement

How fast do the atoms in a liquid move?

The instantaneous speed is not the answer. An argon atom in the runs on this
page, at their mean temperature of 84.9 K, is travelling at about 230 m s⁻¹, but it is trapped in a cage of neighbours and spends most of
its time rattling back and forth without going anywhere. What you want is how
far it gets in the end — its **net** wandering — and that is what the mean
squared displacement measures.

## Averaging over particles and over starting times

Take one atom, look at where it is at time $t$ and again at time $t+\tau$, and
square the distance between. Do that for every atom, and for every possible
starting time $t$:

$$
\mathrm{MSD}(\tau) =
\big\langle\, |\mathbf{r}_i(t+\tau) - \mathbf{r}_i(t)|^{2} \,\big\rangle_{i,\,t}.
$$

Two averages are hiding in those angle brackets, and both matter.

Averaging over **particles** $i$ is obvious: 500 atoms give 500 samples.
Averaging over **starting times** $t$ is the one people forget. A 30 ps
trajectory contains only one 30 ps interval, but it contains thousands of 1 ps
intervals — every frame is a legitimate starting point. Using all of them is
what `Msd` does by default (`origins="all"`), and it is why the short-lag end
of an MSD curve is smooth while the long-lag end is ragged.

Neither average removes the noise of the trajectory itself. Lennard-Jones
dynamics is chaotic: rerun the same 500 atoms from a different random set of
initial velocities and the long-lag end of the curve, and the $D$ fitted to it,
move by several percent. A third average — over **independent runs** — is the
one that gives a number an error bar. The figures and numbers on this page come
from eight runs of the same state point that differ only in the seed of their
initial velocities.

Note also that displacement is squared, not signed. A random walk goes nowhere
on average, $\langle \mathbf{r}(t+\tau)-\mathbf{r}(t)\rangle = 0$, so the signed
average would tell you nothing. Squaring keeps the magnitude.

## The curve has three regimes

<figure id="fig-msd-argon" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
data: {$file: series/msd/argon_msd.json}
encoding:
  x:
    field: t
    type: quantitative
    title: "τ (fs)"
    scale: {type: log, domain: [10, 30000], nice: false}
    axis: {tickCount: 5, format: "~s"}
  y:
    field: msd
    type: quantitative
    title: "MSD (Å²)"
    scale: {type: log, domain: [4e-4, 50], nice: false}
    axis: {tickCount: 5, format: "~g"}
layer:
  - transform: [{filter: "datum.series === 'τ²'"}]
    mark:
      type: line
      strokeWidth: 1.4
      color: "#7a8a80"
      strokeDash: [7, 5]
      opacity: 0.85
    encoding:
      x: {field: t, type: quantitative}
      y: {field: msd, type: quantitative}
  - transform: [{filter: "datum.series === '6Dτ'"}]
    mark:
      type: line
      strokeWidth: 1.4
      color: "#7a8a80"
      strokeDash: [2, 4]
      opacity: 0.85
    encoding:
      x: {field: t, type: quantitative}
      y: {field: msd, type: quantitative}
  - transform: [{filter: "datum.series === 'MSD'"}]
    mark:
      type: line
      strokeWidth: 2.6
      color: "#0c5da5"
      interpolate: monotone
    encoding:
      x: {field: t, type: quantitative}
      y: {field: msd, type: quantitative}
  # Regime labels — native VL text (coords on/near the measured curve)
  - data:
      values:
        - {t: 28, msd: 0.012, label: ballistic}
        - {t: 9000, msd: 22, label: diffusive}
    mark:
      type: text
      fontSize: 13
      color: "#18432b"
      align: center
      baseline: bottom
      dy: -4
    encoding:
      x: {field: t, type: quantitative}
      y: {field: msd, type: quantitative}
      text: {field: label, type: nominal}
```

</div>

**Figure 1.** MSD of liquid argon on log–log axes, averaged over eight
independent runs (mean temperature 84.9 K). Solid blue is the measurement. The two muted dashed lines are
**not** fits to the shape: the steep one is $\langle v^2\rangle\tau^2$ from
equipartition, the shallow one is $6D\tau$ with $D$ the mean of the eight runs'
straight-line fits between 5 and 20 ps. Labels mark the
ballistic and diffusive windows used in the text.
</figure>

Log–log axes are used because power laws become straight lines, and their
exponent becomes a slope you can read by eye.

**Short times — ballistic, slope 2.** Before an atom has collided with anything,
it moves in a straight line: $\mathbf{r}(\tau) = \mathbf{v}\tau$, so
$\mathrm{MSD} = \langle v^2\rangle \tau^2$. Fitting the first 50 fs of these
runs gives a log–log slope of **1.994**. The atoms are simply flying, and
$\langle v^2\rangle$ follows from equipartition, $3k_BT/m$ — no dynamics
required. At $\tau = 10$ fs both the measurement and that prediction give
0.00053 Å².

**Intermediate times — the cage.** Around a few hundred femtoseconds the curve
bends below the ballistic line. The atom has hit its neighbours and is being
turned back. This is the regime that carries the interesting physics, and in
glasses or polymer melts it can stretch over many decades as a plateau.

**Long times — diffusive, slope 1.** Once an atom has forgotten which way it was
originally going, its motion is a random walk, and a random walk has
$\mathrm{MSD} \propto \tau$. This is the regime the Einstein relation applies to:

$$
\boxed{\;D = \lim_{\tau\to\infty} \frac{\mathrm{MSD}(\tau)}{2d\,\tau}
= \frac{1}{6}\lim_{\tau\to\infty}\frac{\mathrm{MSD}(\tau)}{\tau}\;}
$$

The $2d$ is just dimensionality — each of the $d = 3$ independent directions
contributes $2D\tau$ — so in three dimensions you divide the slope by 6. Fitting
each of the eight argon runs between 5 and 20 ps and averaging gives

$$
D = (1.95 \pm 0.12)\times10^{-5}\ \mathrm{cm^{2}\,s^{-1}},
$$

the mean over the eight runs with their sample standard deviation. The single
runs range from $1.80$ to $2.20\times10^{-5}$, so one 30 ps trajectory of 500
atoms pins $D$ down to no better than about 6 %; the mean of eight is good to
its standard error, $0.04\times10^{-5}$ (2 %).

Rahman's 1964 molecular-dynamics study of argon at the same density but a
slightly higher temperature (94.4 K) reported $2.43\times10^{-5}$ cm² s⁻¹.
Ours is colder and diffuses more slowly, which is the right direction and the
right magnitude. Getting a number you can check against the literature is the
point of running a standard system.

The [VACF](vacf.md) page reaches $D = (2.01 \pm 0.10)\times10^{-5}$ cm² s⁻¹
from the same eight runs by integrating the velocity autocorrelation instead of
fitting displacements. The two routes are mathematically equivalent, so this is
not an independent measurement of the physics — but they break differently
(unwrapping and fit window here; dump rate and drift there), so agreement within
their spreads tells you neither set of analysis choices is distorting the
result.

## A random walk shows the whole pipeline

The quickest way to trust an MSD is to feed it a trajectory whose answer you
already know. A random walk with independent Gaussian steps of width $\sigma$
per Cartesian direction has $\mathrm{MSD}(\tau) = 3\sigma^2\,\tau/\Delta t$
exactly, so $D = \sigma^2 / (2\Delta t)$.

`Msd` is configured with the longest lag it should report, in frames. It is a
**stream**: `accumulator()` hands out an accumulator, each frame's
`(n_atoms, 3)` positions are pushed into it in order together with the time
`dt` between frames, and `finish()` reads the curve. Memory stays bounded by
`max_lag`, however long the trajectory. What comes back is a
`TimeCorrelation` — the lag axis `lag_times` in the unit of `dt`, and
`values[lag, channel]` with one channel, the curve averaged over atoms and time
origins:

```python
import numpy as np
from molpy.compute import Msd

rng = np.random.default_rng(0)
sigma, dt = 0.1, 2.0                                  # Å per step, fs per frame
walk = np.cumsum(rng.normal(0.0, sigma, size=(400, 50, 3)), axis=0)

accumulator = Msd(max_lag=100).accumulator()
for positions in walk:                                # one (50, 3) frame at a time
    accumulator.push(positions, dt=dt)
msd = accumulator.finish()
print(msd.n_lags, msd.n_channels)                     # -> 101 1
print(msd.lag_times[:3])                              # -> [0. 2. 4.]
print(round(float(msd.values[1, 0]), 4), round(3 * sigma**2, 4))  # -> 0.03 0.03
```

One step in, the measured MSD already sits on $3\sigma^2$.

To extract $D$, fit a straight line through the part of the curve that is
actually linear — not the whole thing. The fit is its own compute, and it takes
the curve as its input. Its window is a pair of fractions of the lag axis:
`LinearFit(0.2, 0.8)` fits the lags from 20 % to 80 % of the way along. The
slope divided by $2d = 6$ is $D$:

```python
from molpy.compute import LinearFit

fit = LinearFit(0.2, 0.8).compute(correlation=msd, channel=0)
print(fit.fit_start, fit.fit_end, round(fit.r2, 4))   # -> 20 80 1.0
D = fit.slope / 6.0                                   # Å² / fs in 3D
print(round(D, 5), round(sigma**2 / (2 * dt), 5))     # -> 0.00262 0.0025
```

The 5 % gap is the noise of one 400-frame walk of 50 particles, not a bias:
change the seed and it moves to the other side.

On a real trajectory the window is a physical time range, not a convenient
fraction. Convert it once against the lag axis — for the argon figures, 5 to
20 ps:

```python
# docs: skip — continues from a 30 ps argon MSD with lag_times in fs
t_max = msd.lag_times[-1]
fit = LinearFit(5000.0 / t_max, 20000.0 / t_max).compute(correlation=msd, channel=0)
```

Choosing that window is a judgement you have to make and report. Start it after
the ballistic and caging regions end, and stop it well before the end of the
trajectory, where too few time origins remain. `fit.r2` is there so you can
check the window really was linear before quoting the slope.

## Unwrapped coordinates are not optional

`Msd` measures real displacements. That only works if each atom's path is a
**continuous** trajectory through space.

In practice, dynamics dumps are already written that way: LAMMPS
`dump … xu yu zu` (or `x y z ix iy iz` that you unwrap once on read), GROMACS
with no-jump, etc. The compute does not read frames: you read the coordinate
columns out of each frame and push them, one frame at a time. A normal
analysis pipeline over a dump of `xu yu zu` columns is therefore:

```python
# docs: skip — needs your own trajectory file
from pathlib import Path
import molpy as mp
from molpy.compute import Msd

accumulator = Msd(max_lag=2000).accumulator()
for frame in mp.io.lammps.LammpsDumpReader(Path("run.lammpstrj")):
    unwrapped = frame["atoms"]["xu", "yu", "zu"]               # (N, 3)
    accumulator.push(unwrapped, dt=10.0, simbox=frame.box)
msd = accumulator.finish()
```

`dt` is whatever interval your dump used, in fs. The figure on this page was
built from continuous paths like these.

What goes wrong if you feed **wrapped** positions (`x y z` folded into the
box)? An atom that left through one face and re-entered the opposite side is
recorded as a jump of nearly a full box length $L$. Those fake jumps dominate
the average: the curve loses any clean ballistic → diffusive shape and levels
off near a meaningless $\sim L^{2}/2$ set by the box size, not by the physics.

That is why each frame's box is worth passing, as `simbox=`. With it in hand,
the accumulator checks every frame-to-frame step, and a step longer than half
the box is refused rather than averaged:

```python
import molpy as mp

box = mp.Box(mp.core.Cell.lengths([10.0] * 3))
wrapped = walk[:50, :4] + 5.0
wrapped[25:, 0, 0] -= 10.0              # atom 0 "re-enters" through the far face

checked = Msd(max_lag=10).accumulator()
for positions in wrapped[:25]:          # frames 0-24 are continuous
    checked.push(positions, dt=dt, simbox=box)
print(checked.n_frames)                               # -> 25
```

Pushing frame 25, `checked.push(wrapped[25], dt=dt, simbox=box)`, raises

```text
ValueError: item 0 moved more than half the box between frames 24 and 25: the trajectory is wrapped; unwrap it before a displacement analysis
```

and a refused frame leaves the accumulator exactly as it was.

`Unwrap` recovers the continuous path. Given the int32 image flags of a dump
(`images=`), it uses them; given only the boxes, it follows each atom by the
minimum image between consecutive frames:

```python
from molpy.compute import Unwrap

boxes = [box] * len(wrapped)
continuous = Unwrap().compute(positions=wrapped, boxes=boxes)
accumulator_ok = Msd(max_lag=10).accumulator()
for positions in continuous:
    accumulator_ok.push(positions, dt=dt, simbox=box)
print(accumulator_ok.finish().n_lags)                 # -> 11
```

Following by continuity is only correct while nothing moves half a box between
two stored frames. If you have neither unwrapped coordinates nor image flags
and dump rarely, re-dump the run.

## The estimator has three knobs

`Msd` is the squared-displacement configuration of `Correlator`, molpy's one
time-correlation engine; what you tune is how the average is taken.

| Option | Choices (default first) | What it changes |
|---|---|---|
| `origins` | `"all"`, `"first"` | every frame is a time origin, or frame 0 only |
| `reduction` | `"mean"`, `"per_item"`, `"per_component"` | one curve averaged over atoms, one curve per atom, or one per Cartesian axis |
| `stride` | `1` | use every `stride`-th frame as an origin |

For a diffusion coefficient, keep `origins="all"`. With $T$ frames the
first-origin estimator, $\langle|\mathbf{r}(t)-\mathbf{r}(0)|^2\rangle$, has
exactly one sample per atom at every lag, while the all-origins one has
$T - \tau$ of them. The stream pairs each new frame with the last `max_lag`
frames it keeps, so the extra samples cost $\mathcal{O}(T \cdot
\text{max\_lag})$ time and no more than $\mathcal{O}(\text{max\_lag})$ frames of
memory. Use `"first"` only when you want the literal displacement from the
start of the run.

`reduction="per_component"` is the quick anisotropy check — in a bulk liquid
the three channels agree, under confinement they do not. `per_item` gives one
channel per atom, which is how you split an MSD by species after the fact:

```python
per_atom = Msd(max_lag=100, reduction="per_item").accumulator()
per_axis = Msd(max_lag=100, reduction="per_component").accumulator()
for positions in walk:
    per_atom.push(positions, dt=dt)
    per_axis.push(positions, dt=dt)
print(per_atom.finish().n_channels, per_axis.finish().n_channels)   # -> 50 3
```

## When it goes wrong

**MSD is enormous and jagged from the very first lag.**
Wrapped coordinates. Push each frame with `simbox=` so the accumulator
refuses them, and run `Unwrap` first. See above.

**MSD bends downward at long lag, or turns noisy and non-monotonic.**
Too few time origins out there. Fit well inside the trajectory — a common rule
is to use no more than the first half — or run longer.

**The log–log slope never reaches 1.**
The system has not reached the diffusive regime. This is normal and physically
meaningful in polymers, glasses, and gels; a "diffusion coefficient" fitted to a
sub-diffusive curve is not a diffusion coefficient. Report the exponent instead.

**$D$ changes a lot when you move the fitting window.**
Then the curve is not linear there. Plot MSD/$\tau$ against $\tau$: a genuine
diffusive regime shows up as a flat plateau, and if there is no plateau, there
is no $D$ to quote.

**$D$ is systematically small compared with experiment.**
Expected, and it is a finite-size effect rather than a bad force field. A
diffusing particle drags the surrounding fluid with it, and under periodic
boundaries it also drags the fluid around all of its own periodic images, which
holds it back. The resulting error shrinks only as $1/L$, so it does not go away
quickly. Compare against other simulations at the same box size, or apply the
Yeh–Hummer correction below.

**The centre of mass of the whole system drifts.**
Any net momentum adds a spurious ballistic $\propto \tau^2$ term at long times.
Remove the centre-of-mass motion before computing displacements.

## Check yourself

- Run the random walk above with a different step width $\sigma$ and confirm the
  MSD slope changes as $3\sigma^2$. You know the answer analytically, so any
  disagreement is your error, not physics.
- Compute the log–log slope of the first few points of your own MSD. If it is
  not close to 2, your timestep is too coarse to resolve the ballistic regime.
- Fit $D$ over two different windows. If the two answers differ by more than a
  few percent, you have not found the diffusive regime.

## References

- A. Einstein, *Ann. Phys.* **322**, 549 (1905) — the relation between mean
  squared displacement and the diffusion coefficient.
- A. Rahman, *Phys. Rev.* **136**, A405 (1964) — MD of liquid argon; the
  reference value quoted above. DOI: 10.1103/PhysRev.136.A405
- I.-C. Yeh, G. Hummer, *J. Phys. Chem. B* **108**, 15873 (2004) — the $1/L$
  finite-size correction to diffusion coefficients.

## See also

- [VACF](vacf.md) — the Green–Kubo route to the same $D$
- [PMSD](pmsd.md) — the same idea applied to collective charge displacement
- [Onsager](onsager.md) — cross-correlations between species
- [Van Hove](van_hove.md) — the full distribution behind the mean
- [API reference](../api/compute.md)
