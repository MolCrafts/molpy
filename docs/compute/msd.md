# Mean squared displacement

How fast do the atoms in a liquid move?

The instantaneous speed is not the answer. An argon atom at 85 K is travelling
at about 230 m s⁻¹, but it is trapped in a cage of neighbours and spends most of
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
what `MSD(method="window")` does, and it is why the short-lag end of an MSD
curve is smooth while the long-lag end is ragged.

Note also that displacement is squared, not signed. A random walk goes nowhere
on average, $\langle \mathbf{r}(t+\tau)-\mathbf{r}(t)\rangle = 0$, so the signed
average would tell you nothing. Squaring keeps the magnitude.

## The curve has three regimes

<figure id="fig-msd-argon" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
data: {$file: data/msd/argon_msd.json}
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

**Figure 1.** MSD of liquid argon at 85 K on log–log axes. Solid blue is the
measurement. The two muted dashed lines are **not** fits to the shape: the
steep one is $\langle v^2\rangle\tau^2$ from equipartition, the shallow one is
$6D\tau$ with $D$ from a straight-line fit between 5 and 20 ps. Labels mark the
ballistic and diffusive windows used in the text.
</figure>

Log–log axes are used because power laws become straight lines, and their
exponent becomes a slope you can read by eye.

**Short times — ballistic, slope 2.** Before an atom has collided with anything,
it moves in a straight line: $\mathbf{r}(\tau) = \mathbf{v}\tau$, so
$\mathrm{MSD} = \langle v^2\rangle \tau^2$. Fitting the first 50 fs of this
trajectory gives a log–log slope of **1.994**. The atoms are simply flying, and
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
this argon between 5 and 20 ps gives

$$
D = 2.21\times10^{-5}\ \mathrm{cm^{2}\,s^{-1}}.
$$

Rahman's 1964 molecular-dynamics study of argon at the same density but a
slightly higher temperature (94.4 K) reported $2.43\times10^{-5}$ cm² s⁻¹.
Ours is colder and diffuses more slowly, which is the right direction and the
right magnitude. Getting a number you can check against the literature is the
point of running a standard system.

The [VACF](vacf.md) page reaches $D = 2.23\times10^{-5}$ cm² s⁻¹ from the same
trajectory by integrating the velocity autocorrelation instead of fitting
displacements. The two routes are mathematically equivalent, so this is not an
independent measurement of the physics — but they break differently (unwrapping
and fit window here; dump rate and drift there), so agreement tells you neither
set of analysis choices is distorting the result. Note also that neither number
carries an error bar, so do not read significance into the last digit; block
averaging over trajectory segments is how you would get one.

## Unwrapped coordinates are not optional

`MSD` measures real displacements. That only works if each atom’s path is a
**continuous** trajectory through space.

In practice, dynamics dumps are already written that way: LAMMPS
`dump … xu yu zu` (or `x y z ix iy iz` that you unwrap once on read), GROMACS
with no-jump, etc. A normal analysis pipeline is therefore:

```python
# docs: skip — needs your own trajectory file
import numpy as np
from molpy.io import read_lammps_trajectory
from molpy.compute import MSD

frames = read_lammps_trajectory("run.lammpstrj").read_all()
series = MSD(method="window")(frames)
msd = np.asarray(series.mean)
lag = np.arange(len(msd)) * 10.0   # fs, whatever Δt your dump used
```

There is no separate “unwrap” step in molpy’s compute path: **the frames you
pass in must already carry unwrapped coordinates.** The figure on this page was
built that way.

What goes wrong if you instead feed **wrapped** positions (`x y z` folded into
the box)? Nothing crashes. An atom that left through one face and re-entered
the opposite side is recorded as a jump of nearly a full box length $L$. Those
fake jumps dominate the average: the curve loses any clean ballistic →
diffusive shape and levels off near a meaningless $\sim L^{2}/2$ set by the box
size, not by the physics. That is why production diffusion work dumps unwrapped
paths up front.

If you only have wrapped columns plus image flags (`ix iy iz`), recover the
continuous path **before** analysis with `Box.unwrap` (and `Box.get_images` when
you need images from a single configuration). If you have neither unwrapped
coordinates nor image flags, re-dump the run — reconstructing images by
detecting jumps larger than $L/2$ between frames fails for anything that moves
that far in one dump interval.

## Choosing the estimator

`MSD` offers two estimators and they are not interchangeable.

| `method` | Definition | Use when |
|---|---|---|
| `"direct"` (default) | $\langle|\mathbf{r}(t)-\mathbf{r}(0)|^2\rangle$, frame 0 is the only origin | you want the literal displacement from the start of the run |
| `"window"` | averaged over every time origin | you want a diffusion coefficient |

For a diffusion coefficient, use `"window"`. With $T$ frames the direct
estimator has exactly one sample at the longest lag, while the windowed one has
$T - \tau$ of them, and it costs $\mathcal{O}(T\log T)$ rather than
$\mathcal{O}(T^2)$ because it goes through an FFT.

The result is an `MSDTimeSeries`; `series.mean` is the curve, one value per lag,
with lag $k$ corresponding to $k \times \Delta t$ in the time unit of your dump
spacing.

To extract $D$, fit a straight line over the part of the curve that is actually
linear — not the whole thing:

```python
# docs: skip — continues from series / lag above
from molpy.compute import LinearFit

window = (lag >= 5000.0) & (lag <= 20000.0)   # fs; match your ballistic→diffusive crossover
fit = LinearFit(0.0, 1.0).fit(lag[window], msd[window])
D = fit["slope"] / 6.0                        # Å² / fs in 3D
```

Choosing that window is a judgement you have to make and report. Start it after
the ballistic and caging regions end, and stop it well before the end of the
trajectory, where too few time origins remain.

## When it goes wrong

**MSD is enormous and jagged from the very first lag.**
Wrapped coordinates. See above.

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
