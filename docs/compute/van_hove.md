# Van Hove correlation

[$g(r)$](rdf.md) is a photograph. It tells you how atoms are arranged, averaged
over frames, and says nothing about how that arrangement came about or how long
it lasts.

The **Van Hove correlation function** is the movie. Given a particle at the
origin at time 0, what is the density of particles at distance $r$ a time $t$
later? At $t=0$ it reduces to $g(r)$; as $t$ grows it shows structure
dissolving.

## Two parts, two different physics

$$
G(r,t) = \underbrace{
  \frac{1}{N}\Big\langle \sum_i \delta\big(r-|\mathbf{r}_i(t)-\mathbf{r}_i(0)|\big)\Big\rangle
}_{G_s(r,t)\ \text{same particle}}
\;+\;
\underbrace{
  \frac{1}{N}\Big\langle \sum_{i\neq j} \delta\big(r-|\mathbf{r}_i(t)-\mathbf{r}_j(0)|\big)\Big\rangle
}_{G_d(r,t)\ \text{different particles}}
$$

Splitting on whether $i$ and $j$ are the same particle is not bookkeeping. The
two halves answer different questions.

**$G_s(r,t)$ follows one particle.** It is the probability distribution of how
far a tagged atom has travelled in time $t$: a spike at the origin when $t=0$,
broadening as the atom wanders. Its second moment is the [MSD](msd.md),

$$
\langle r^2(t)\rangle = \int_0^\infty r^2\,G_s(r,t)\,\mathrm{d}r ,
$$

which is the point of the page. The MSD compresses this entire distribution into
one number, and compression hides things: two systems with identical MSDs can
have completely different $G_s$ — one where every particle drifts alike, one
where most are trapped and a few hop far.

**$G_d(r,t)$ follows the neighbours.** At $t=0$ it is exactly $g(r)$. As $t$
grows the coordination shells wash out and it flattens toward 1: the memory of
where the neighbours were has gone. How fast that happens is the structural
relaxation time.

## Watching a distribution spread

<figure id="fig-vanhove" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
config:
  legend:
    orient: bottom
    direction: horizontal
    title: null
data: {$file: series/van_hove/argon_self.json}
mark: {type: line, strokeWidth: 2.2, interpolate: monotone}
encoding:
  x:
    field: r
    type: quantitative
    title: "r (Å)"
    scale: {domain: [0, 8]}
  y:
    field: g
    type: quantitative
    title: "G_s(r, t)"
  color:
    field: lag
    type: nominal
    title: null
```

</div>

**Figure 1.** Self part of the Van Hove function for liquid argon, from the
reference run (seed 0, 88.3 K), at
four lags. The distribution starts narrow and spreads; its root-mean-square
width grows from 0.22 Å at 0.1 ps to 2.62 Å at 6 ps.
</figure>

Every curve is a normalized probability distribution — they all integrate to 1,
which is the first thing to confirm. What changes is the width, and the widths
are not arbitrary. Taking $\int r^2 G_s\,\mathrm{d}r$ at each lag gives 0.054,
0.71, 2.71 and 7.67 Å², against [MSD](msd.md) values of 0.053, 0.70, 2.71 and
7.67 Å² from the same trajectory's unwrapped positions at the same time origins.
They agree to within 2 %, the residual coming from the bin width of the
histogram, which matters most at the shortest lag where the whole distribution
spans a few bins.

That is the cross-check to run on your own system: if the second moment of your
$G_s$ does not reproduce your MSD, one of the two is wrong.

For purely diffusive motion $G_s$ is Gaussian. The departure from that is
quantified by the **non-Gaussian parameter**

$$
\alpha_2(t) = \frac{3\langle r^4\rangle}{5\langle r^2\rangle^2} - 1,
$$

which is 0 for a Gaussian and rises wherever motion is heterogeneous — some
particles caged, others hopping. In supercooled liquids $\alpha_2$ peaks near the
end of the caging plateau and is the standard measure of dynamic heterogeneity.
Compute it from the moments of $G_s$; it is not a field on the result.

## Computing it

`VanHove` is configured with the radial binning — an `Axis` over $[0,
r_\max)$ — and the lags to evaluate, **in frames**. The data is the
trajectory: `(n_frames, n_atoms, 3)` [unwrapped](msd.md) positions, for the
same reason the MSD needs them, and one box per frame, which the distinct part
needs for the minimum image. Given those boxes, a wrapped trajectory is refused
rather than analysed.

A random walk in a periodic box makes a test whose answers are known: every
$G_s$ is a Gaussian shell, and its second moment is the walk's MSD.

```python
import numpy as np
import molpy as mp
from molpy.compute import Axis, Msd, VanHove

rng = np.random.default_rng(0)
L, n_frames, n_atoms = 20.0, 200, 200
start = rng.uniform(0.0, L, size=(n_atoms, 3))
positions = start + np.cumsum(rng.normal(0.0, 0.1, size=(n_frames, n_atoms, 3)), axis=0)
boxes = [mp.Box(mp.core.Cell.lengths([L] * 3))] * n_frames

result = VanHove(Axis(0.0, 6.0, 60), lags=[10, 50, 150]).compute(
    positions=positions, boxes=boxes
)
print(result.g_self.shape, result.lags)   # -> (3, 60) [ 10  50 150]
```

On your own data the positions are the coordinate columns of each frame,
stacked: `np.stack([frame.coords for frame in frames])`, with
`[frame.box for frame in frames]` as the boxes.

The result holds `g_self` and `g_distinct`, one row per requested lag, on the
bins `r_centers` / `r_edges` of width `dr`. Each self-part row is a probability
density in $r$ and integrates to 1, and its second moment reproduces the MSD:

```python
r, dr = result.r_centers, result.dr
print(np.round((result.g_self * dr).sum(axis=1), 6))      # -> [1. 1. 1.]
second = (r**2 * result.g_self * dr).sum(axis=1)
accumulator = Msd(max_lag=150).accumulator()
for frame_positions in positions:
    accumulator.push(frame_positions, dt=1.0)
msd = accumulator.finish().values[result.lags, 0]
print(np.round(second / msd, 3))                          # -> [1.004 1.001 1.001]
```

The distinct part is a number density: it tends to $\rho = N/V$ where the
neighbours have forgotten each other, so `g_distinct / rho` is the curve that
starts at $g(r)$ and flattens toward 1. The non-Gaussian parameter is
arithmetic on the moments of `g_self`. For a random walk the true value is
zero at every lag:

```python
fourth = (r**4 * result.g_self * dr).sum(axis=1)
alpha2 = 3 * fourth / (5 * second**2) - 1
print(np.round(alpha2, 2))                                # -> [0.   0.06 0.1 ]
```

The growth with lag is not physics. At 150 frames each atom has only a
handful of independent displacements left, so the fourth moment of 200 atoms is
a noisy estimate — computing $\alpha_2$ straight from the raw displacements
gives the same three numbers. Before reading heterogeneity into $\alpha_2$,
check that it is larger than this floor for a run of your length.

Choose $r_\max$ from the longest lag you care about: it must cover the bulk of
$G_s$, not just the peak. If $r_\max$ is too small the second moment falls
short of the true MSD while the integral can still read 1.000 — the
distribution is renormalized over whatever grid you gave it, so normalization
alone will not tell you that you clipped the tail. Only the moments will.

Three options tune the estimator. `origins="all"` (the default) averages over
every time origin and `"first"` uses frame 0 alone; `stride` spaces the
origins — raise it to trade statistics for speed. `upper_edge` decides the one
sample that lands exactly on $r_\max$: `"open"` (the default) keeps every bin
half-open, $[r_k, r_{k+1})$, and drops it; `"closed"` counts it in the last
bin, the convention of codes whose final bin is $[r_{n-1}, r_\max]$.

## When it goes wrong

**$G_s$ does not integrate to 1.**
Check `dr`: the curves are densities, not counts, so the sum needs the bin
width.

**The second moment is lower than your MSD.**
`r_max` is clipping the tail of the displacement distribution. The integral will
still come out as 1, so this failure is invisible unless you check the moment.
Raise `r_max` until the answer stops changing.

**The compute refuses the trajectory as wrapped.**
It is. Displacement kernels need continuous paths; run `Unwrap` first, see
[MSD](msd.md).

**$G_d(r,0)$ does not equal $g(r)$.**
It must, by definition, once divided by the number density $\rho = N/V$. A
remaining mismatch means a different density normalization between the two
calls.

**The curves at large lag are noisy.**
Few time origins remain out there. Lower `stride`, or shorten the longest lag.

**It is slow.**
Cost scales as the number of bins × `len(lags)` times the number of origins
processed. Ask for the handful of lags you will actually plot, not a dense
sweep, and raise `stride`.

## Check yourself

- Confirm every $G_s$ curve integrates to 1.
- Compute the second moment at each lag and compare with your MSD at the same
  time. Agreement within a few percent means both are right.
- For a random walk, compute $\alpha_2$; you should get approximately 0 at every
  lag, because the displacement distribution really is Gaussian.

## References

- L. Van Hove, *Phys. Rev.* **95**, 249 (1954) — the original correlation
  function.
- J.-P. Hansen, I. R. McDonald, *Theory of Simple Liquids*, 4th ed. (2013),
  ch. 7 — $G_s$, $G_d$, and the intermediate scattering function.
- W. Kob, H. C. Andersen, *Phys. Rev. E* **51**, 4626 (1995) — the non-Gaussian
  parameter and dynamic heterogeneity.

## See also

- [MSD](msd.md) — the second moment of $G_s$, in one number
- [RDF](rdf.md) — $G_d$ at $t = 0$
- [Reorientation](reorientation.md) — the rotational counterpart
- [API reference](../api/compute.md)
