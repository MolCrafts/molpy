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
data: {$file: data/van_hove/argon_self.json}
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

**Figure 1.** Self part of the Van Hove function for liquid argon at 85 K, at
four lags. The distribution starts narrow and spreads; its root-mean-square
width grows from 0.23 Å at 0.1 ps to 2.76 Å at 6 ps.
</figure>

Every curve is a normalized probability distribution — they all integrate to 1,
which is the first thing to confirm. What changes is the width, and the widths
are not arbitrary. Taking $\int r^2 G_s\,\mathrm{d}r$ at each lag gives 0.051,
0.67, 2.65 and 7.64 Å², against MSD values of 0.051, 0.68, 2.75 and 8.04 Å²
from the [MSD](msd.md) page. They agree to within a few percent, the residual
coming from different time-origin sampling in the two calculations.

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

`VanHove` takes the lags up front, **in frames**, and needs
[unwrapped](msd.md) coordinates for the same reason the MSD does. Load an
unwrapped dump and pass the frame list straight in:

```python
# docs: skip — needs your own trajectory file
from molpy.compute import VanHove
from molpy.io import read_lammps_trajectory

frames = read_lammps_trajectory("run.lammpstrj").read_all()
result = VanHove(n_rbins=100, r_max=12.0, lags=[10, 50, 200], stride=10)(frames)

g_self = np.asarray(result.g_self)
print(g_self.shape)    # -> (n_lags, n_rbins), one row per requested lag
```

Useful fields: `g_self`, `g_distinct` (when requested), `r_centers`, `r_edges`,
`dr`, `lags`. Each self-part row should integrate to 1 over $r$ — check
`(g_self[i] * dr).sum()`. The second moment $\int r^{2} G_s(r,t)\,\mathrm{d}r$
is the MSD at that lag, so it should match [`MSD`](msd.md) on the same frames.

Choose `r_max` from the longest lag you care about: it must cover the bulk of
$G_s$, not just the peak. If `r_max` is too small the second moment falls short
of the true MSD while the integral can still read 1.000 — the distribution is
renormalized over whatever grid you gave it, so normalization alone will not
tell you that you clipped the tail. Only the moments will.

`stride` sets the spacing between time origins — raise it to trade statistics
for speed. `has_distinct` reports whether the distinct part was accumulated.

## When it goes wrong

**$G_s$ does not integrate to 1.**
Check `dr`: the curves are densities, not counts, so the sum needs the bin
width.

**The second moment is lower than your MSD.**
`r_max` is clipping the tail of the displacement distribution. The integral will
still come out as 1, so this failure is invisible unless you check the moment.
Raise `r_max` until the answer stops changing.

**$G_s$ has weight at large $r$ that grows with lag.**
Wrapped coordinates. Displacement kernels need unwrapping; see [MSD](msd.md).

**$G_d(r,0)$ does not equal $g(r)$.**
It must, by definition. A mismatch means a different density normalization
between the two calls.

**The curves at large lag are noisy.**
Few time origins remain out there. Lower `stride`, or shorten the longest lag.

**It uses far more memory than expected.**
Cost scales as `n_rbins × len(lags)` times the number of origins processed. Ask
for the handful of lags you will actually plot, not a dense sweep.

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
