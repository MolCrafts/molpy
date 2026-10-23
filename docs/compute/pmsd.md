# Einstein conductivity

An electrolyte conducts because ions move. So can you get the conductivity by
measuring how fast the ions diffuse, using the [MSD](msd.md) machinery you
already have?

Almost — and the gap between "almost" and "exactly" is the physics of this page.

## Why self-diffusion is not enough

Conductivity is a **collective** property. Current flows only when positive and
negative charge move in *opposite* directions. If a cation and an anion drift
along together as a neutral pair, both have a perfectly respectable
self-diffusion coefficient and together they carry no current at all.

Estimating conductivity from self-diffusion alone gives the **Nernst–Einstein**
result

$$
\sigma_\mathrm{NE} = \frac{e^{2}}{V k_B T}\sum_\alpha N_\alpha z_\alpha^{2} D_\alpha ,
$$

which assumes every ion moves independently of every other. Real electrolytes —
concentrated ones and ionic liquids especially — have correlated motion, and the
true conductivity usually lands 20–70 % *below* $\sigma_\mathrm{NE}$. The ratio
$\sigma/\sigma_\mathrm{NE}$ is the ionicity, or degree of uncorrelated ion
motion, and measuring it is often the entire point of the calculation.

To get $\sigma$ itself you need a quantity that already contains the
cancellation. That quantity is the **collective charge displacement**

$$
\mathbf{M}(t) = \sum_a q_a \mathbf{r}_a(t),
$$

sometimes called the Helfand moment or the itinerant dipole. A cation–anion pair
moving together contributes nothing to $\mathbf{M}$: their charges are equal and
opposite, so the pair cancels itself out, exactly as it should.

## The Einstein relation, applied to charge

Now run the [MSD](msd.md) argument again with $\mathbf{M}$ in place of a
particle position:

$$
\mathrm{MSD}_M(\tau) = \big\langle
|\mathbf{M}(t+\tau)-\mathbf{M}(t)|^{2}\big\rangle_{t},
$$

$$
\boxed{\;\sigma = \lim_{\tau\to\infty}\frac{1}{6\,V k_B T}\,
\frac{\mathrm{d}\,\mathrm{MSD}_M(\tau)}{\mathrm{d}\tau}\;}
$$

Same shape as $D = \lim \mathrm{MSD}/6\tau$, same factor of 6 from three
dimensions, same requirement to find a genuinely linear region before fitting.
Everything on the [MSD](msd.md) page transfers — including that **$\mathbf{M}$
must be built from unwrapped coordinates**. One ion teleporting across the box
injects a spurious jump of $qL$ into the collective sum.

## Computing it

There is no conductivity class to call, because there is nothing new to
compute: the Einstein conductivity is two existing computes in a row. The
collective dipole $\mathbf{M}(t)$ is a `DipoleMoment`, and its mean squared
displacement is an `Msd` of a "trajectory" with a single item in it.

`DipoleMoment` takes unwrapped positions, shape `(n_frames, n_atoms, 3)`, and
one charge per atom. Without a `partition` it sums the whole system, and
`series(0)` hands back $\mathbf{M}(t)$ as an `(n_frames, 3)` array. Only you
know which atoms carry which charge, so the charges come from your own
topology — typically the `q` column of the atoms block:

```python
import numpy as np
from molpy.compute import DipoleMoment

rng = np.random.default_rng(1)
charges = np.array([1.0, -1.0, 1.0, -1.0])
unwrapped = np.cumsum(rng.normal(0.0, 0.05, size=(400, 4, 3)), axis=0)

M = DipoleMoment().compute(positions=unwrapped, charges=charges).series(0)
print(M.shape)                            # -> (400, 3)
```

`Msd` is a stream that takes one `(n_items, 3)` frame per `push`;
$\mathbf{M}$ is one item, so give each frame the axis it lacks. To keep the example's answer known in advance, the
$\mathbf{M}$ used from here on is a pure random walk:

```python
from molpy.compute import Msd

walk = np.random.default_rng(1)
M = np.cumsum(walk.normal(0, 0.02, size=(400, 3)), axis=0)
accumulator = Msd(max_lag=100).accumulator()
for m in M:
    accumulator.push(m[None, :], dt=10.0)                # (1, 3): one item
raw = accumulator.finish()
print(raw.n_lags, raw.n_channels)         # -> 101 1
```

A partition splits the sum by species: give `DipoleMoment` one unsigned label
per atom (`partition=labels`) and `moments[frame, group, :]` holds one
$\mathbf{P}_\alpha(t)$ per label — the input of the [Onsager](onsager.md)
cross-correlations.

The `Msd` returns the raw curve and stops. Fitting its linear region is the
next compute, `LinearFit`, and its window is a *fraction* of the lag axis, so
`(0.1, 0.5)` means from 10 % to 50 % of the way along it:

```python
from molpy.compute import LinearFit

fit = LinearFit(0.1, 0.5).compute(correlation=raw, channel=0)
print(round(fit.r2, 3))                   # -> 0.989
```

Check `r2` before believing the slope. This synthetic $\mathbf{M}$ is a pure
random walk, so it is linear by construction and scores 0.989 — the shortfall
from 1 is finite-sample noise, not curvature. A real $\mathrm{MSD}_M$ with a
caging shoulder scores much worse if you include the shoulder in the window,
which is precisely the warning you want.

Unwrapping belongs before `DipoleMoment`, not after it. Run `Unwrap` on the
ion positions (with their image flags, or by continuity from the per-frame
boxes) so no wrapped ion ever enters $\mathbf{M}$. Do not pass `simbox=` when
pushing $\mathbf{M}$ into `Msd`: a dipole is not a position in the box, and the
half-box check that guards a particle MSD means nothing for it.

### Getting to S/m

`fit.slope` is in $e^2\,\text{Å}^2\,\text{fs}^{-1}$ when charges are in
elementary units, positions in Å, and `dt` in fs. Converting to SI is one
constant:

$$
\sigma\ [\mathrm{S\,m^{-1}}] = 3.0988\times10^{9}\;
\frac{\text{slope}\ [e^{2}\,\text{Å}^{2}\,\text{fs}^{-1}]}
{V\ [\text{Å}^{3}]\; \times\; T\ [\mathrm{K}]}
$$

```python
def conductivity_si(slope, volume_a3, temperature_k):
    """Einstein conductivity in S/m from a fitted MSD_M slope."""
    return 3.0988e9 * slope / (volume_a3 * temperature_k)
```

Resist applying that to the `fit` above. The `M` in this example is a random
walk with an arbitrary step size, not a physical charge displacement, so it
would produce an arbitrary number that happens to look like a plausible
conductivity. The helper is right; the input is a placeholder.

That constant is $e^2\,\text{Å}^2 / (6\,\text{fs}\cdot\text{Å}^3 k_B)$ evaluated
in SI. The factor of 6 from the Einstein relation is already inside it — do not
divide by 6 again.

!!! note "No figure on this page yet — TODO"
    A meaningful $\mathrm{MSD}_M$ figure needs an electrolyte trajectory with
    charges: a molten salt, or a solvated salt with Ewald electrostatics. The
    reference system behind the other compute pages is neutral argon, for which
    $\mathbf{M} \equiv 0$. Plotting a random walk here and labelling it
    "conductivity" would be decoration, not measurement. This page gets a figure
    when a charged reference trajectory exists under `scripts/docs_series/`.

## When it goes wrong

**$\mathrm{MSD}_M$ has huge discrete jumps.**
Wrapped coordinates. One ion crossing the box adds $qL$ to $\mathbf{M}$ in a
single step.

**$\sigma$ comes out far too large.**
Check that you summed *signed* $q_a$ rather than $|q_a|$. Check also that the
system is neutral: unless $\sum_a q_a = 0$, the value of $\mathbf{M}$ depends on
where you put the origin, and so does your answer.

**$\mathrm{MSD}_M$ is far noisier than a particle MSD.**
Expected, and it is the main practical difficulty. $\mathbf{M}$ is one
collective vector per frame, so each time origin gives one sample instead of
$N$. Collective transport needs much longer trajectories than self-diffusion —
often tens of nanoseconds where $D$ would be converged in one.

**$\sigma$ changes a lot with the fitting window.**
There is no linear region yet. Plot $\mathrm{MSD}_M/\tau$ and look for a
plateau before quoting anything.

**$\sigma \approx \sigma_\mathrm{NE}$ exactly.**
Suspicious for a concentrated electrolyte. Check that $\mathbf{M}$ is a genuine
collective sum and not a per-ion quantity averaged afterwards.

## Check yourself

- Compute $\sum_a q_a$ for your system. If it is not zero to machine precision,
  stop — $\mathbf{M}$ is not well defined and neither is your conductivity.
- Build $\mathbf{M}$ for a frozen configuration repeated many times. The
  $\mathrm{MSD}_M$ must be identically zero.
- Compare your $\sigma$ with $\sigma_\mathrm{NE}$ from ion self-diffusion. A
  ratio above 1 is a red flag; 0.3–0.8 is typical.

## References

- E. Helfand, *Phys. Rev.* **119**, 1 (1960) — the moment whose mean squared
  displacement yields a transport coefficient.
- H. K. Kashyap et al., *J. Phys. Chem. B* **115**, 13212 (2011) — collective
  versus self transport in ionic liquids.
- N. Molinari, J. P. Mailoa, B. Kozinsky, *Chem. Mater.* **31**, 8748 (2019) —
  Nernst–Einstein deviations in concentrated electrolytes.

## See also

- [JACF](jacf.md) — the Green–Kubo route to the same $\sigma$
- [Onsager](onsager.md) — which species correlations cause the deviation
- [MSD](msd.md) — the single-particle version of this argument
- [Dielectric](dielectric.md) — the frequency-resolved response
- [API reference](../api/compute.md)
