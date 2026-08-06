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

MolPy will not scan a trajectory for you here, because only you know which atoms
carry which charge. Assemble $\mathbf{M}(t)$ yourself, shape `(n_frames, 3)`,
then hand it over:

```python
import numpy as np
from molpy.compute import EinsteinConductivity, LinearFit

rng = np.random.default_rng(1)
M = np.ascontiguousarray(np.cumsum(rng.normal(0, 0.02, size=(400, 3)), axis=0))

raw = EinsteinConductivity().compute(M, dt=10.0, max_correlation_time=100)
print(sorted(raw), raw["msd"].shape)     # -> ['lag_times', 'msd'] (101,)
```

In real work `M` comes from your own charges and unwrapped positions:

```python
# docs: skip — needs your own trajectory and charge array
M = np.array([
    (charges[:, None] * unwrapped[i]).sum(axis=0)
    for i in range(len(unwrapped))
])
```

The compute returns the raw curve and stops. Fit the linear region yourself —
the window is a *fraction* of the curve, so `(0.1, 0.5)` means from 10 % to 50 %
of the way along it:

```python
fit = LinearFit(0.1, 0.5).fit(raw["lag_times"], raw["msd"])
print(round(fit["r2"], 3))                # -> 0.989
```

Check `r2` before believing the slope. This synthetic $\mathbf{M}$ is a pure
random walk, so it is linear by construction and scores 0.989 — the shortfall
from 1 is finite-sample noise, not curvature. A real $\mathrm{MSD}_M$ with a
caging shoulder scores much worse if you include the shoulder in the window,
which is precisely the warning you want.

### Getting to S/m

`fit["slope"]` is in $e^2\,\text{Å}^2\,\text{fs}^{-1}$ when charges are in
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
    when a charged reference trajectory exists under `scripts/docs_data/`.

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
