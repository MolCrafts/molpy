# Velocity autocorrelation

[MSD](msd.md) asks how far an atom got. This page asks a different question:
**how long does an atom remember which way it was going?**

The answer turns out to contain the same diffusion coefficient, reached by a
completely different route, plus the vibrational spectrum of the system for
free. That is the appeal of the velocity autocorrelation function — one
correlation function, two observables.

## Correlating a velocity with its own past

Take an atom's velocity now and its velocity a time $\tau$ later, and take the
dot product. Average over atoms and over starting times:

$$
C_{vv}(\tau) = \big\langle\, \mathbf{v}_i(t)\cdot\mathbf{v}_i(t+\tau)
\,\big\rangle_{i,\,t}.
$$

At $\tau = 0$ this is $\langle v^2\rangle$, which equipartition fixes exactly:

$$
C_{vv}(0) = \frac{3k_BT}{m}.
$$

That makes the zero-lag value a free thermometer, and the first thing you should
check. Averaged over the eight independent argon runs used here (they differ
only in the seed of their initial velocities), the measured $C_{vv}(0)$ is
$5.288\times10^{-6}$ Å² fs⁻², and $3k_BT/m$ at the runs' mean temperature of
84.9 K is $5.299\times10^{-6}$. Agreement to 0.2 % means the velocities are in
the units you think they are.

As $\tau$ grows, collisions scramble the velocity and the correlation decays to
zero. *How* it decays is where the physics is.

## Reading a real curve

<figure id="fig-vacf-argon" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
data: {$file: series/vacf/argon_vacf.json}
mark: {type: line, strokeWidth: 2.4, interpolate: monotone}
encoding:
  x:
    field: t
    type: quantitative
    title: "lag τ (fs)"
    scale: {domain: [0, 2500]}
  y:
    field: c
    type: quantitative
    title: "Ĉ_vv(τ)"
    scale: {domain: [-0.2, 1.0]}
```

</div>

**Figure 1.** Normalized velocity autocorrelation of liquid argon, averaged
over the eight independent runs (mean temperature 84.9 K) with the mean kept.
The correlation crosses zero at 310 fs,
reaches a minimum of $-0.105$ at 440 fs, and has essentially decayed by 2 ps.
</figure>

The shape has three features worth naming.

**It starts flat, not with a kink.** Expanding for small $\tau$ gives
$C_{vv}(\tau) \approx C_{vv}(0)\left(1 - \tfrac12\omega_E^2\tau^2\right)$: the
leading correction is quadratic, so the curve leaves 1 with zero slope. The
curvature defines the Einstein frequency $\omega_E$, the frequency at which an
atom rattles in the cage formed by its neighbours.

**It goes negative.** This is the signature of a dense liquid and it is not
noise. At 440 fs the correlation is $-0.105$: the average atom is moving
*backwards* relative to where it started. It has bounced off the wall of its
cage of neighbours — the same first coordination shell that [$g(r)$](rdf.md)
shows at 3.68 Å. A dilute gas has no cage and decays monotonically to zero; a
solid oscillates for far longer.

**It relaxes to zero.** By 2 ps the atom has forgotten its initial velocity
entirely. Compare that with the [MSD](msd.md) of the same runs, which needs about
5 ps before it becomes reliably linear: velocity correlations die *faster* than
displacements become diffusive, which is why the VACF needs finely spaced
frames and the MSD does not.

## The Green–Kubo route to D

Linear-response theory relates the time integral of a flux autocorrelation to
the corresponding transport coefficient. For self-diffusion:

$$
\boxed{\;D = \frac{1}{d}\int_0^{\infty}
\big\langle \mathbf{v}(0)\cdot\mathbf{v}(t)\big\rangle\,\mathrm{d}t
= \frac{1}{3}\int_0^{\infty} C_{vv}(t)\,\mathrm{d}t\;}
$$

This is mathematically equivalent to the Einstein relation on the [MSD](msd.md)
page: one integrates velocities, the other differentiates displacements, and for
an infinitely long trajectory they are the same number by construction.

Be precise about what that means for running both. Because they are equivalent,
agreement is **not** independent confirmation of the physics — it is the same
dynamics measured twice. What differs is the *numerical path*, and the two paths
fail in different ways. The Einstein route is vulnerable to unwrapping errors
and to a fitting window that strays into the sub-diffusive region; the
Green–Kubo route is vulnerable to a velocity dump too coarse to resolve the
initial decay, and to drift that makes the integral diverge. So a disagreement
localizes a mistake, and an agreement says your analysis choices are not
distorting the answer. That is worth having, and it is less than "two
independent measurements".

In practice you never integrate to infinity. You compute the **running
integral**

$$
D(\tau) = \frac{1}{3}\int_0^{\tau} C_{vv}(t)\,\mathrm{d}t
$$

and look for where it stops changing.

<figure id="fig-vacf-running" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
data: {$file: series/vacf/argon_running_diffusion.json}
mark: {type: line, strokeWidth: 2.4, interpolate: monotone}
encoding:
  x:
    field: t
    type: quantitative
    title: "τ_max (fs)"
  y:
    field: D
    type: quantitative
    title: "D(τ) (cm² s⁻¹)"
    scale: {domain: [0, 3.5e-5]}
```

</div>

**Figure 2.** Running Green–Kubo integral for argon, from the VACF of
Figure 1, averaged over the eight runs. It overshoots while the
VACF is still positive, is pulled back down by the negative lobe, and settles
onto a plateau once the correlation has decayed.
</figure>

Follow the shape and the physics is visible. The integral climbs while the
correlation is positive and peaks at $2.86\times10^{-5}$ cm² s⁻¹ at **310 fs** —
which is exactly where Figure 1 shows the VACF crossing zero, because that is
the moment the integrand changes sign. Then the negative lobe, the cage pushing
atoms back, *subtracts* from the integral and drags it down to
$2.06\times10^{-5}$ by 1.5 ps. Past 2 ps there is nothing left to add and the
curve is flat.

**Quote the plateau, never the peak and never the last point.** Here the plateau
is $2.01\times10^{-5}$ cm² s⁻¹. Taking each run's own plateau gives
$(2.01 \pm 0.10)\times10^{-5}$ cm² s⁻¹ (mean and sample standard deviation of
the eight runs), against $(1.95 \pm 0.12)\times10^{-5}$ from the Einstein fits
of the same runs: the two routes agree within their spreads.

If there is no plateau, do not pick a number off the curve. It means the
trajectory is too short, the lag window is too small, or centre-of-mass drift is
adding a constant that integrates without bound.

## Computing it

`Vacf` is configured with the longest lag it reports, `max_lag`, which counts
**frames, not femtoseconds**: fifty lags at a 10 fs frame spacing reach 500 fs,
and the result has `max_lag + 1` lags because lag 0 is included. The data goes
to `compute`: the velocities and the frame spacing `dt`, which sets the lag
axis `lag_times`.

The velocities are a `(n_frames, n_dof)` series — one column per Cartesian
degree of freedom. A `(T, N, 3)` velocity trajectory is reshaped to
`(T, 3N)`, which is free in NumPy. `Vacf` correlates every column with its own
past and averages the columns *after* correlating, so the curve is the memory
of a typical atom along a typical axis:

$$
C(\tau) = \big\langle v_{i\alpha}(t)\,v_{i\alpha}(t+\tau)\big\rangle_{i,\,\alpha,\,t}
= \tfrac13\,C_{vv}(\tau).
$$

That factor of three is the whole difference from the dot-product form above:
$C(0) = k_BT/m$, and the Green–Kubo integral of $C$ is $D$ itself, with no
$1/d$ left to apply. Like every time correlation in `molpy.compute`, the
result is a `TimeCorrelation` — `values[lag, channel]` with one channel here:

```python
import numpy as np
from molpy.compute import Vacf

rng = np.random.default_rng(0)
velocities = rng.normal(0.0, 0.02, size=(2000, 64, 3))   # Å/fs, (T, N, 3)

vacf = Vacf(max_lag=50).compute(velocities=velocities.reshape(2000, -1), dt=10.0)
print(vacf.n_lags, vacf.n_channels)                       # -> 51 1
print(round(float(vacf.values[0, 0]), 6))                 # -> 0.000401
```

These velocities are uncorrelated noise with a per-component width of
0.02 Å fs⁻¹, so $C(0)$ should be $\sigma^2 = 0.0004$, and it is. The next lag
is already down by a factor of a thousand — normalising is arithmetic on
`values`:

```python
normalized = vacf.values[:, 0] / vacf.values[0, 0]
print(round(float(normalized[1]), 4))                     # -> 0.0004
```

White noise has no memory. Real velocities do, and that difference is the whole
measurement.

Two conventions sit underneath that curve, and both are configuration. The
first is what to do with each column's time mean: by default `Vacf` subtracts
it before correlating (`mean="subtract"`), and `mean="keep"` correlates the
velocities exactly as given. The next section is about why that choice decides
the Green–Kubo answer. The second is how each lag is averaged: by default every
lag is divided by the number of origins that reach it
(`normalization="unbiased"`), `"biased"` divides every lag by the number of
frames, and `"none"` returns the raw sums.

The Green–Kubo $D$ is a composition of two more computes, each taking the
previous curve as its input. `CumulativeTrapezoid` turns the correlation into
its **running integral** — itself a `TimeCorrelation` on the same lag axis,
precisely so you can look for the plateau — and `Plateau` averages a window of
it, given as fractions of the lag axis:

```python
from molpy.compute import CumulativeTrapezoid, Plateau

running = CumulativeTrapezoid().compute(correlation=vacf, channel=0)
plateau = Plateau(0.8, 1.0).compute(correlation=running, channel=0)
print(running.n_lags, plateau.n_samples)                  # -> 51 11
D = plateau.value                                         # Å²/fs: C is per degree of freedom
```

`plateau.std` is the spread of the running integral inside the window; if it
is not small next to `plateau.value`, there is no plateau to quote.

Units follow your inputs. Velocities in Å fs⁻¹ give $D$ in Å² fs⁻¹, and
1 Å² fs⁻¹ = 0.1 cm² s⁻¹.

Be careful not to sum atoms *before* correlating. A summed velocity is the
**collective** momentum, and its autocorrelation is a different quantity — the
current correlation of the [JACF](jacf.md) page, `CurrentAcf`. For a
per-particle VACF, hand `Vacf` every degree of freedom and let it average after
correlating.

### Subtracting the mean costs a finite run part of $D$

Subtracting each column's time mean makes the correlation immune to a constant
drift, but over a finite run it is not free. An atom that diffused a distance
$\Delta r$ in a run of length $T$ has a time-mean velocity $\Delta r/T$ along
each axis, and that is real motion, not drift. Removing it lowers every lag of
the correlation by about $2D/T$, so the running integral out to $\tau_{\max}$
comes out low by a fraction $2\tau_{\max}/T$. For the argon runs here,
$\tau_{\max} = 2.5$ ps and $T = 30$ ps, that is about 17 %: with the mean
subtracted, the eight runs give $D = 1.68\times10^{-5}$ cm² s⁻¹ against the
$2.01\times10^{-5}$ quoted above.

In equilibrium the true mean velocity is zero, so the curve with the mean kept
is the Green–Kubo integrand itself, and that is what every figure and number on
this page uses. It is the same compute with one convention changed:

```python
vacf_kept = Vacf(max_lag=50, mean="keep").compute(
    velocities=velocities.reshape(2000, -1), dt=10.0
)
running_kept = CumulativeTrapezoid().compute(correlation=vacf_kept, channel=0)
D_kept = Plateau(0.8, 1.0).compute(correlation=running_kept, channel=0).value
print(round(float(vacf_kept.values[0, 0]), 6))            # -> 0.000401
```

Nothing else changes: the curve is still per degree of freedom, so its
integral is $D$ with no $1/d$ to apply. On this white noise the two policies give the
same zero-lag value, because its time mean is already close to zero; on a real
liquid their Green–Kubo integrals part by the 17 % above.

Keep the mean when the run is long next to the lag window and net momentum
was removed before dumping; subtract it when a residual drift would otherwise
integrate without bound.

## Vibrations, from the same correlation function

The Wiener–Khinchin theorem says the Fourier transform of an autocorrelation
function is a power spectrum. Applied to velocities, that spectrum is the
**vibrational density of states**:

$$
g(\omega) \propto \int_{-\infty}^{\infty}
\big\langle \mathbf{v}(0)\cdot\mathbf{v}(t)\big\rangle\,
e^{-i\omega t}\,\mathrm{d}t .
$$

Unlike infrared or Raman spectra, there are no selection rules: every motion the
atoms actually perform contributes. The zero-frequency weight is proportional to
$D$ — the same integral as above, seen as the $\omega \to 0$ limit — so a liquid
has $g(0) > 0$ while a solid (after removing drift) has $g(0) = 0$.

The transform is one more compute. `Spectrum(window, pad)` tapers a real
signal with the window, zero-pads it to `pad` times its length and returns the
one-sided transform: angular frequencies $\omega$ in radians per unit of `dt`,
and the real and imaginary parts, unscaled. The VDOS is the real part of the
transform of the VACF:

```python
from molpy.compute import Spectrum

vdos = Spectrum("cosine_sq", 2).compute(signal=vacf.channel(0), dt=10.0)
print(vdos.n_fft, vdos.frequencies.shape == vdos.real.shape)   # -> 128 True
wavenumber = vdos.frequencies / (2 * np.pi * 2.99792458e-5)    # rad/fs -> cm⁻¹
print(round(float(wavenumber[-1])))                            # -> 1668
```

The window is selectable (`"cosine_sq"`, `"hann"`, `"blackman"`, `"none"`);
the cosine² taper is one at lag zero, so the integral of the spectrum — $D$
again — is untouched.

The spectral resolution is set by the length $T$ of the correlation function,
$\Delta\tilde\nu \approx 1/(cT)$ — here 51 lags at 10 fs is $T = 0.51$ ps and
about 65 cm⁻¹, which is coarse. Padding interpolates the grid; it does not
resolve anything new. Resolving narrow features means correlating out to longer
lags, not sampling more finely.

See [Spectra](spectra.md), which generalizes the same machinery to dipole and
polarizability fluxes.

**Sampling sets your frequency ceiling.** The Nyquist limit is
$\tilde{\nu}_{\max} \approx 16678/(\Delta t/\mathrm{fs})$ cm⁻¹. At $\Delta t =
10$ fs — the spacing used for argon — that is 1668 cm⁻¹, fine for a monatomic
liquid whose motion is all below 100 cm⁻¹, but hopeless for O–H stretches near
3600 cm⁻¹. Those need sub-femtosecond dumps.

## When it goes wrong

**$C_{vv}(0)$ does not equal $3k_BT/m$.**
Units. Å fs⁻¹ against m s⁻¹, or a mass in the wrong system. Fix this before
looking at anything else on the page.

**The correlation never decays to zero; the running integral grows without
bound.**
Centre-of-mass drift. Subtracting each degree of freedom's time mean (the
`Vacf` default) removes a constant drift, but a slowly wandering collective
velocity still correlates with itself over long lags. Remove net momentum before
dumping velocities.

**There is no negative lobe in a dense liquid.**
Frames are too far apart. The lobe here sits at 440 fs; sampling every 500 fs
would step straight over it.

**$D(\tau)$ has no plateau.**
`max_lag` is too short to reach where $C_{vv} \approx 0$, or the trajectory is
too short for the tail to average out. Lengthen one or the other; do not quote
the endpoint.

**Green–Kubo and Einstein disagree by more than a few percent.**
Suspect the MSD side first: unwrapping, or a fit window that includes
sub-diffusive lags. If both are clean, suspect the thermostat.

**The VACF looks over-damped and $D$ comes out low.**
A strong Langevin or Berendsen thermostat imposes its own friction on the
dynamics — it adds random kicks and drag that are not in your force field, and
those decorrelate velocities faster than the physics does. Equilibrate with the
thermostat, then switch it off and sample transport properties at constant
energy (an NVE run, fixed particle number, volume, and energy).

## Check yourself

- Compute $C_{vv}(0)$ from your own trajectory and divide by $3k_B T/m$. You
  should get 1.00. If not, stop and find the unit error.
- Integrate the VACF to get $D$, then compute $D$ from the [MSD](msd.md) of the
  same run. Agreement within a few percent means both are working.
- Take the argon running integral and read it at 400 fs, before the negative
  lobe has done its work. You get $2.77\times10^{-5}$ cm² s⁻¹ — 38 % above the
  plateau, and a concrete demonstration of why the plateau, not an early value,
  is the answer.

## References

- M. S. Green, *J. Chem. Phys.* **22**, 398 (1954); R. Kubo, *J. Phys. Soc.
  Jpn.* **12**, 570 (1957) — the linear-response relations.
- A. Rahman, *Phys. Rev.* **136**, A405 (1964) — the first VACF of liquid argon,
  including the negative lobe.
- D. Frenkel, B. Smit, *Understanding Molecular Simulation*, 2nd ed. (2002),
  §4.4 — practical Green–Kubo estimation.
- J.-P. Hansen, I. R. McDonald, *Theory of Simple Liquids*, 4th ed. (2013),
  ch. 7 — time correlation functions.

## See also

- [MSD](msd.md) — the Einstein route to the same $D$
- [Spectra](spectra.md) — VDOS, IR, and Raman from the same transform
- [API reference](../api/compute.md)
