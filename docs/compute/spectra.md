# Spectra

A normal-mode calculation gives you a stick spectrum of a molecule frozen at the
bottom of a well: harmonic, at 0 K, one conformer. Real spectra are none of
those things. They are broadened by the environment, shifted by anharmonicity,
and averaged over every configuration the molecule actually visits.

Molecular dynamics gives you all of that for free — if you know how to read a
spectrum out of a trajectory. The route is always the same, and it is worth
seeing the shape of it before any of the individual spectra:

**Pick the fluctuating quantity the light couples to. Correlate it with its own
past. Fourier transform. Multiply by a prefactor.**

## One recipe, five spectra

Formally, the spectral density of any dynamical variable $A(t)$ is the Fourier
transform of its time-correlation function,

$$
I(\omega) \;\propto\; Q(\omega)\int_{-\infty}^{\infty}
\big\langle A(0)\,A(t)\big\rangle\,e^{-i\omega t}\,\mathrm{d}t ,
$$

with $Q(\omega)$ a method-specific prefactor. Every spectrum on this page is
that expression with a different $A$:

| Spectrum | $A$ — the quantity light couples to | Correlation | Transform |
|---|---|---|---|
| VDOS | atomic velocities | `Vacf` | `Spectrum` |
| Infrared | dipole flux $\dot{\mathbf{M}}$ | `DipoleMoment` → `Derivative` → `CurrentAcf` | `Spectrum` |
| VCD | electric ⊗ magnetic dipole flux, $\dot{\boldsymbol\mu}\cdot\dot{\mathbf m}$ | `Derivative` of both → `Correlator` cross | `Spectrum` |
| Raman | polarizability flux (iso + aniso) | `PolarizabilityParts` → `Derivative` → `Correlator` | `RamanSpectrum` |
| Resonance Raman | resonant polarizability flux | as Raman, on $\alpha(\omega_L)$ | `RamanSpectrum` |
| ROA | $\dot\alpha$ against $\dot G'$ | parts of both → `Correlator` cross | `RamanSpectrum` |

So the physics of "which modes are visible" is not in the algorithm at all — it
is in the choice of $A$. A mode that does not modulate the dipole contributes
nothing to $\langle\dot{\mathbf M}(0)\cdot\dot{\mathbf M}(t)\rangle$ and is
therefore infrared-silent. That is the familiar selection rule, arriving here as
a property of a correlation function rather than as a symmetry table — and
unlike the symmetry table, it comes with anharmonicity and temperature already
included.

Every row is a composition of three kinds of step, each its own compute: a
**series operation** that builds $A(t)$ from the trajectory, a **correlation**
that turns it into $\langle A(0)A(t)\rangle$, and a **transform** that turns the
curve into a spectrum. None of them takes frames: you read positions,
velocities or charges out of your frames with the Frame/Block API, and hand
arrays over. Only the transform differs between the vibrational spectra, and it
is the same `Spectrum` for all but the Raman family, whose transform also owns
the scattering prefactors.

## The vibrational density of states

The simplest case needs no electronic structure at all — only velocities:

$$
\boxed{\;g(\omega) \propto \int_{-\infty}^{\infty}
\big\langle \mathbf{v}(0)\cdot\mathbf{v}(t)\big\rangle\,
e^{-i\omega t}\,\mathrm{d}t\;}
$$

Because velocities couple to everything, the VDOS shows **every** mode the atoms
execute, IR-active or not. That makes it the natural first spectrum to compute
and the natural reference against which to ask "why is this peak missing from my
IR?"

<figure id="fig-vdos" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
data: {$file: series/spectra/argon_vdos.json}
mark: {type: line, strokeWidth: 2.4, interpolate: monotone}
encoding:
  x:
    field: nu
    type: quantitative
    title: "ν̃ (cm⁻¹)"
    scale: {domain: [0, 200]}
  y:
    field: I
    type: quantitative
    title: "g(ν̃)"
```

</div>

**Figure 1.** Vibrational density of states of liquid argon, from the
velocity autocorrelation of the reference run (seed 0, 88.3 K) — one of the
eight runs averaged on the [VACF](vacf.md) page. The
band peaks at 20 cm⁻¹ and has died away by 150 cm⁻¹.
</figure>

Two features carry the physics. The spectrum is **non-zero at zero frequency** —
0.67 of the peak height — because $g(0) \propto D$ and a liquid diffuses; a
solid, after removing drift, goes to zero there. And the whole band lies below
about 150 cm⁻¹, because argon is monatomic and has no internal vibrations at
all: everything visible is atoms rattling in their cages, the same motion the
[VACF](vacf.md) shows as a negative lobe at 440 fs. A molecular liquid adds
sharp intramolecular bands one to two orders of magnitude higher.

## Three numbers that decide whether your spectrum is meaningful

Before computing anything, settle these. They are not tuning parameters; they
are hard limits set by how you sampled.

**The ceiling is your timestep.** With frame spacing $\Delta t$ in fs,

$$
\tilde\nu_{\max} \approx \frac{1}{2c\,\Delta t}
\approx \frac{16678}{\Delta t/\mathrm{fs}}\ \mathrm{cm}^{-1}.
$$

At $\Delta t = 0.5$ fs that is 33 356 cm⁻¹ — everything. At 10 fs it is
1668 cm⁻¹, so C–H and O–H stretches near 3000 cm⁻¹ do not merely come out
inaccurate, they **alias** back into the fingerprint region as fake peaks.
Reaching 3000 cm⁻¹ needs $\Delta t \lesssim 2.5$ fs.

**The resolution is your total correlation length.** An ACF of duration $T$
resolves no better than $\Delta\tilde\nu \approx 1/(cT)$: about 33 cm⁻¹ for a
1 ps correlation, 3.3 cm⁻¹ for 10 ps. Sampling faster does not help here — only
correlating longer does. These are independent knobs and it is easy to reach for
the wrong one.

**Classical intensities are wrong at high frequency**, because classical
mechanics puts $k_BT$ into every mode while quantum mechanics does not.
The usual harmonic correction is

$$
Q(\omega)=\frac{\beta\hbar\omega}{1-e^{-\beta\hbar\omega}},
\qquad \beta=1/(k_B T),
$$

applied after the transform. It matters most where $\hbar\omega \gtrsim k_BT$ —
above roughly 200 cm⁻¹ at room temperature, which is to say for every
intramolecular band.

`Spectrum` does not apply it: it is the bare transform of whatever curve you
give it, and multiplying its intensities by $Q(\omega)$ is a line in your own
script. `RamanSpectrum` applies the Bose–Einstein factor
$1/(1-e^{-hc\tilde\nu/k_BT})$ when — and only when — you pass the state's
`temperature` to `compute`. Either way, whether you corrected belongs in your
methods section.

## Correlating costs $\mathcal{O}(T\log T)$, not $\mathcal{O}(T^2)$

The autocorrelation of a series of length $T$ looks like $T$ separate sums, one
per lag, which is $\mathcal{O}(T^2)$ and quickly impossible. The
Wiener–Khinchin theorem gives a shortcut: the autocorrelation is the inverse
Fourier transform of the power spectrum, so an FFT there and back costs
$\mathcal{O}(T\log T)$. For a 10⁵-frame trajectory the difference is between
seconds and hours, and it is why every correlation in molpy runs on one engine,
`Correlator`. The named correlations on these pages — `Vacf`, `CurrentAcf`,
`DipoleAcf`, `Msd`, … — are `Correlator` with the configuration that defines
the quantity fixed; a correlation with no name of its own is `Correlator`
configured directly.

Test it on a signal whose correlation you can write down. A cosine of period
$P$ has an autocorrelation that is also a cosine of period $P$. `Correlator`
takes an `(n_frames, n_items, n_components)` array, here one item with one
component, and the frame spacing `dt`:

```python
import numpy as np
from molpy.compute import Correlator

t = np.arange(2048, dtype=float)
series = np.cos(2 * np.pi * t / 64)                       # period 64 frames

acf = Correlator("product", 200).compute(a=series[:, None, None], dt=10.0)
c = acf.channel(0)
print(acf.n_lags, acf.lag_times[1])                       # -> 201 10.0
print(round(float(c[64] / c[0]), 2), round(float(c[32] / c[0]), 2))   # -> 1.0 -1.0
```

At a lag of one full period the correlation is back to +1; at half a period it
is −1. Note the length: `max_lag=200` gives **201** lags, because lag 0 is a lag.
And note the estimator: every lag is divided by the number of origins that
contribute to it, $T - t$ (`normalization="unbiased"`, the default), so `c[0]`
is $\langle\cos^2\rangle = 0.5$, not 1 and not the raw sum. `"biased"` divides
every lag by $T$, `"none"` leaves the raw sum; divide by `c[0]` yourself when
you want a curve that starts at 1.

## A truncated correlation must be tapered before it is transformed

An FFT assumes the signal repeats forever. If your correlation function has not
decayed to zero by the last lag, the transform sees a discontinuity where the
end wraps around to the beginning, and that artificial step scatters energy
across the whole spectrum as ringing — sinc side-lobes around every real peak.

The fix is to multiply by a window that goes smoothly to zero at the end before
transforming. `Spectrum(window, pad)` does both steps: it tapers the signal,
zero-pads it to `pad` times its length (rounded up to a power of two), and
returns the one-sided transform. The cosine's correlation above stops at 201
lags, which is not a whole number of periods, so it is a truncated signal; its
largest side-lobe, relative to the peak, shows what each window buys:

```python
from molpy.compute import Spectrum

for window in ("none", "cosine_sq", "hann", "blackman"):
    spectrum = Spectrum(window, 1).compute(signal=c, dt=10.0)
    power = np.abs(spectrum.real)
    peak = int(np.argmax(power))
    far = np.abs(spectrum.frequencies - spectrum.frequencies[peak]) > 0.05
    print(window, f"{power[far].max() / power[peak]:.0e}")
# -> none 1e-02
# -> cosine_sq 1e-02
# -> hann 3e-05
# -> blackman 1e-05
```

`hann` and `blackman` go to zero at both ends and suppress the leakage by three
orders of magnitude; Blackman harder, at the price of broader peaks.
`cosine_sq`, the default of every transform on these pages, is one at $t = 0$
and zero at the last lag: it tapers only the tail, which is the end a one-sided
correlation function needs tapered, and keeps the zero-lag value that Hann and
Blackman would destroy. It does nothing for a signal that oscillates at full
amplitude to the end, as this one does — which a decayed correlation function
does not.

The cost of any window is resolution: tapering throws away information at the
end, which broadens every peak. Window when your correlation function is
truncated before it decays; when it has genuinely decayed, a window only
broadens peaks for nothing.

## The frequency axis is in radians per unit of `dt`

A `OneSidedSpectrum` carries `frequencies`, the unscaled `real` and `imag` bins
of the transform, and `n_fft`, the padded length. The frequencies are **angular**,
in radians per unit of the `dt` you passed — rad fs⁻¹ for a femtosecond frame
spacing — running from 0 to the Nyquist frequency $\pi/\Delta t$:

```python
spectrum = Spectrum("hann", 1).compute(signal=c, dt=10.0)
print(spectrum.frequencies.shape, round(float(spectrum.frequencies.max()), 3))
# -> (129,) 0.314
```

To reach the wavenumbers a spectroscopist quotes, divide by $2\pi c$:

```python
from molpy.core import UnitRegistry, constants

units = UnitRegistry()
c_cm_per_fs = constants.SPEED_OF_LIGHT * units.factor("m/s", "cm/fs")
wavenumbers = spectrum.frequencies / (2 * np.pi * c_cm_per_fs)   # rad/fs -> cm^-1
print(round(float(wavenumbers.max())))                           # -> 1668
```

1668 cm⁻¹ is the Nyquist limit $16678/(\Delta t/\mathrm{fs})$ at 10 fs. If a
computed spectrum comes out a factor of $2\pi$ or $c$ from where you expect,
this conversion is the first place to look. The bins are unscaled on purpose:
the prefactor of a power spectrum, a continuous transform or a cross-section is
physics that belongs to the quantity, so it is written in your script (or owned
by `RamanSpectrum` and the [dielectric](dielectric.md) routes), not hidden in
the transform.

## The VDOS is a velocity correlation, transformed

Build the correlation function first. `Vacf` takes the velocities as a
`(n_frames, n_dof)` array — every Cartesian component of every atom is one
degree of freedom, so an `(n_frames, n_atoms, 3)` array is reshaped — and
averages the mean-subtracted, unbiased autocorrelation over them:

```python
from molpy.compute import Vacf

rng = np.random.default_rng(0)
velocities = rng.normal(0.0, 0.02, size=(1024, 32, 3))

vacf = Vacf(256).compute(velocities=velocities.reshape(1024, -1), dt=0.5)
vdos = Spectrum("cosine_sq", 4).compute(signal=vacf.channel(0), dt=0.5)

nu = vdos.frequencies / (2 * np.pi * c_cm_per_fs)
print(vdos.n_fft, round(float(nu.max())))                 # -> 2048 33356
```

33 356 cm⁻¹ is exactly $16678/0.5$ — the Nyquist limit, confirming that the
$\Delta t$ you passed is the one you meant. The intensity is `vdos.real`, the
cosine transform of the curve; the resolution is set by the correlation length,
$1/(cT)$ for the 256 lags × 0.5 fs here, not by the zero padding, which only
interpolates the grid.

## Infrared is the current correlation of the dipole

Infrared takes the same transform, but the correlation must be built from the
**dipole flux** rather than velocities:

$$
\boxed{\;I_\mathrm{IR}(\omega) \propto Q(\omega)\,\frac{\beta}{3cV}
\int_{-\infty}^{\infty}
\big\langle \dot{\mathbf{M}}(0)\cdot\dot{\mathbf{M}}(t)\big\rangle
\,e^{-i\omega t}\,\mathrm{d}t\;}
$$

where $\mathbf{M} = \sum_i q_i\mathbf{r}_i$ for a fixed-charge force field, or
comes from [Voronoi integration](voronoi.md) of the electron density for *ab
initio* MD. Using the flux $\dot{\mathbf M}$ rather than $\mathbf M$ is
equivalent up to factors of $\omega^2$ and much better behaved numerically.

Each step is a compute. `DipoleMoment` sums $q_i\mathbf{r}_i$ over the atoms —
of the whole system, or of each group of a partition — for every frame;
`Derivative` differentiates the series in time; `CurrentAcf` is the
autocorrelation of a `(n_frames, 3)` current, summed over its Cartesian
components. From charges and unwrapped positions:

```python
from molpy.compute import CurrentAcf, Derivative, DipoleMoment

charges = rng.normal(0.0, 0.4, size=32)
positions = np.cumsum(rng.normal(0.0, 0.01, size=(1024, 32, 3)), axis=0)

dipole = DipoleMoment().compute(positions=positions, charges=charges).series(0)
flux = Derivative("central").compute(series=dipole, dt=0.5)
print(dipole.shape, flux.shape)                           # -> (1024, 3) (1022, 3)

flux_acf = CurrentAcf(256).compute(current=flux, dt=0.5)
ir = Spectrum("cosine_sq", 4).compute(signal=flux_acf.channel(0), dt=0.5)
print(flux_acf.values.shape, ir.real.shape)               # -> (257, 1) (1025,)
```

The central difference has no neighbour at either end, so the flux is two
frames shorter than the dipole. The positions here are a random walk, so this
spectrum means nothing physically — but the *shape* of the pipeline is the part
that is hard to guess. `CurrentAcf` keeps the mean of the current (a net
current is physics); `normalization="none"` selects the raw lagged sum of the
older convention instead of the unbiased estimator, which rescales the spectrum
and leaves its shape alone.

## Raman needs two correlations, isotropic and anisotropic

Raman scattering separates into an isotropic part (from the trace of the
polarizability tensor, which measures how much the molecule's overall
polarizability breathes) and an anisotropic part (from what is left after
removing that trace, which measures how much its *shape* changes).
`PolarizabilityParts` splits a `(n_frames, 6)` polarizability series in Voigt
order `[xx, yy, zz, xy, xz, yz]` into the mean polarizability, one column, and a
six-component anisotropy vector whose squared norm is the invariant $\gamma^2$.
Both parts are linear in $\alpha$, so they commute with the time derivative, and
each Raman correlation is a plain product correlation of one part's flux:

```python
from molpy.compute import PolarizabilityParts, RamanSpectrum

alpha = rng.normal(size=(1024, 6))                         # your α(t), Voigt order
parts = PolarizabilityParts().compute(alpha=alpha)
rate = Derivative("central")
correlate = Correlator("product", 256)

iso_dot = rate.compute(series=parts.isotropic, dt=0.5)
aniso_dot = rate.compute(series=parts.anisotropic, dt=0.5)
iso = correlate.compute(a=iso_dot[:, None, :], dt=0.5)
aniso = correlate.compute(a=aniso_dot[:, None, :], dt=0.5)

raman = RamanSpectrum(averaged=True).compute(
    isotropic=iso, anisotropic=aniso, temperature=300.0
)
print(raman.isotropic.shape, raman.parallel is not None)  # -> (1025,) True
```

`RamanSpectrum` owns the transform and the prefactors: it reports wavenumbers
directly (`frequencies_cm1`), applies the Bose factor at the `temperature` you
pass, and the scattering cross-section $(\tilde\nu_0-\tilde\nu)^4/\tilde\nu$
when configured with the laser's `incident_frequency`. With `averaged=True` it
also returns `parallel` and `perpendicular`, the two scattering polarizations
an experiment measures. Their ratio $I_\perp/I_\parallel$ is the
**depolarization ratio**, bounded above by 3/4; a mode at exactly 3/4 is
depolarized (non-totally-symmetric) and one well below is polarized. Computing
it is a cheap check that the isotropic and anisotropic correlations were not
swapped.

## The chiral spectra are cross-correlations

Vibrational circular dichroism correlates the electric dipole flux against the
magnetic one, $\langle\dot{\boldsymbol\mu}(0)\cdot\dot{\mathbf m}(t)\rangle$.
There is no named surface for it: it is `Correlator` with the product kernel
and a second series `b`, then the same `Spectrum`:

```python
mu = rng.normal(size=(1024, 3))                            # your μ(t)
m = rng.normal(size=(1024, 3))                             # your m(t)
mu_dot = rate.compute(series=mu, dt=0.5)
m_dot = rate.compute(series=m, dt=0.5)

vcd_corr = correlate.compute(a=mu_dot[:, None, :], b=m_dot[:, None, :], dt=0.5)
vcd = Spectrum("cosine_sq", 4).compute(signal=vcd_corr.channel(0), dt=0.5)
print(vcd_corr.values.shape)                              # -> (257, 1)
```

Raman optical activity is the same move inside the Raman family: the parts of
$\dot\alpha$ cross-correlated against the parts of $\dot G'$ (`a=` and `b=`),
handed to `RamanSpectrum` as its isotropic and anisotropic inputs. Resonance
Raman is `RamanSpectrum` on the autocorrelations of a resonant polarizability
$\alpha(\omega_L)$. Both chiral spectra are **signed**, so peaks point up or
down according to the handedness of the mode, and both need far longer
trajectories than their parent spectra because the signal is orders of
magnitude smaller.

!!! note "No IR or Raman figure yet — TODO"
    Only the VDOS above is shown with real data, because it needs nothing but
    velocities. An honest IR or Raman figure needs a molecular trajectory with
    charges — or, for *ab initio* intensities, an electron density to partition
    — and the reference system behind these pages is monatomic argon, whose
    dipole is identically zero. Add these when a molecular trajectory exists
    under `scripts/docs_series/`.

## When it goes wrong

**Peaks appear at frequencies where the molecule has no modes.**
Aliasing. Your $\Delta t$ puts real high-frequency modes above Nyquist and they
fold back. Check $16678/(\Delta t/\text{fs})$ against your highest expected
band before believing any peak.

**The spectrum is covered in regular ripples.**
Spectral leakage: the correlation was truncated before it decayed. Use a
tapering window in `Spectrum` (`"hann"` or `"blackman"` when even the default
`"cosine_sq"` is not enough) — at the cost of broader peaks.

**Everything is one broad hump.**
Resolution. $1/(cT)$ is larger than the splittings you are trying to see;
correlate to longer lags.

**Intensities are far too small at high frequency.**
The quantum correction $Q(\omega)$ has not been applied.

**A huge spike at zero frequency.**
Centre-of-mass drift, which is a non-decaying component in the correlation
function. Remove net momentum before dumping.

**The IR spectrum is empty.**
Your system is neutral and non-polar, or you built $\mathbf{M}$ with unsigned
charges so everything cancelled. Check $\sum_i q_i\mathbf{r}_i$ is not
identically zero.

**Peak positions are systematically high.**
Classical MD on a harmonic surface is a known offender, but check the mundane
causes first: constrained bonds (SHAKE/LINCS) remove modes entirely, and a
thermostat that is too aggressive shifts and broadens everything.

## Check yourself

- Compute the VDOS first, whatever spectrum you actually want. If a band is
  missing there, it is missing from the trajectory, not from the selection rule.
- Confirm the largest wavenumber equals $16678/(\Delta t/\text{fs})$.
- Correlate white noise: the correlation is a spike at lag 0 and nothing else,
  and its spectrum is flat.
- Halve the frame spacing and recompute. Peak *positions* must not move; if they do, you
  were aliasing.
- Integrate the VDOS and compare with the zero-lag value of the VACF. The transform conserves the
  zero-lag value, so the integral is $\langle v^2\rangle$ — which grows with
  temperature. If you want a temperature-independent check, normalize the ACF by
  its own first element before transforming, and then the integral is fixed.

## References

- M. Thomas, M. Brehm, R. Fligg, P. Vöhringer, B. Kirchner, *Phys. Chem. Chem.
  Phys.* **15**, 6608 (2013) — computing IR, Raman, VCD and ROA from MD
  correlation functions; the reference this implementation follows.
- R. Ramírez, T. López-Ciudad, P. Kumar P, D. Marx, *J. Chem. Phys.* **121**,
  3973 (2004) — quantum correction factors and what they do to intensities.
- D. A. McQuarrie, *Statistical Mechanics*, Harper & Row (1976), ch. 21 — the
  correlation-function formulation of spectroscopy.
- N. Wiener, *Acta Math.* **55**, 117 (1930); A. Khinchin, *Math. Ann.* **109**,
  604 (1934) — the theorem behind FFT correlation.
- F. J. Harris, *Proc. IEEE* **66**, 51 (1978) — windows and spectral leakage,
  still the standard reference.
- J. G. Dickey, A. Paskin, *Phys. Rev.* **188**, 1407 (1969) — the Raman
  prefactors.

## See also

- [VACF](vacf.md) — the velocity correlation the VDOS transforms
- [Dielectric](dielectric.md) — the same machinery at far lower frequency
- [Voronoi](voronoi.md) — where *ab initio* dipoles come from
- [API reference](../api/compute.md)
