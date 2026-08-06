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

| Spectrum | $A$ — the quantity light couples to | Compute |
|---|---|---|
| VDOS | atomic velocities | `PowerSpectrum` |
| Infrared | dipole flux $\dot{\mathbf{M}}$ | `IRSpectrum` |
| Raman | polarizability (iso + aniso) | `RamanSpectrum` |
| VCD | electric ⊗ magnetic dipole | `VcdSpectrum` |
| ROA | ROA invariants | `RoaSpectrum` |
| Resonance Raman | resonant polarizability | `ResonanceRamanSpectrum` |

So the physics of "which modes are visible" is not in the algorithm at all — it
is in the choice of $A$. A mode that does not modulate the dipole contributes
nothing to $\langle\dot{\mathbf M}(0)\cdot\dot{\mathbf M}(t)\rangle$ and is
therefore infrared-silent. That is the familiar selection rule, arriving here as
a property of a correlation function rather than as a symmetry table — and
unlike the symmetry table, it comes with anharmonicity and temperature already
included.

These computes do **not** take frames. They take a correlation function you have
already built and the frame spacing in fs, and they return frequencies in
wavenumbers.

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
data: {$file: data/spectra/argon_vdos.json}
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

**Figure 1.** Vibrational density of states of liquid argon at 85 K, from the
velocity autocorrelation of the trajectory used on the [VACF](vacf.md) page. The
band peaks at 18 cm⁻¹ and has died away by 150 cm⁻¹.
</figure>

Two features carry the physics. The spectrum is **non-zero at zero frequency** —
0.81 of the peak height — because $g(0) \propto D$ and a liquid diffuses; a
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

None of the computes here apply it, because none of them takes a temperature.
Multiplying `intensities` by $Q(\omega)$ is a line in your own script, and
whether you did it belongs in your methods section.

## Computing them

Build the correlation function first. For a VDOS, one scalar ACF per degree of
freedom, averaged:

```python
import numpy as np
from molpy.compute import Acf, PowerSpectrum

rng = np.random.default_rng(0)
velocities = np.ascontiguousarray(rng.normal(0.0, 0.02, size=(1024, 32, 3)))

vacf = np.asarray(Acf().compute(velocities, max_lag=256).acf)
vdos = PowerSpectrum()(vacf, dt_fs=0.5)

print(sorted(vdos))
# -> ['frequencies_cm1', 'intensities', 'n_frames', 'resolution']
print(round(float(np.asarray(vdos["frequencies_cm1"]).max())))   # -> 33356
```

33 356 cm⁻¹ is exactly $16678/0.5$ — the Nyquist limit, confirming the
$\Delta t$ you passed is the one you meant. Read `resolution` as a **count** of
lags, not a frequency; the real resolution is $1/(cT)$ as above.

Infrared takes the same shape of input — an ACF and `dt_fs` — but the ACF must be
built from the **dipole flux** rather than velocities:

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

Building that ACF is your job, and it is the step the API cannot do for you,
so here it is explicitly. From charges and unwrapped positions:

```python
charges = rng.normal(0.0, 0.4, size=32)
positions = np.cumsum(rng.normal(0.0, 0.01, size=(1024, 32, 3)), axis=0)

dipole = np.einsum("i,tij->tj", charges, positions)       # M(t), shape (T, 3)
flux = np.gradient(dipole, 0.5, axis=0)                   # dM/dt, dt_fs = 0.5
print(dipole.shape, flux.shape)                           # -> (1024, 3) (1024, 3)
```

`flux` is a single 3-vector per frame, so correlate it as one entity —
`(T, 1, 3)` — rather than as 32 separate particles:

```python
from molpy.compute import IRSpectrum

flux_acf = np.asarray(Acf().compute(
    np.ascontiguousarray(flux[:, None, :]), max_lag=256
).acf)
ir = IRSpectrum()(flux_acf, dt_fs=0.5)
print(sorted(ir))
# -> ['frequencies_cm1', 'intensities', 'n_frames', 'resolution']
```

The positions here are a random walk, so this spectrum means nothing physically
— but the *shape* of the pipeline is the part that is hard to guess, and it is
the same for Raman with a polarizability series in place of the dipole.

Raman needs **two** correlation functions, because scattering separates into an
isotropic part (from the trace of the polarizability tensor, which measures how
much the molecule's overall polarizability breathes) and an anisotropic part
(from what is left after removing that trace, which measures how much its
*shape* changes). It therefore returns four spectra:

```python
from molpy.compute import RamanSpectrum

raman = RamanSpectrum()(vacf, vacf, dt_fs=0.5)   # (acf_iso, acf_aniso)
print(sorted(raman))
# -> ['anisotropic', 'frequencies_cm1', 'isotropic', 'n_frames',
#     'parallel', 'perpendicular', 'resolution']
```

`parallel` and `perpendicular` are the two scattering polarizations an
experiment measures. Their ratio $I_\perp/I_\parallel$ is the **depolarization
ratio**, and it is bounded above by 3/4; a mode at exactly 3/4 is
depolarized (non-totally-symmetric) and one well below is polarized. Computing
it is a cheap check that the isotropic and anisotropic ACFs were not swapped.

`VcdSpectrum` (electric ⊗ magnetic dipole cross-correlation) and `RoaSpectrum`
follow the same patterns: VCD takes one ACF, ROA takes the isotropic and
anisotropic pair. Both chiral spectra are **signed**, so peaks point up or down
according to the handedness of the mode, and both need far longer trajectories
than their parent spectra because the signal is orders of magnitude smaller.

!!! note "No IR or Raman figure yet — TODO"
    Only the VDOS above is shown with real data, because it needs nothing but
    velocities. An honest IR or Raman figure needs a molecular trajectory with
    charges — or, for *ab initio* intensities, an electron density to partition
    — and the reference system behind these pages is monatomic argon, whose
    dipole is identically zero. Add these when a molecular trajectory exists
    under `scripts/docs_data/`.

## When it goes wrong

**Peaks appear at frequencies where the molecule has no modes.**
Aliasing. Your $\Delta t$ puts real high-frequency modes above Nyquist and they
fold back. Check $16678/(\Delta t/\text{fs})$ against your highest expected
band before believing any peak.

**The spectrum is covered in regular ripples.**
Spectral leakage: the ACF was truncated before it decayed. Apply a window with
[`signal.apply_window`](signal.md) — at the cost of broader peaks.

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
- Confirm `frequencies_cm1.max()` equals $16678/(\Delta t/\text{fs})$.
- Halve `dt_fs` and recompute. Peak *positions* must not move; if they do, you
  were aliasing.
- Integrate the VDOS and compare with `vacf[0]`. The transform conserves the
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

## See also

- [VACF](vacf.md) — the velocity correlation the VDOS transforms
- [Signal](signal.md) — windows, the frequency grid, and `acf_fft`
- [Dielectric](dielectric.md) — the same machinery at far lower frequency
- [Voronoi](voronoi.md) — where *ab initio* dipoles come from
- [API reference](../api/compute.md)
