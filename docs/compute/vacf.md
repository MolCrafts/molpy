# Velocity Autocorrelation & Vibrational Density of States

This page is a self-contained, textbook-style introduction to the **velocity
autocorrelation function (VACF)** — the memory function of particle motion — and
the two things it buys you: the **Green–Kubo route** to the diffusion
coefficient and the **vibrational density of states (VDOS)**. It is the
time-domain companion to [Diffusion & Ionic Transport](transport.md) (the
displacement picture of the *same* physics) and the entry point to
[Vibrational Spectra from MD](spectra.md) (which generalises the
velocity → spectrum recipe to dipoles and polarizabilities). No background
beyond undergraduate statistical mechanics is assumed.

The ACF accumulation itself is a streaming NumPy kernel
([`compute_acf`](../api/compute.md), a rolling-cache
direct correlator adapted from the *tame* library); the spectral transform
(`PowerSpectrum` and the whole [spectra.md](spectra.md) family) runs in Rust
(`molrs`), as does the FFT correlator behind the collective analyzers.

!!! note "Conventions used throughout"
    - Velocity of atom $i$ at time $t$: $\mathbf{v}_i(t)$, components
      $(v_x, v_y, v_z)$.
    - $\langle\cdots\rangle_{i,t}$ averages over **particles** $i$ and **time
      origins** $t$ (equilibrium dynamics are stationary — every frame is an
      origin).
    - Frame spacing $\Delta t$ (the *capture* interval, not necessarily the MD
      timestep). Spectra take it in femtoseconds (`dt_fs`); diffusion constants
      inherit whatever length²/time your velocities carry (LAMMPS *real*:
      Å²·fs⁻¹ from Å/fs, Å²·ps⁻¹ from Å/ps).
    - $d = 3$ spatial dimensions; the Green–Kubo factor is $1/d = 1/3$.

---

## 1. What the VACF measures

The VACF is the simplest dynamical correlation function:

$$
C_{vv}(\tau) = \big\langle\, \mathbf{v}_i(t)\cdot\mathbf{v}_i(t+\tau)\,\big\rangle_{i,t},
\qquad
\hat{C}_{vv}(\tau) = \frac{C_{vv}(\tau)}{C_{vv}(0)} .
$$

It asks: *how much does a particle remember its velocity a time $\tau$ later?*
The zero-lag value is fixed by equipartition,

$$
C_{vv}(0) = \big\langle |\mathbf{v}|^2 \big\rangle = \frac{3 k_B T}{m},
$$

which makes $C_{vv}(0)$ a free thermometer — if it does not reproduce your
thermostat temperature, your units or your velocity dump are wrong before any
physics enters.

The *shape* of the decay classifies the phase:

- **Dilute gas** — velocities decorrelate by uncorrelated binary collisions:
  a featureless, near-exponential decay to zero.
- **Liquid** — after the initial (ballistic) decay the particle bounces off its
  neighbour cage: $C_{vv}$ crosses **negative** (backscattering) before healing
  to zero. The negative dip is the *cage effect* in its rawest form.
- **Solid** — the particle rattles in a nearly harmonic well: $C_{vv}$
  **oscillates** at the lattice/molecular vibrational frequencies and decays
  only through anharmonicity, and its running integral (→ §2) tends to zero —
  no diffusion.

---

## 2. The Green–Kubo route to the diffusion coefficient

Diffusion is accumulated velocity memory. Writing the displacement as the
integral of the velocity and expanding the MSD (see
[transport.md §1](transport.md#1-the-random-walk-and-the-einstein-relation))
gives the **Green–Kubo relation**[^green][^kubo]:

$$
\boxed{\;D = \frac{1}{3}\int_0^\infty \big\langle \mathbf{v}(0)\cdot\mathbf{v}(t)\big\rangle\, dt\;}
$$

The Einstein/MSD slope and the Green–Kubo integral are **mathematically
identical** — one differentiates the MSD, the other integrates its second
derivative. In practice they are complementary:

- **Einstein (MSD)** is robust at coarse time resolution — use it when frames
  are widely spaced (see [transport.md](transport.md)).
- **Green–Kubo (VACF)** needs finely sampled velocities (the ACF decays in
  ~0.1–1 ps for liquids), but exposes *why* $D$ is what it is: the cage dip
  eats diffusivity; the long-time tail feeds it.

The practical estimator is the **running integral**
$D(\tau) = \frac{1}{3}\int_0^\tau C_{vv}(t)\,dt$, which must reach a plateau;
quote the plateau, never the value at your largest lag.

---

## 3. The VDOS: vibrations from velocities

The Fourier transform of the VACF is the **vibrational density of states**
(power spectrum)[^dickey]:

$$
\boxed{\;g(\omega) \;\propto\; \int_{-\infty}^{\infty} \big\langle \mathbf{v}(0)\cdot\mathbf{v}(t)\big\rangle\, e^{-i\omega t}\, dt\;}
$$

Every vibrational mode the atoms actually execute shows up — IR-active or not,
Raman-active or not — because velocities carry no selection rules. That makes
the VDOS the *reference map* of a system's dynamics: harmonic modes appear as
peaks at their frequencies, diffusion as a $\omega \to 0$ intensity
($g(0) \propto D$), and anharmonicity as peak broadening. The IR, Raman, VCD
and ROA spectra of [spectra.md](spectra.md) are the same construction with the
velocity replaced by dipole/polarizability fluxes (which *do* carry selection
rules).

---

## 4. Computing the VACF in MolPy

The per-particle ACF kernel is [`compute_acf`](../api/compute.md):
it takes a `(n_frames, n_particles, 3)` velocity array, dots each particle's
velocity with its own lagged velocity, and averages over particles and time
origins — exactly the $\langle\mathbf{v}_i(0)\cdot\mathbf{v}_i(\tau)\rangle_{i,t}$
above, **unnormalized** (index 0 is $\langle v^2\rangle$).

```python
import numpy as np
from molpy.compute import PowerSpectrum, compute_acf

# velocities: (n_frames, n_atoms, 3), sampled every dt
vacf = compute_acf(velocities, cache_size=4096)   # raw <v(0)·v(t)>, one value per lag

D = np.trapezoid(vacf, dx=dt) / 3.0               # Green–Kubo D (plateau-check first!)
D_running = np.cumsum(vacf) * dt / 3.0            # running integral D(tau)

vdos = PowerSpectrum()(vacf, dt_fs=dt_fs)         # -> {frequency (cm^-1), intensity}
```

`PowerSpectrum` applies the one-sided FFT and spectral prefactor in `molrs`;
the same object feeds the whole [spectra.md](spectra.md) family.

---

## 5. Parameters and their meaning

| Parameter | Where | Meaning |
|---|---|---|
| `data` (velocities) | `compute_acf` | `(n_frames, n_particles, 3)` array; units set the units of $C_{vv}$ and $D$ |
| `cache_size` | `compute_acf` | curve length **in frames**: the returned array covers lags $0 \dots \text{cache\_size}-1$ (max lag = `cache_size − 1`) |
| `dropnan` | `compute_acf` | NaN policy for ragged/partial data (`"partial"` default) |
| `dt` / `dt_fs` | your bookkeeping / `PowerSpectrum` | frame spacing; converts lags to time and sets the frequency axis |

---

## 6. Hyperparameter effects

- **Frame spacing $\Delta t$ (sampling rate).** The spectrum is cut off at the
  Nyquist frequency $\tilde{\nu}_\text{max} = 1/(2c\,\Delta t) \approx
  16678/(\Delta t/\text{fs})$ cm⁻¹. At $\Delta t = 0.5$ fs you resolve up to
  ~33 000 cm⁻¹ (all molecular vibrations); at $\Delta t = 10$ fs everything
  above ~1700 cm⁻¹ is *aliased* — C–H and O–H stretches fold back onto false
  low frequencies. Choose $\Delta t$ from the stiffest mode you care about,
  not from disk budget.
- **Maximum lag (`cache_size`).** Sets both the Green–Kubo integration window
  and the spectral resolution $\Delta\tilde{\nu} \approx
  33356/(\tau_\text{max}/\text{fs})$ cm⁻¹. Too short: the $D(\tau)$ plateau is
  never reached and VDOS peaks are artificially broadened. Too long: the tail
  is pure noise — the statistical error of an ACF estimate grows like
  $\sqrt{\tau/T_\text{traj}}$ as the number of independent origins shrinks —
  and integrating noise makes $D$ drift. A good default is 5–10× the visible
  decay time of $\hat{C}_{vv}$.
- **Trajectory length $T_\text{traj}$.** Averaging over time origins means the
  relative error at fixed lag scales as $1/\sqrt{T_\text{traj}}$; doubling the
  run beats doubling the lag window.
- **Thermostat coupling.** Strong stochastic thermostats (tight Langevin /
  aggressive velocity rescaling) inject friction and noise directly into
  $\mathbf{v}(t)$ — they *reshape* the VACF, damping the cage dip and shifting
  VDOS peaks. Sample velocities in **NVE** (or with a very weakly coupled
  thermostat) after equilibrating.
- **Mean subtraction / drift.** A net centre-of-mass drift adds a constant
  offset to $C_{vv}$ that never decays and makes the Green–Kubo integral
  diverge linearly. Remove COM motion before dumping velocities.
- **Cost.** `compute_acf` is a *direct* streaming correlator: each new frame
  is dotted against the `cache_size` frames held in a rolling cache, so time
  scales as $O(n_\text{frames} \times \text{cache\_size} \times
  n_\text{atoms})$ and memory stays at
  $O(\text{cache\_size} \times n_\text{atoms})$, independent of trajectory
  length. Keep `cache_size` modest (§ above) — it is a *time* knob as well as
  a physics knob. The $O(N\log N)$ FFT correlator (`molrs.signal.acf_fft`)
  is what the collective analyzers (`ACFAnalyzer`, [spectra.md](spectra.md))
  use; it needs the whole series in memory at once.

---

## 7. Reading the results

| Check | Expectation | Diagnosis if violated |
|---|---|---|
| $C_{vv}(0)$ | $3k_BT/m$ per particle (equipartition) | wrong units, wrong mass, or velocities not what you think |
| Negative dip (liquids) | shallow minimum at ~0.1–0.5 ps | none in a liquid → sampling too coarse |
| $\hat{C}_{vv}(\tau\to\infty)$ | decays to 0 | constant offset → COM drift |
| $D(\tau)$ running integral | plateau at intermediate $\tau$ | no plateau → lag window too short or drift |
| $D$ (Green–Kubo) vs $D$ (MSD slope) | agree within statistics | disagreement → fit window or unwrapping problem on the MSD side |
| VDOS at $\omega=0$ | $\propto D$; zero for solids | spurious zero-frequency spike → drift |

---

## 8. Pitfalls checklist

1. **Sampling velocities too coarsely** → aliased spectra and a VACF that
   misses the cage dip entirely. The VACF decays ~10× faster than the MSD
   becomes linear.
2. **Quoting $D$ at the last lag instead of the plateau** → integrating tail
   noise; always plot $D(\tau)$.
3. **Thermostat contamination** → Langevin friction visibly damps
   $\hat{C}_{vv}$; use NVE for production VACF/VDOS runs.
4. **COM drift not removed** → non-decaying ACF offset, divergent Green–Kubo
   integral, $\omega = 0$ spike in the VDOS.
5. **Using a collective ACF where a per-particle one is meant** — an analyzer
   that averages atoms *first* and then correlates (e.g. `ACFAnalyzer`, built
   for collective signals like the total dipole) measures
   $\langle\bar{\mathbf{v}}(0)\cdot\bar{\mathbf{v}}(t)\rangle$, which is COM
   memory, not the VACF. Use `compute_acf` (per-particle dot product) for
   VACF/VDOS.
6. **Unit slips in $D$** — Å/fs velocities give $D$ in Å²·fs⁻¹
   (1 Å²·fs⁻¹ = 0.1 cm²·s⁻¹), Å/ps give Å²·ps⁻¹ (= 10⁻⁴ cm²·s⁻¹); state your
   units next to every number.
7. **`cache_size` exceeding usable origins** — lags near `n_frames` average
   over almost nothing; cap the lag at a small fraction of the trajectory.

---

## 9. References

- M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.
  (2017), §2.7, §8.5 — time-correlation functions and transport coefficients.
- D. Frenkel, B. Smit, *Understanding Molecular Simulation*, 2nd ed. (2002),
  §4.4 — Green–Kubo vs Einstein estimators.
- D. A. McQuarrie, *Statistical Mechanics*, Harper & Row (1976), ch. 21 —
  spectral densities as Fourier transforms of TCFs.
- J.-P. Hansen, I. R. McDonald, *Theory of Simple Liquids*, 4th ed., ch. 7 —
  the velocity autocorrelation function and the cage effect.

[^green]: M. S. Green, *J. Chem. Phys.* **22**, 398 (1954) — transport
    coefficients from time-correlation functions.
[^kubo]: R. Kubo, *J. Phys. Soc. Jpn.* **12**, 570 (1957) — the
    fluctuation–dissipation formalism behind all Green–Kubo relations.
[^dickey]: J. M. Dickey, A. Paskin, *Phys. Rev.* **188**, 1407 (1969) — phonon
    spectra from the velocity autocorrelation function in MD.

## See also

- [Diffusion & Ionic Transport](transport.md) — the Einstein/MSD route to $D$
  and the collective (current) Green–Kubo conductivity.
- [Vibrational Spectra from MD](spectra.md) — IR, Raman, VCD, ROA: the same
  ACF → spectrum machinery with selection-rule-carrying fluxes.
- [Dielectric Spectroscopy](dielectric.md) — frequency-dependent response and
  the spectral estimators in depth.
- [Compute overview](index.md) — the Compute → Result pattern.
- [API reference: Compute](../api/compute.md).
