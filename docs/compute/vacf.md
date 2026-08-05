# VACF

This page introduces the **velocity autocorrelation function (VACF)** — the
memory function of particle motion — and the two things it buys you: the
**Green–Kubo route** to the diffusion coefficient and the **vibrational density
of states (VDOS)**. Companion pages: [Diffusion & Ionic Transport](msd.md)
(displacement picture) and [Spectra](spectra.md)
(velocity → spectrum generalised to dipoles and polarizabilities).

The practical pipeline uses `molpy.compute.signal.acf_fft` for the raw
correlation, `CumulativeTrapezoid` for the Green–Kubo integral, and
`PowerSpectrum` for the VDOS. The same compose pattern as
[transport](msd.md) applies: raw curve → Fit / spectrum → physical scale.

!!! note "Conventions used throughout"
    - Velocity of atom $i$ at time $t$: $\mathbf{v}_i(t)$.
    - $\langle\cdots\rangle_{i,t}$ averages over particles and time origins.
    - Frame spacing $\Delta t$ in **fs** (LAMMPS *real*). Spectra take `dt_fs`
    in femtoseconds; $D$ inherits length²/time from your velocities
    (Å/fs → Å²·fs⁻¹).
    - $d = 3$; Green–Kubo factor $1/d = 1/3$.
    - VACF normalization is **unbiased**: $C(\tau)$ uses $1/(n-\tau)$ origins
    (not $1/n$).

---

## 1. What the VACF measures

$$
C_{vv}(\tau) = \big\langle\, \mathbf{v}_i(t)\cdot\mathbf{v}_i(t+\tau)\,\big\rangle_{i,t},
\qquad
\hat{C}_{vv}(\tau) = \frac{C_{vv}(\tau)}{C_{vv}(0)}.
$$

Zero lag is fixed by equipartition: $C_{vv}(0) = 3 k_B T / m$ per particle —
a free thermometer.

### 1.1 Ballistic, caged, and free motion

At very short lag $\tau\to 0$, every particle moves as if free:

$$
C_{vv}(\tau)\;\approx\; C_{vv}(0)\,\Bigl(1 - \tfrac12\omega_E^2\tau^2 + \cdots\Bigr),
$$

so $C_{vv}$ starts with zero slope and a *negative* curvature fixed by the
Einstein frequency $\omega_E$ of the liquid. In a dense fluid the first-shell
"cage" reverses the velocity of the tagged particle: $C_{vv}$ develops a
**negative lobe**, then decays to zero. A dilute gas decays almost
monoexponentially; a solid oscillates without a long-time plateau of the
running integral of $C_{vv}$.



<figure id="fig-vacf" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:9"
mark:
  type: line
  strokeWidth: 2.2
  interpolate: monotone
data:
  values:
    - {t: 0, c: 1.0}
    - {t: 0.05, c: 0.7}
    - {t: 0.1, c: 0.2}
    - {t: 0.15, c: -0.15}
    - {t: 0.25, c: -0.05}
    - {t: 0.4, c: 0.02}
    - {t: 0.6, c: 0.0}
    - {t: 1.0, c: 0.0}
encoding:
  x:
    field: t
    type: quantitative
    title: τ (ps)
  y:
    field: c
    type: quantitative
    scale: {zero: false}
    title: Ĉ_vv(τ)
  color:
    value: "#0284c7"
```

</div>

**Figure 1.** Schematic normalised VACF of a dense liquid: ballistic decay, cage-induced negative lobe, then relaxation to zero.
</figure>

---

## 2. The Green–Kubo route to the diffusion coefficient

Linear response equates the self-diffusion coefficient to the time integral of
the VACF (Green, 1954; Kubo, 1957):

$$
\boxed{\;D = \frac{1}{d}\int_0^\infty \big\langle \mathbf{v}(0)\cdot\mathbf{v}(t)\big\rangle\, dt\;}
$$

with $d=3$ in bulk. The Einstein relation $D=\lim_{t\to\infty}\mathrm{MSD}(t)/(2d\,t)$
is mathematically equivalent; the two routes are a mutual consistency check.

### 2.1 Running integral and the plateau

Define the cumulative integral

$$
D(\tau) = \frac{1}{3}\int_0^\tau C_{vv}(t)\,\mathrm{d}t.
$$

$D(\tau)$ rises during the ballistic/caged regime and must **plateau** once
$C_{vv}$ has decayed. Quote that plateau — never $D$ at the last lag, where noise
and residual COM drift re-accumulate. If there is no plateau, the trajectory is
too short, the lag window is wrong, or drift has not been removed.

### 2.2 When to prefer VACF vs MSD

| Prefer VACF / GK | Prefer MSD / Einstein |
|---|---|
| Velocities dumped finely (≤ few fs) | Only positions available |
| You need VDOS from the same ACF | Coarse dumps (10–100 ps) |
| Diagnosing cage physics | Robust $D$ with long trajectories |

---

## 3. The VDOS: vibrations from velocities

The Wiener–Khinchin theorem turns the VACF into a power spectrum — the
**vibrational density of states**:

$$
\boxed{\;g(\omega) \;\propto\; \int_{-\infty}^{\infty} \big\langle \mathbf{v}(0)\cdot\mathbf{v}(t)\big\rangle\, e^{-i\omega t}\, dt\;}
$$

No optical selection rules: every mode the atoms execute appears. The
$\omega\to 0$ weight is proportional to $D$ (liquids); a solid (after COM
removal) has $g(0)=0$. IR/Raman/VCD/ROA in [spectra.md](spectra.md) replace
velocity with dipole or polarizability fluxes and add the corresponding
prefactors.

---

## 4. Computing the VACF

Build a velocity array `(n_frames, n_entities, n_components)`, form the ACF,
then integrate for $D$ or FFT for the VDOS. Compose **raw ACF → Fit /
spectrum** — never bake $D$ into the raw curve.

Two equivalent public routes:

1. **`Acf`** — multi-entity vector series `(T, N, 3)`, unbiased origin average,
   returns `.acf` / `.lags`.
2. **`signal.acf_fft`** — single 1-D series (e.g. one DOF or a pre-averaged
   scalar flux).

```python
import numpy as np
from molpy.compute import Acf, signal, CumulativeTrapezoid, PowerSpectrum

rng = np.random.default_rng(0)
# (n_frames, n_atoms, 3) — preferred multi-particle layout for Acf
velocities = np.ascontiguousarray(rng.standard_normal((128, 64, 3)), dtype=np.float64)
dt_fs = 10.0

# Route A: multi-entity Acf (recommended for particle VACF)
vacf = Acf().compute(velocities, max_lag=32)
C = vacf.acf  # shape (max_lag + 1,)

# Route B: scalar series via FFT helper (e.g. one collective flux)
C_scalar = signal.acf_fft(
    np.ascontiguousarray(velocities.mean(axis=(1, 2))),
    max_lag=32,
)

integ = CumulativeTrapezoid().fit(C, dt=dt_fs)["integral"]
D_running = integ / 3.0

vdos = PowerSpectrum()(C, dt_fs=dt_fs)  # frequency in cm⁻¹
```

Cross-check Green–Kubo $D$ against the Einstein route in
[Diffusion & Transport](msd.md) (`MSD` + `LinearFit`).

Do **not** average atoms first and then correlate if you want the per-particle
VACF — that measures COM memory, not $\langle\mathbf{v}_i\cdot\mathbf{v}_i\rangle$.
Use `Acf` on the full `(T, N, 3)` array instead.

---

## 5. Parameters

| Parameter | Where | Meaning |
|---|---|---|
| `series` | `Acf().compute` | `(n_frames, n_entities, n_components)`; units set $C_{vv}$ and $D$ units |
| `max_lag` | `Acf` / `signal.acf_fft` | longest lag in frames (clamped to $T-1$) |
| `dt` | `CumulativeTrapezoid.fit` | frame spacing, **fs** |
| `dt_fs` | `PowerSpectrum` | same spacing, fs → frequency axis in cm⁻¹ |

---

## 6. Hyperparameter effects

- **Sampling rate.** Nyquist
 $\tilde{\nu}_\text{max} \approx 16678/(\Delta t/\text{fs})$ cm⁻¹. At 0.5 fs you
 resolve all molecular vibrations; at 10 fs stretches alias.
- **Maximum lag.** Sets GK integration window and spectral resolution. Too
 short: no $D(\tau)$ plateau; too long: noise. Default 5–10× visible decay of
 $\hat{C}_{vv}$.
- **Trajectory length.** Error $\sim 1/\sqrt{T_\text{traj}$.
- **Thermostat.** Strong Langevin reshapes the VACF — sample in **NVE** (or
 weak thermostat) after equilibration.
- **COM drift.** Non-decaying offset → divergent $\int C$ and $\omega=0$ VDOS
 spike. Remove COM motion before dumping velocities.

---

## 7. Reading the results

| Check | Expectation | If violated |
|---|---|---|
| $C_{vv}(0)$ | $3k_BT/m$ | wrong units / mass / dump |
| Negative dip (liquids) | ~0.1–0.5 ps | sampling too coarse |
| $\hat{C}_{vv}(\tau\to\infty)$ | → 0 | COM drift |
| $D(\tau)$ | plateau | lag too short or drift |
| GK $D$ vs MSD $D$ | agree | fit window / unwrapping on MSD side |
| VDOS at $\omega=0$ | $\propto D$; 0 for solids | drift spike |

---

## 8. Pitfalls checklist

1. Sampling velocities too coarsely (VACF decays ~10× faster than MSD linearises).
2. Quoting $D$ at the last lag instead of the plateau.
3. Thermostat contamination of $\mathbf{v}(t)$.
4. COM drift not removed.
5. Collective ACF (atoms averaged first) mistaken for per-particle VACF.
6. Unit slips: Å/fs → $D$ in Å²·fs⁻¹ (1 Å²·fs⁻¹ = 0.1 cm²·s⁻¹).
7. `max_lag` exceeding usable origins ($T-1$).

---

## 9. References

- M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed. (2017).
- D. Frenkel, B. Smit, *Understanding Molecular Simulation*, 2nd ed. (2002), §4.4.
- D. A. McQuarrie, *Statistical Mechanics*, Harper & Row (1976), ch. 21.
- J.-P. Hansen, I. R. McDonald, *Theory of Simple Liquids*, 4th ed., ch. 7.

[^green]: M. S. Green, *J. Chem. Phys.* **22**, 398 (1954).
[^kubo]: R. Kubo, *J. Phys. Soc. Jpn.* **12**, 570 (1957).
[^dickey]: J. M. Dickey, A. Paskin, *Phys. Rev.* **188**, 1407 (1969).

## See also

- [Diffusion & Ionic Transport](msd.md)
- [Spectra](spectra.md)
- [Compute overview](index.md)
- [API reference: Compute](../api/compute.md)
