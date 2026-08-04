# Velocity Autocorrelation & Vibrational Density of States

This page introduces the **velocity autocorrelation function (VACF)** — the
memory function of particle motion — and the two things it buys you: the
**Green–Kubo route** to the diffusion coefficient and the **vibrational density
of states (VDOS)**. Companion pages: [Diffusion & Ionic Transport](transport.md)
(displacement picture) and [Vibrational Spectra from MD](spectra.md)
(velocity → spectrum generalised to dipoles and polarizabilities).

The raw VACF Compute and the spectral Fits live in **molrs**. MolPy re-exports
`PowerSpectrum` and `signal` helpers; import `VACF` /
`GreenKuboDiffusion` from `molrs.compute.transport` (SSOT).

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
\hat{C}_{vv}(\tau) = \frac{C_{vv}(\tau)}{C_{vv}(0)} .
$$

Zero lag is fixed by equipartition: $C_{vv}(0) = 3 k_B T / m$ per particle —
a free thermometer. Shape classifies the phase (gas: exponential; liquid: cage
dip / negative lobe; solid: oscillation, running integral → 0).

---

## 2. The Green–Kubo route to the diffusion coefficient

$$
\boxed{\;D = \frac{1}{3}\int_0^\infty \big\langle \mathbf{v}(0)\cdot\mathbf{v}(t)\big\rangle\, dt\;}
$$

Einstein (MSD slope) and Green–Kubo (VACF integral) are mathematically
identical. In practice: MSD is robust at coarse sampling; VACF needs finely
sampled velocities but exposes *why* $D$ is what it is.

Quote the **plateau** of the running integral
$D(\tau) = \frac{1}{3}\int_0^\tau C_{vv}(t)\,dt$, never the value at the largest lag.

---

## 3. The VDOS: vibrations from velocities

$$
\boxed{\;g(\omega) \;\propto\; \int_{-\infty}^{\infty} \big\langle \mathbf{v}(0)\cdot\mathbf{v}(t)\big\rangle\, e^{-i\omega t}\, dt\;}
$$

No selection rules — every mode the atoms execute appears. IR/Raman/VCD/ROA in
[spectra.md](spectra.md) replace velocity with dipole/polarizability fluxes.

---

## 4. Computing the VACF

### 4.1 Preferred: molrs `VACF` / `GreenKuboDiffusion`

```python
import numpy as np
from molrs.compute.transport import VACF, GreenKuboDiffusion
from molpy.compute import CumulativeTrapezoid, PowerSpectrum

rng = np.random.default_rng(0)
# VACF takes (n_frames, n_dof) — flatten atoms × xyz
velocities = np.ascontiguousarray(rng.standard_normal((128, 64 * 3)), dtype=np.float64)
dt_fs = 10.0
resolution = 32

raw = VACF().compute(velocities, dt=dt_fs, resolution=resolution)
# GreenKuboDiffusion is the same raw curve under the diffusion name:
# raw = GreenKuboDiffusion().compute(velocities, dt=dt_fs, resolution=resolution)

C = raw["acf"]  # unnormalized C_vv(tau); raw["lag_times"] in fs
integ = CumulativeTrapezoid().fit(C, dt=dt_fs)["integral"]
D_running = integ / 3.0

vdos = PowerSpectrum()(C, dt_fs=dt_fs)  # frequency cm^-1, intensity
```

`GreenKuboDiffusion` returns the same raw ACF family as `VACF`; $D = (1/d)\int$
is always a separate Fit + scale step — never baked into the raw Compute.

### 4.2 FFT ACF helper (single series)

For a single scalar time series, use `molpy.compute.signal.acf_fft` (molrs
kernel, $O(N\log N)$):

```python
import numpy as np
from molpy.compute import signal, PowerSpectrum

rng = np.random.default_rng(0)
v_series = np.ascontiguousarray(rng.standard_normal(256), dtype=np.float64)
dt_fs = 10.0
acf = signal.acf_fft(v_series, 64)
vdos = PowerSpectrum()(acf, dt_fs=dt_fs)
```

Do **not** average atoms first and then correlate if you want the per-particle
VACF — that measures COM memory, not $\langle\mathbf{v}_i\cdot\mathbf{v}_i\rangle$.

---

## 5. Parameters

| Parameter | Where | Meaning |
|---|---|---|
| `velocities` | `VACF.compute` | `(n_frames, n_atoms, 3)`; units set $C_{vv}$ and $D$ units |
| `dt` | VACF / Fits | frame spacing, **fs** |
| `resolution` | VACF | curve length in frames (max lag = resolution − 1 or as documented) |
| `dt_fs` | `PowerSpectrum` | same spacing, fs → frequency axis in cm⁻¹ |

---

## 6. Hyperparameter effects

- **Sampling rate.** Nyquist
  $\tilde{\nu}_\text{max} \approx 16678/(\Delta t/\text{fs})$ cm⁻¹. At 0.5 fs you
  resolve all molecular vibrations; at 10 fs stretches alias.
- **Maximum lag.** Sets GK integration window and spectral resolution. Too
  short: no $D(\tau)$ plateau; too long: noise. Default 5–10× visible decay of
  $\hat{C}_{vv}$.
- **Trajectory length.** Error $\sim 1/\sqrt{T_\text{traj}}$.
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
7. `resolution` / lag exceeding usable origins.

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

- [Diffusion & Ionic Transport](transport.md)
- [Vibrational Spectra from MD](spectra.md)
- [Compute overview](index.md)
- [API reference: Compute](../api/compute.md)
