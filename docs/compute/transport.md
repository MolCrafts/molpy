# Diffusion & Ionic Transport

This page is a self-contained, textbook-style introduction to how MolPy / molrs
turn an equilibrium MD trajectory into **transport coefficients** — diffusion,
Onsager coefficients, and ionic conductivity. It starts from the random walk and
builds up to the collective correlation functions used in modern electrolyte
analysis.

It is the conceptual companion to [Dielectric Spectroscopy](dielectric.md): that
page derives the *frequency-dependent* response; this page focuses on
*diffusion* and the *displacement* picture.

**molrs is the single source of truth.** MolPy re-exports the same Compute types
identity-style. There is no parallel recipe layer (`IonicConductivity`,
trajectory-scanning recipe classes). You assemble collective arrays, call
a **raw Compute**, then **compose** a Fit and an SI scale.

!!! note "Conventions used throughout"
    - Position of atom $i$ at time $t$: $\mathbf{r}_i(t)$; displacement over lag
      $\tau$: $\Delta\mathbf{r}_i(\tau) = \mathbf{r}_i(t+\tau) - \mathbf{r}_i(t)$.
    - $\langle\cdots\rangle_t$ denotes an average over **time origins** $t$.
    - Units (LAMMPS *real*): length **Å**, **time fs**, charge $e$, volume Å³,
      temperature K. Diffusion comes out in Å²·fs⁻¹ (or convert); conductivity
      in S·m⁻¹ after the SI prefactor.
    - $d = 3$ spatial dimensions; the Einstein factor $1/(2d) = 1/6$.
    - Displacement kernels need **unwrapped** coordinates (no MIC inside the MSD
      kernel).

---

## 0. Compose pattern (read this first)

| Route | Raw Compute | Fit | Scale (your script) |
|-------|-------------|-----|---------------------|
| Self-diffusion (Einstein) | `MSD(method="window")` or `EinsteinDiffusion` | `LinearFit` | $D = \mathrm{slope}/(2d)$ |
| Self-diffusion (Green–Kubo) | `VACF` / `GreenKuboDiffusion` | `CumulativeTrapezoid` | $D = (1/d)\int C_{vv}$ |
| Onsager $L_{ij}$ | `Onsager.correlation(P_i, P_j, …)` | `LinearFit` on $L(\tau)$ | $\Omega_{ij} = \mathrm{slope}/(6 k_B T V N_A)$ |
| $\sigma$ Einstein | `EinsteinConductivity` | `LinearFit` | $\sigma = \mathrm{slope}/(6 V k_B T)\times$ SI |
| $\sigma$ Green–Kubo | `GreenKuboConductivity` | `CumulativeTrapezoid` | $\sigma = \int C/(3 V k_B T)\times$ SI |

---

## 1. The random walk and the Einstein relation

A particle in a liquid is kicked around by its neighbours. Over a short time it
moves *ballistically*, but after many uncorrelated kicks its motion becomes a
**random walk**. The natural measure of that spread is the **mean-squared
displacement (MSD)**:

$$
\mathrm{MSD}(\tau) = \big\langle\,|\mathbf{r}_i(t+\tau) - \mathbf{r}_i(t)|^2\,\big\rangle_{i,t}.
$$

For a diffusive process the MSD grows **linearly** in time, and the slope defines
the **self-diffusion coefficient** $D$ (Einstein, 1905):

$$
\boxed{\;D = \lim_{\tau\to\infty}\frac{1}{2d\,\tau}\,\mathrm{MSD}(\tau)
       = \lim_{\tau\to\infty}\frac{1}{6\tau}\big\langle|\Delta\mathbf{r}(\tau)|^2\big\rangle\;}
$$

### 1.1 The three regimes

- **Ballistic** (short $\tau$): $\mathrm{MSD}\propto\tau^2$.
- **Diffusive** (intermediate $\tau$): $\mathrm{MSD}\propto\tau$ — **fit the slope here**.
- **Noisy** (long $\tau$): few time origins remain.

Raw Computes return the **full correlation curve** so you can inspect it before
fitting. Place the linear window yourself with `LinearFit(start_frac, end_frac)`.

### 1.2 Averaging over time origins

$$
\mathrm{MSD}(\tau) = \frac{1}{N_\text{origins}}\sum_{t} |\mathbf{r}(t+\tau)-\mathbf{r}(t)|^2,
\qquad N_\text{origins} = N_\text{frames}-\tau.
$$

Use `MSD(method="window")` for this average.

### 1.3 Minimum-image unwrapping (the usual trap)

Periodic images must be unwrapped **before** displacement MSD. Crossing a box
face by $L$ is a real continuous path, not a jump. Unwrap with successive
minimum-image steps (or dump unwrapped coordinates from the engine). molrs MSD /
Einstein kernels do **not** apply MIC inside the lag loop.

---

## 2. Self vs distinct diffusion

A single diffusion coefficient hides multi-component coupling. The **mean
displacement correlation** generalises the MSD:

**Self** — ordinary MSD of one species → $D^\mathrm{s}_\alpha$.

**Distinct** — cross-correlation of displacements of species $\alpha$ and
$\beta$ → $D^\mathrm{d}_{\alpha\beta}$ (collective).

MolPy evaluates the distinct term as the collective cross-correlation
$\big\langle(\sum_i\Delta\mathbf{r}_i)\cdot(\sum_j\Delta\mathbf{r}_j)\big\rangle$
— the form that feeds Onsager coefficients. For normalized $\Omega_{\alpha\beta}$
use §3.

```python
import numpy as np
from molpy.compute import MSD, Onsager

rng = np.random.default_rng(0)
# Collective unwrapped coordinates P_alpha: (n_frames, 3)
P_cat = np.ascontiguousarray(np.cumsum(rng.normal(0, 0.01, size=(40, 3)), axis=0))
P_an = np.ascontiguousarray(np.cumsum(rng.normal(0, 0.01, size=(40, 3)), axis=0))
L11 = Onsager.correlation(P_cat, P_cat, dt=10.0, max_correlation_time=10)
L12 = Onsager.correlation(P_cat, P_an, dt=10.0, max_correlation_time=10)
L11["lag_times"], L11["correlation"]  # L_11(tau)

# Single-particle MSD needs a frame list with unwrapped positions
import molpy as mp
frames = []
for step in range(12):
    xyz = rng.uniform(0.0, 10.0, size=(20, 3)) + 0.05 * step
    f = mp.Frame()
    f["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    f.box = mp.Box.cubic(10.0)
    frames.append(f)
series = MSD(method="window")(frames)
series.mean  # <|r(tau+t)-r(tau)|^2>
```

---

## 3. Onsager phenomenological coefficients

Define the **collective coordinate** of a species (summed unwrapped positions),

$$
\mathbf{P}_\alpha(t) = \sum_{i\in\alpha}\mathbf{r}_i(t),
\qquad
\Delta\mathbf{P}_\alpha(\tau) = \mathbf{P}_\alpha(t+\tau)-\mathbf{P}_\alpha(t).
$$

$$
L_{\alpha\beta}(\tau) = \big\langle\,\Delta\mathbf{P}_\alpha(\tau)\cdot\Delta\mathbf{P}_\beta(\tau)\,\big\rangle_t,
\qquad
\boxed{\;\Omega_{\alpha\beta} = \lim_{\tau\to\infty}\frac{L_{\alpha\beta}(\tau)}{6\,k_B T\,V\,N_A\,\tau}\;}
$$

- Diagonal $\Omega_{\alpha\alpha}$: collective MSD of species $\alpha$.
- Off-diagonal: cation–anion coupling (negative → ion pairing signature).

`Onsager.correlation` returns the raw $L_{\alpha\beta}(\tau)$ curve only — take
the long-time slope with `LinearFit` and apply the prefactor yourself.

### 3.1 From Onsager coefficients to conductivity

$$
\sigma = \frac{e^2}{V k_B T}\sum_{\alpha\beta} z_\alpha z_\beta\,\Omega_{\alpha\beta}.
$$

If off-diagonal terms vanish this collapses to the **Nernst–Einstein** estimate
from self-diffusion alone. The ratio $\sigma/\sigma_\text{NE}$ (ionicity / Haven
ratio) measures correlation suppression of conduction.

---

## 4. Ionic conductivity: two equivalent routes

Both routes start from **pre-assembled** charge-weighted series. There is no
`cation_type=` / trajectory-scanning recipe in the public API.

### 4.1 Einstein route — polarization / charge-dipole MSD

Build the collective charge displacement (translational dipole) of the ions
from **unwrapped** positions,

$$
\mathbf{M}(t) = \sum_a q_a\,\mathbf{r}_a(t),
$$

then

$$
\mathrm{MSD}_M(\tau) = \big\langle|\mathbf{M}(t+\tau)-\mathbf{M}(t)|^2\big\rangle_t,
\qquad
\sigma = \lim_{\tau\to\infty}\frac{1}{6\,V k_B T}\,\frac{d}{d\tau}\,\mathrm{MSD}_M(\tau).
$$

```python
import numpy as np
from molpy.compute import EinsteinConductivity, LinearFit

rng = np.random.default_rng(1)
# M: (n_frames, 3) — sum q_i * r_i_unwrapped; dt in fs
M = np.ascontiguousarray(np.cumsum(rng.normal(0, 0.02, size=(60, 3)), axis=0))
raw = EinsteinConductivity().compute(M, dt=10.0, max_correlation_time=20)
lags, msd = raw["lag_times"], raw["msd"]
fit = LinearFit(start_frac=0.1, end_frac=0.5).fit(lags, msd)
slope = fit["slope"]  # e^2 Å^2 / fs
# sigma = slope / (6 * V * k_B * T) * SI_prefactor
```

Full SI bookkeeping is in
[Dielectric §7](dielectric.md#7-ionic-conductivity-einsteinhelfand).

### 4.2 Green–Kubo route — current autocorrelation

$$
\mathbf{J}(t)=\sum_a q_a\mathbf{v}_a(t),
\qquad
\boxed{\;\sigma = \frac{1}{3\,V k_B T}\int_0^\infty \big\langle\mathbf{J}(0)\cdot\mathbf{J}(t)\big\rangle\,dt\;}
$$

```python
import numpy as np
from molpy.compute import GreenKuboConductivity, CumulativeTrapezoid

rng = np.random.default_rng(2)
# J: (n_frames, 3) — sum q_i * v_i; velocities in Å/fs for real units
J = np.ascontiguousarray(rng.normal(0, 1.0, size=(60, 3)))
raw = GreenKuboConductivity().compute(J, dt=10.0, max_correlation_time=20)
C = raw["jacf"]  # <J(0)·J(t)>
running = CumulativeTrapezoid().fit(C, dt=10.0)["integral"]
# Quote sigma at the plateau of running / (3 V k_B T) * SI_prefactor
```

The Einstein and Green–Kubo routes are mathematically identical; Einstein is
often more robust at coarse
sampling, while the current ACF exposes memory and integral convergence.
Frequency-dependent $\sigma(\omega)$ is in
[Dielectric §8](dielectric.md#8-spectrum-route-ii-greenkubo-current-autocorrelation).

The single-particle Green–Kubo route to $D$ via the VACF is in
[Velocity Autocorrelation & VDOS](vacf.md).

---

## 5. Reading the results

| Quantity | Compute | Physical meaning |
|---|---|---|
| $\mathrm{MSD}(\tau)$ / $D^\mathrm{s}$ | `MSD(method="window")` + `LinearFit` | single-particle diffusion |
| $L_{\alpha\beta}(\tau)$ / $\Omega_{\alpha\beta}$ | `Onsager.correlation` + `LinearFit` | coupled transport; ion pairing |
| $\mathrm{MSD}_M(\tau)$ | `EinsteinConductivity` | collective charge transport |
| $\sigma$ (Einstein) | + `LinearFit` + SI scale | DC conductivity, S/m |
| $C(\tau)=\langle J\cdot J\rangle$ | `GreenKuboConductivity` | current memory |
| $\sigma$ (Green–Kubo) | + `CumulativeTrapezoid` + SI scale | DC conductivity, S/m |

**Cross-checks.** Einstein and Green–Kubo $\sigma$ must agree within statistics.
Nernst–Einstein (from self-MSD) should exceed the correlated conductivity when
ion pairing is significant — their ratio is the ionicity.

---

## 6. Parameters and hyperparameters

### 6.1 Parameters

| Parameter | Where | Meaning |
|---|---|---|
| `dt` | all transport Computes / Fits | frame spacing, **fs** (capture interval, not MD timestep) |
| `max_correlation_time` | Einstein / GK / Onsager / Persist | longest lag **in frames** (clamped to $N-1$) |
| `start_frac`, `end_frac` | `LinearFit` | linear-fit window as fractions of the lag axis (typical `0.1`, `0.5`) |
| `p_i`, `p_j` | `Onsager.correlation` | collective coordinates `(n_frames, 3)`, unwrapped |
| `translational_dipole` / `current` | Einstein / GK `.compute` | `M(t)` or `J(t)` as `(n_frames, 3)` |

Raw curves do **not** include volume normalization or SI conversion — that is
intentional so the fit window stays under your eyes.

### 6.2 Hyperparameter effects

- **Fit window.** Too early → ballistic bias; too late → few origins, high
  variance. Report sensitivity when carriers are few.
- **`max_correlation_time` vs trajectory length.** Keep lags ≪ run length
  (rule of thumb $\le N_\text{frames}/5$).
- **Frame spacing.** Einstein/MSD is robust at coarse `dt`. Green–Kubo current
  ACF decays in ~0.1–1 ps — under-resolving it biases $\sigma$.
- **Dimensionality.** Prefactors assume $d=3$; use $d=2$ factors for quasi-2-D.
- **COM drift.** Collective quantities are one realization per species; remove
  net COM motion before quoting off-diagonal Onsager terms or $\sigma$.
- **$T$ / $V$.** $\sigma \propto 1/(V T)$ — use production-run averages for NPT.
- **GK integration limit.** Quote $\sigma$ at the plateau of the running
  integral, never the last lag.

---

## 7. Pitfalls checklist

1. **No unwrapping** → boundary crossings inject $L$-sized jumps; every MSD is
   garbage.
2. **Fitting outside the diffusive window** → ballistic head or noisy tail.
3. **Too few carriers / short trajectory** → collective curves are noisy; report
   a range.
4. **Wrong velocity / time units** → current must match $e\cdot$Å·fs⁻¹ with
   `dt` in fs for real-unit SI factors; a unit slip rescales $\sigma$ linearly.
5. **Ignoring distinct diffusion** → Nernst–Einstein alone overestimates $\sigma$
   when ions pair.
6. **Expecting a trajectory-scanning recipe** — species selection and charge
   weighting are **your** preprocessing; the Compute only sees arrays.

---

## 8. References

- A. Einstein, *Ann. Phys.* **322**, 549 (1905).
- M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed. (2017).
- J.-P. Hansen, I. R. McDonald, *Theory of Simple Liquids*, 4th ed.
- D. Frenkel, B. Smit, *Understanding Molecular Simulation*, 2nd ed. (2002), §4.4.
- L. Onsager, *Phys. Rev.* **37**, 405 (1931); **38**, 2265 (1931).

[^gudla]: H. Gudla, Y. Shao et al., *J. Phys. Chem. Lett.* **12**, 8460 (2021) —
    distinct diffusion with persistence to extract pairing transport.

## See also

- [Dielectric Spectroscopy](dielectric.md) — $\varepsilon^*(\omega)$ and full
  conductivity derivations.
- [Pair Persistence](persistence.md) — residence times and pairing.
- [Velocity Autocorrelation & VDOS](vacf.md) — Green–Kubo $D$ via VACF.
- [Compute overview](index.md) — patterns and catalogue.
- [API reference: Compute](../api/compute.md).
