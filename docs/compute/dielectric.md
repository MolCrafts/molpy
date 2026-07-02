# Dielectric Spectroscopy

This page is a self-contained, textbook-style derivation of how MolPy computes
the **frequency-dependent dielectric permittivity** $\varepsilon^*(\omega)$ and
the **ionic conductivity** $\sigma$ of a polar liquid or electrolyte from an
equilibrium molecular-dynamics (MD) trajectory. Every prefactor in the code is
explained, every formula is derived or motivated, and the whole pipeline is
illustrated on one running example. No prior exposure to linear-response theory
is assumed beyond undergraduate electromagnetism, a little statistical
mechanics, and the Fourier transform.

All spectral physics — autocorrelation, windowing, FFT, prefactors — runs in
Rust inside `molrs`. The MolPy layer (`molpy.compute.DielectricSusceptibility`)
only extracts positions and charges, unwraps coordinates, and assembles the
dipole series before delegating to `molrs.dielectric`.

!!! note "Conventions used throughout"
    - Complex permittivity: $\varepsilon^*(\omega) = \varepsilon'(\omega) - i\,\varepsilon''(\omega)$ (**positive-loss** convention, so $\varepsilon'' \ge 0$).
    - Fourier transform: $X(\omega) = \int_0^\infty f(t)\,e^{-i\omega t}\,dt$.
    - Units (LAMMPS *real*): length Å, charge $e$, time ps, volume Å³, temperature K, angular frequency rad·ps⁻¹. GROMACS trajectories are read in native **nm**, so scale lengths by 10 before handing them to the kernels.

---

## The running example

| Quantity | Value |
|---|---|
| Solvent | 852 SPC water molecules (each O, H, H point charges) |
| Ions | 16 Na⁺ + 16 Cl⁻ (32 charge carriers, ≈ 1 mol/L) |
| Atoms | 852×3 + 32 = 2588 |
| Ensemble | NVT (canonical) |
| Temperature | 298.15 K |
| Cell | cubic, $L \approx 2.996$ nm, $V \approx 2.69\times10^4$ Å³ |
| Length / step | 20 ns, 1 fs timestep |
| Output | positions **and** velocities every 10 fs |
| Charges | water O = −0.82 e, H = +0.41 e; Na = +1 e, Cl = −1 e |

Two properties of this system drive every methodological choice below:

1. **It has both orientational polarization (water rotation) and translational
   current (ion diffusion).** These must be treated *separately* — feeding the
   whole-system dipole into a static-permittivity formula yields a nonsensical
   $\varepsilon \approx 7000$ (see [§5](#5-the-electrolyte-step-decompose-the-dipole)).
2. **It stores velocities**, so both the dipole-autocorrelation route and the
   current-autocorrelation route are available and can cross-check each other.

---

## 1. Physical picture: what does a dielectric spectrum measure?

### 1.1 Polarization and permittivity

Place a dielectric in a field $\mathbf{E}$ and its charges rearrange — molecular
dipoles reorient, electron clouds shift — producing a **polarization**
$\mathbf{P}$ (dipole moment per unit volume). In the linear regime,

$$
\mathbf{P} = \varepsilon_0\,(\varepsilon - 1)\,\mathbf{E}.
$$

A larger relative permittivity $\varepsilon$ means the medium is more easily
polarized and screens fields more strongly. Real water has $\varepsilon \approx
78$; the SPC model used here gives $\varepsilon \approx 54$ — **a known feature
of the SPC force field, not an error.**

### 1.2 Why a *spectrum* — frequency dependence

Under an oscillating field $\mathbf{E}(t) = \mathbf{E}_0\cos(\omega t)$, whether
the polarization keeps up depends on frequency:

- **Low $\omega$**: molecules have time to reorient → full polarization → large $\varepsilon$.
- **High $\omega$**: molecules cannot follow → only fast electronic polarization remains → $\varepsilon \to \varepsilon_\infty$.
- **Intermediate $\omega$**: polarization lags the field, energy is dissipated as heat → an **absorption peak**.

Hence permittivity becomes a **complex, frequency-dependent** quantity:

$$
\boxed{\;\varepsilon^*(\omega) = \varepsilon'(\omega) - i\,\varepsilon''(\omega)\;}
$$

- $\varepsilon'(\omega)$ (real part) — energy storage; high at low $\omega$, decays to $\varepsilon_\infty$.
- $\varepsilon''(\omega)$ (imaginary part) — **loss / absorption**; always $\ge 0$ for a passive, causal system; peaks at the relaxation frequency. (A microwave oven heats water by working near its $\varepsilon''$ peak.)

### 1.3 The fluctuation–dissipation theorem

This is the conceptual key. The **fluctuation–dissipation theorem (FDT)** states:

> A system's *response* to an external perturbation is fully determined by the
> statistics of its *spontaneous equilibrium fluctuations*.

For dielectrics this means we do **not** need to apply an oscillating field.
Observing how the system's **total dipole moment $\mathbf{M}(t)$ fluctuates on
its own** at equilibrium is enough: the autocorrelation of those fluctuations
encodes the entire dielectric response. That is why a plain NVT trajectory
suffices.

---

## 2. Building block: the total dipole moment $\mathbf{M}(t)$

The instantaneous total dipole moment of a frame is the charge-weighted sum of
positions:

$$
\mathbf{M}(t) = \sum_{i=1}^{N_\text{atoms}} q_i\,\mathbf{r}_i(t),
\qquad [\mathbf{M}] = e\cdot\text{Å}.
$$

In code this is the single dot product `compute_dipole_moment`.

### 2.1 A critical detail: minimum-image unwrapping

MD runs in a periodic box. When an atom exits one face it re-enters the
opposite face, so its coordinate jumps by a box length $L$. Computing
$\mathbf{M}$ from such jumping coordinates makes it leap by $q\cdot L$ every time
an atom crosses a boundary, ruining the autocorrelation.

**Fix — accumulate minimum-image displacements:**

$$
\Delta\mathbf{r}_k = \mathbf{r}(t_k) - \mathbf{r}(t_{k-1}),
\qquad
\Delta\mathbf{r}_k \leftarrow \Delta\mathbf{r}_k - L\,\mathrm{round}\!\left(\frac{\Delta\mathbf{r}_k}{L}\right),
$$

$$
\mathbf{r}_\text{unwrap}(t_k) = \mathbf{r}_\text{unwrap}(t_{k-1}) + \Delta\mathbf{r}_k.
$$

The `round` term subtracts off any whole-box jump, leaving the true step (always
shorter than half a box). MolPy performs this with
`Box.delta(p1, p2, minimum_image=True)`.

!!! warning
    This is valid only if the per-frame displacement is $< L/2$. At 10 fs per
    frame an atom moves a fraction of an Å, far below $L/2 \approx 15$ Å — safe.
    This is one reason the trajectory is written densely.

---

## 3. Static permittivity $\varepsilon(0)$

The zero-frequency limit is the simplest number and anchors the low-frequency
end of the whole spectrum.

### 3.1 The Neumann fluctuation formula

For simulations using Ewald / "tin-foil" (conducting) boundary conditions, the
static permittivity is given by the **dipole-moment fluctuation**
(Neumann, 1983):

$$
\boxed{\;\varepsilon(0) = \varepsilon_\infty + \frac{4\pi}{3}\,\frac{1}{V k_B T}\,
\big\langle\,|\mathbf{M} - \langle\mathbf{M}\rangle|^2\,\big\rangle\;}
$$

Term by term:

- $\varepsilon_\infty$ — high-frequency (electronic) permittivity. Use **1** for
  non-polarizable force fields like SPC; 1.5–2.5 for polarizable water.
- $\langle|\mathbf{M}-\langle\mathbf{M}\rangle|^2\rangle$ — the **variance** of
  the total dipole, in $(e\cdot\text{Å})^2$. Bigger fluctuations → larger
  permittivity. This *is* the FDT: large fluctuation = strong response.
- $V k_B T$ — volume × thermal energy scale; normalizes to a per-volume density.
- $4\pi/3$ — geometric factor for an isotropic medium under conducting boundary
  conditions.

### 3.2 Making it dimensionless (real units)

In LAMMPS *real* units the Coulomb constant is not 1, so the code carries
$\kappa = 1/(4\pi\varepsilon_0) = 332.0637\ \text{kcal·Å·mol}^{-1}e^{-2}$:

$$
A_\text{stat} = \frac{4\pi}{3}\cdot\frac{\kappa}{V k_B T},
\qquad k_B = 1.987204\times10^{-3}\ \text{kcal·mol}^{-1}\text{K}^{-1}.
$$

Useful identity (used by the Green–Kubo route): $1/\varepsilon_0 = 4\pi\kappa$.

### 3.3 Numerical care: two-pass variance

Computing $\langle M^2\rangle - \langle M\rangle^2$ directly suffers
catastrophic cancellation when $|\mathbf{M}| \gg$ its fluctuation (a real hazard
at $\sim10^6$ frames). The code uses a **two-pass centered variance**: first the
mean $\langle\mathbf{M}\rangle$, then
$\tfrac1N\sum|\mathbf{M}-\langle\mathbf{M}\rangle|^2$. Mathematically identical,
numerically stable.

### 3.4 Isotropic vs per-axis

- **Isotropic** $\varepsilon(0)$: the formula above, prefactor $\propto 4\pi/3$.
- **Per-axis** $\varepsilon_d$ ($d=x,y,z$): prefactor is **3×** larger (no
  $1/3$ averaging):
  $\varepsilon_d = \varepsilon_\infty + \frac{4\pi\kappa}{V k_B T}(\langle M_d^2\rangle-\langle M_d\rangle^2)$.
  An isotropic system should give $\varepsilon_x \approx \varepsilon_y \approx
  \varepsilon_z$ averaging back to the scalar value — a built-in self-check.

!!! example "Running example"
    Using the water dipole $\mathbf{M}_D$ (see [§5](#5-the-electrolyte-step-decompose-the-dipole))
    gives $\varepsilon(0) \approx 54$, matching GROMACS `gmx dipoles` on the
    solvent group (54.06) to within 0.01–0.6 %.

---

## 4. Spectrum route I — Einstein–Helfand (dipole autocorrelation)

This is the workhorse for the *entire* $\varepsilon^*(\omega)$ curve.

### 4.1 The core linear-response relation

The Kubo relation of Caillol–Levesque–Weis (1986, their Eq. 30):

$$
\varepsilon^*(\omega) - \varepsilon_\infty
= A\Big[\langle|\delta\mathbf{M}|^2\rangle - i\omega\,X(\omega)\Big],
\qquad A = \frac{4\pi\kappa}{3\,V k_B T},
$$

with $\delta\mathbf{M} = \mathbf{M} - \langle\mathbf{M}\rangle$ and

$$
C(t) = \langle\delta\mathbf{M}(0)\cdot\delta\mathbf{M}(t)\rangle\ \text{(dipole ACF)},
\qquad X(\omega) = \int_0^\infty C(t)\,e^{-i\omega t}\,dt.
$$

In one sentence: **the dielectric spectrum is the Fourier transform of the
dipole-fluctuation autocorrelation function.** The slower $C(t)$ decays (the
longer the dipole "remembers"), the stronger the low-frequency response.

### 4.2 The key trick: transform the ACF *derivative*

Naively forming $i\omega X(\omega)$ blows up numerically: the discrete transform
$X(\omega)$ carries a frequency-independent floor $\sim \Delta t\,C(0)$ from the
$t=0$ sample, so $\omega X(\omega)$ diverges toward the Nyquist frequency and the
loss spectrum $\varepsilon''$ spuriously rises (and dips negative).

The code sidesteps this by **integration by parts**. Since

$$
\int_0^\infty C'(t)\,e^{-i\omega t}\,dt = -C(0) + i\omega X(\omega)
\;\Rightarrow\;
i\omega X(\omega) = C(0) + \widehat{C'}(\omega),
$$

the $A\,C(0) = A\langle|\delta\mathbf{M}|^2\rangle$ term cancels the explicit
$\langle|\delta\mathbf{M}|^2\rangle$, giving the **equivalent but stable** form:

$$
\boxed{\;\varepsilon^*(\omega) - \varepsilon_\infty = -A\,\widehat{C'}(\omega),
\qquad \widehat{C'}(\omega) = \int_0^\infty C'(t)\,e^{-i\omega t}\,dt.\;}
$$

Why stable? $C'(t)$ vanishes at **both** ends — $C'(0)=0$ (the ACF is even, so
its derivative is zero at the origin) and $C'(\infty)=0$ (correlations decay) —
so its transform decays correctly and $\varepsilon''$ stays finite. Split into
real and imaginary parts (positive-loss convention):

$$
\varepsilon'(\omega) - \varepsilon_\infty = -A\,\mathrm{Re}\,\widehat{C'}(\omega),
\qquad
\varepsilon''(\omega) = A\,\mathrm{Im}\,\widehat{C'}(\omega).
$$

### 4.3 Anchoring the DC bin

The discrete derivative transform reproduces $\omega=0$ only to $O(\Delta t)$, so
the code overwrites the DC bin with the **exact Neumann value**:

$$
\varepsilon'(0) = \varepsilon_\infty + A\,\langle|\delta\mathbf{M}|^2\rangle,
\qquad \varepsilon''(0) = 0.
$$

This makes the spectrum's low-frequency endpoint identical to [§3](#3-static-permittivity-varepsilon0)
(enforced by a unit test).

### 4.4 The algorithm in five steps

This is exactly what `einstein_helfand_spectrum` does:

1. **Centered variance** $\langle|\delta\mathbf{M}|^2\rangle$ for the exact DC bin.
2. **Autocorrelation** of the mean-subtracted dipole, summed over $x,y,z$, via the
   FFT (Wiener–Khinchin, [§6.1](#61-autocorrelation-by-fft-wienerkhinchin)) with
   the unbiased estimator $C(k)=r[k]/(N-k)$ ([§6.2](#62-biased-vs-unbiased-estimator)).
3. **Window** with the one-sided cosine² taper $w[k]=\cos^2(\tfrac{\pi k}{2L})$
   — 1 at $C(0)$, 0 at $C(L)$ ([§6.3](#63-windowing-which-window-and-why)).
4. **Differentiate + FFT**: central-difference $C'(t)$, zero-pad to
   $n_\text{pad}=\big(2(L{+}1)\big)$ rounded up to a power of two, forward FFT,
   multiply by $\Delta t$ ([§6.4](#64-discrete-fft-continuous-transform)).
5. **Assemble** $\varepsilon'(\omega), \varepsilon''(\omega)$, overwriting the DC
   bin with the exact static value.

The dipole ACF is strictly one-sided, so this route always applies the cos²
taper regardless of the `window_type` argument.

---

## 5. The electrolyte step: decompose the dipole

This is the most important physical point for an ion-containing system. Feeding
the **whole-system** dipole $\mathbf{M}_\text{tot}$ into the Neumann formula
gives $\varepsilon \approx 7000$ (≈ 7257 measured here). The reason:

> Ions are **free charge carriers**. Their continual diffusion makes
> $\mathbf{M}_\text{tot}$ perform an unbounded random walk whose variance never
> converges. That is not *polarization*; it is **DC conductivity**, and it has no
> business in a static-permittivity formula.

The fix is to split the total dipole by physical origin
(`decompose_current`, or simply by atom slice):

$$
\mathbf{M}_D(t) = \sum_{i\in\text{water}} q_i\mathbf{r}_i(t)\ \text{(rotational)},
\qquad
\mathbf{M}_J(t) = \sum_{i\in\text{ions}} q_i\mathbf{r}_i(t)\ \text{(translational)}.
$$

- **$\mathbf{M}_D$** (water orientational dipole) — a **bounded fluctuation** →
  feed it to the dielectric routes ([§3](#3-static-permittivity-varepsilon0),
  [§4](#4-spectrum-route-i-einsteinhelfand-dipole-autocorrelation)) for
  $\varepsilon(0)$ and $\varepsilon(\omega)$.
- **$\mathbf{M}_J$** (ionic translational dipole) — **diffusive growth** → feed
  it to the Einstein relation ([§7](#7-ionic-conductivity-einsteinhelfand)) for
  the conductivity $\sigma$.

The sum $\mathbf{M}_D + \mathbf{M}_J$ equals the system current to
floating-point precision, so the decomposition is lossless — it merely separates
two physically distinct processes (orientation vs conduction).

---

## 6. Numerical methods (the signal-processing half)

### 6.1 Autocorrelation by FFT (Wiener–Khinchin)

The direct definition $r[k]=\sum_\tau x[\tau]x[\tau+k]$ is $O(N^2)$ —
intractable for $10^6$ frames. The **Wiener–Khinchin theorem** says the
autocorrelation is the inverse transform of the power spectrum:

$$
r = \mathrm{IFFT}\big(|\mathrm{FFT}(x)|^2\big),
$$

reducing the cost to $O(N\log N)$. The signal **must be zero-padded to $\ge 2N$**
(the code uses $(2N)$ rounded up to a power of two); otherwise the FFT returns
the *circular* autocorrelation, whose tail wraps around and contaminates the
small-lag values.

### 6.2 Biased vs unbiased estimator

The FFT yields the **linear, unnormalized** ACF
$r[k]=\sum_{\tau=0}^{N-1-k}x[\tau]x[\tau+k]$. Larger lags average over fewer
pairs ($N-k$ of them), so the **unbiased ensemble estimator** divides by $N-k$:

$$
C(k\,\Delta t) = \frac{r[k]}{N-k} \approx \langle x(0)\,x(k\Delta t)\rangle.
$$

The cost: large lags are noisier — which is why `max_correlation_time` should
not be too large (see [§6.5](#65-frequency-grid-resolution-and-nyquist)).

### 6.3 Windowing: which window, and why

The ACF is truncated at `max_lag`; a hard cutoff causes spectral ringing
(Gibbs). Multiplying by a **window** that smoothly tapers the tail suppresses
it. But the dielectric ACF is **strictly one-sided** ($t \ge 0$) and its $t=0$
value $C(0)=\langle|\delta\mathbf{M}|^2\rangle$ *is* the static signal — it must
not be tapered away.

- ❌ **Hann / Blackman** (symmetric windows) zero out *both* ends, killing
  $C(0)$. Valid only for two-sided data or a current ACF.
- ✅ **One-sided cosine² taper** $w[k]=\cos^2(\tfrac{\pi k}{2L})$ — 1 at $k=0$
  (preserving $C(0)$), 0 at $k=L$ (smooth cutoff). The EH route always uses it.

### 6.4 Discrete FFT → continuous transform

The physics needs the **continuous** transform $X(\omega)=\int_0^\infty C(t)
e^{-i\omega t}dt$; the FFT gives a **discrete** DFT. The rectangle rule
(Numerical Recipes §13.9) bridges them:

$$
\int_0^T f(t)e^{-i\omega t}dt \approx \Delta t \cdot \mathrm{DFT}[f](\omega_k).
$$

Hence the code multiplies the FFT output by $\Delta t$ (not the FFT's internal
$1/n_\text{pad}$). This keeps the final $\varepsilon$ correctly scaled.

### 6.5 Frequency grid, resolution, and Nyquist

With $n_\text{pad} = \big(2(\text{max\_lag}+1)\big)$ rounded up to a power of two:

$$
\text{number of bins} = \frac{n_\text{pad}}{2}+1,
\qquad
\Delta\omega = \frac{2\pi}{n_\text{pad}\,\Delta t}\ \text{(resolution)},
\qquad
\omega_\text{max} = \frac{\pi}{\Delta t}\ \text{(Nyquist)}.
$$

Trade-offs:

- **Larger `max_lag`** → finer $\Delta\omega$ (better low-frequency resolution),
  but noisier tail. Rule of thumb: `max_lag` $\le$ one quarter of the frame count.
- **Smaller $\Delta t$** (denser output) → higher Nyquist → access to
  higher-frequency features (e.g. the librational resonance at tens of rad/ps).
  This is why the example writes a frame every **10 fs**. For static $\varepsilon$
  and slow relaxation you may subsample (e.g. every 200 frames = 2 ps) to save
  memory.

---

## 7. Ionic conductivity (Einstein–Helfand)

The Green–Kubo integral of the current ACF converges poorly at coarse sampling.
For the **ionic conductivity**, the equivalent and more robust route is the
**Einstein–Helfand** relation — the long-time slope of the mean-squared
displacement (MSD) of the translational dipole $\mathbf{M}_J$:

$$
\boxed{\;\sigma = \lim_{t\to\infty}\frac{1}{6\,V k_B T}\,
\frac{d}{dt}\big\langle|\mathbf{M}_J(t)-\mathbf{M}_J(0)|^2\big\rangle\;}
$$

This is exactly the quantity `gmx current` reports as "Einstein–Helfand".

### 7.1 Procedure

1. **Collective MSD**:
   $\text{MSD}(k)=\langle|\mathbf{M}_J(t{+}k)-\mathbf{M}_J(t)|^2\rangle$,
   averaged over all time origins $t$ (`collective_msd`).
2. **Linear fit for the slope**: the MSD is *ballistic* at short times,
   *diffusive* in the middle, and *noisy* at long times. Fit a straight line only
   in the diffusive window $\lbrack\text{fit\_start\_frac},\text{fit\_end\_frac}\rbrack\cdot
   \text{max\_lag}$, typically $[0.1, 0.5]$.
3. **Convert to S/m** (SI constants fold Å, ps into m, s):

$$
\sigma\,[\text{S/m}] = \text{slope}\,[(e\text{Å})^2/\text{ps}]\cdot
\frac{e^2\cdot 10^{-8}}{6\,V[\text{Å}^3]\cdot 10^{-30}\,k_B T},
$$

with $e=1.602\times10^{-19}$ C, $k_B=1.381\times10^{-23}$ J/K; the $10^{-8}$
folds the Å²→m² and ps→s conversions.

### 7.2 An honest caveat: few carriers → uncertain $\sigma$

The example has only **32 ions over 20 ns**. The ion-dipole MSD is mildly
super-diffusive, so $\sigma$ is sensitive to the fit window: it drifts from
≈ 5.8 to ≈ 10 S/m as the window moves from `[50,200]` to `[1000,3000]` ps. **Report
a range, not a single digit.** The default `[100,400]` ps window gives
$\sigma \approx 6.1$ S/m (matching `gmx current`'s 6.12 S/m to 1.2 %); longer
windows approach ≈ 8.5 S/m (the experimental value for 1 M NaCl). Tighter
convergence requires more carriers and longer trajectories.

---

## 8. Spectrum route II — Green–Kubo (current autocorrelation)

The second route starts from the **current** and is the equivalent path for
conducting systems, plus a natural cross-check of route I.

### 8.1 Current density

The current density is the time derivative of the dipole, per volume,
discretized by finite difference:

$$
\mathbf{J}(t) = \frac{\dot{\mathbf{M}}}{V}
\approx \frac{\mathbf{M}(t)-\mathbf{M}(t-\Delta t)}{V\,\Delta t},
\qquad [\mathbf{J}] = e\cdot\text{Å}^{-2}\text{ps}^{-1}.
$$

`compute_current_density` does this. **Row 0 is `NaN`** (no previous frame), and
all consumers must skip it (the Green–Kubo kernel does so internally).

### 8.2 Conductivity spectrum → permittivity

$$
\sigma(\omega) = \frac{V}{3 k_B T}\int_0^\infty \langle\mathbf{J}(0)\cdot\mathbf{J}(t)\rangle e^{-i\omega t}dt
= \sigma'(\omega) + i\sigma''(\omega).
$$

(The prefactor is $V/(3k_BT)$ because the input is current density
$\mathbf{J}=\dot{\mathbf{M}}/V$; for the total $\dot{\mathbf{M}}$ it would be
$1/(3Vk_BT)$, differing by $V^2$ since $\langle\dot M\dot M\rangle=V^2\langle JJ\rangle$.)
Maxwell's relation links conductivity to permittivity:

$$
\boxed{\;\varepsilon^*(\omega) - \varepsilon_\infty = -\frac{i\,\sigma(\omega)}{\omega\,\varepsilon_0}\;}
\;\Rightarrow\;
\varepsilon'(\omega)-\varepsilon_\infty = \frac{\sigma''(\omega)}{\omega\varepsilon_0},
\quad
\varepsilon''(\omega) = \frac{\sigma'(\omega)}{\omega\varepsilon_0},
$$

with $1/\varepsilon_0 = 4\pi\kappa$. The DC bin ($\omega=0$) is the indeterminate
$\sigma/\omega = 0/0$ and is regularized to $(\varepsilon_\infty, 0)$; the true
static value comes from [§3](#3-static-permittivity-varepsilon0) or low-$\omega$
extrapolation. In the Debye limit this route agrees with route I (verified by a
unit test).

---

## 9. End-to-end with MolPy

The high-level `DielectricSusceptibility` compute packages extraction →
unwrapping → dipole assembly → both routes → static $\varepsilon$ in one call;
the physics still runs in `molrs`.

```python
from molpy.compute import DielectricSusceptibility

dc = DielectricSusceptibility(
    dt=0.01,                  # ps between kept frames (10 fs here)
    temperature=298.15,       # K
    max_correlation_time=2000,# frames (sets the resolution; keep <= n_frames/4)
    epsilon_inf=1.0,          # non-polarizable SPC water
    window_type="cosine_sq",
    routes=["einstein-helfand", "green-kubo"],
)
result = dc(trajectory)       # a molpy Trajectory of unwrapped frames

eh = result.results["EH-full"]
# eh.frequency      -> rad/ps
# eh.epsilon_real   -> epsilon'(omega)
# eh.epsilon_imag   -> epsilon''(omega)
# eh.epsilon_static -> epsilon(0) (Neumann), attached to every route

# Debye relaxation time, fitted in NumPy (no SciPy) -- see section 10.1
fit = eh.fit_debye()        # fit.tau (ps), fit.delta_eps, fit.omega_peak
```

The **ionic conductivity** has its own compute, `IonicConductivity`, wrapping
the Einstein-Helfand kernel ([§7](#7-ionic-conductivity-einsteinhelfand)). Pass
it an ion-only trajectory (decomposition by *selection*, [§5](#5-the-electrolyte-step-decompose-the-dipole)):

```python
from molpy.compute import IonicConductivity

sigma = IonicConductivity(
    dt=0.01, temperature=298.15, max_correlation_time=1000,
    fit_start_frac=0.1, fit_end_frac=0.5,
)(ion_trajectory)
# sigma.sigma (S/m), sigma.slope, sigma.msd, sigma.time (lag ps)
```

Each frame's `atoms` block must carry `x, y, z` (Å) and `charge` (e), and a
non-free `Box`. For an electrolyte, build two trajectories (or two dipole series)
restricted to the water and ion atoms as in [§5](#5-the-electrolyte-step-decompose-the-dipole),
run the dielectric routes on the water dipole, and the conductivity on the ion
dipole.

For full manual control, the underlying kernels are callable directly:

| Step | `molrs.dielectric` function |
|------|------------------------------|
| Total / sub-system dipole $\mathbf{M}=\sum q_i\mathbf{r}_i$ | `compute_dipole_moment` |
| Current density $\mathbf{J}=\dot{\mathbf{M}}/V$ | `compute_current_density` |
| Split water/ion current | `decompose_current` |
| Static $\varepsilon(0)$ | `static_dielectric_constant` |
| Spectrum (dipole route) | `einstein_helfand_spectrum` |
| Spectrum (current route) | `green_kubo_spectrum` |
| Ionic conductivity $\sigma$ | `einstein_helfand_conductivity` |

---

## 10. Fitting the spectrum

Curve fitting is application-specific and intentionally **not** part of the
compute kernels — it lives in your analysis script as a `scipy` recipe. The
common physical models follow.

### 10.1 Debye relaxation (single relaxation time)

The simplest polar-liquid model: a single-exponential ACF
$C(t)=C(0)e^{-t/\tau}$, giving

$$
\varepsilon^*(\omega) = \varepsilon_\infty + \frac{\varepsilon(0)-\varepsilon_\infty}{1+i\omega\tau},
$$

$$
\varepsilon'(\omega) = \varepsilon_\infty + \frac{\Delta\varepsilon}{1+(\omega\tau)^2},
\qquad
\varepsilon''(\omega) = \frac{\Delta\varepsilon\,\omega\tau}{1+(\omega\tau)^2},
\qquad
\Delta\varepsilon = \varepsilon(0)-\varepsilon_\infty.
$$

- $\varepsilon'$ steps down from $\varepsilon(0)$ to $\varepsilon_\infty$, inflecting at $\omega\tau=1$.
- $\varepsilon''$ is a peak (symmetric on a log axis) at $\omega_\text{peak}=1/\tau$, height $\Delta\varepsilon/2$.
- **Fastest estimate**: read the loss-peak position → $\tau = 1/\omega_\text{peak}$.
- **In MolPy**: `DielectricResult.fit_debye()` returns $\tau$, $\Delta\varepsilon$,
  and $\omega_\text{peak}$ using only NumPy — $\tau$ is the least-squares slope of
  the exact identity $\varepsilon''/(\varepsilon'-\varepsilon_\infty)=\omega\tau$
  over the rising branch (more robust than a single bin), with a loss-peak
  fallback. `DebyeFit.epsilon(omega)` evaluates the fitted model. SciPy is only
  needed for the broadened/skewed fits below.

!!! example "Running example"
    The water ($\mathbf{M}_D$) loss peak gives $\tau \approx 6.5$ ps (with clean
    10 fs data; coarse 2 ps sampling underestimates it to ≈ 4.7 ps), consistent
    with the known SPC relaxation time.

### 10.2 Non-Debye: Cole–Cole / Cole–Davidson / Havriliak–Negami

Real liquids have a *distribution* of relaxation times, broadening or skewing
the peak. The general **Havriliak–Negami (HN)** model:

$$
\varepsilon^*(\omega) = \varepsilon_\infty + \frac{\Delta\varepsilon}{\big[1+(i\omega\tau)^\alpha\big]^\beta},
\qquad 0<\alpha\le1,\ 0<\beta\le1.
$$

- $\alpha<1,\beta=1$ → **Cole–Cole** (symmetric broadening).
- $\alpha=1,\beta<1$ → **Cole–Davidson** (high-frequency skew).
- $\alpha=\beta=1$ → Debye.

Fit $\varepsilon'$ and $\varepsilon''$ jointly for $(\Delta\varepsilon,\tau,\alpha,\beta,\varepsilon_\infty)$
with `scipy.optimize.curve_fit`.

### 10.3 Multiple processes

Water has several processes (a main relaxation, a fast relaxation, and a
high-frequency librational resonance). Superpose Debye/HN terms:

$$
\varepsilon^*(\omega) = \varepsilon_\infty + \sum_j \frac{\Delta\varepsilon_j}{[1+(i\omega\tau_j)^{\alpha_j}]^{\beta_j}}
+ (\text{librational resonance}).
$$

The **librational (hindered-rotation) resonance** at tens of rad·ps⁻¹ is a true
*resonant absorption* (a damped harmonic oscillator), not a monotonic
relaxation; fit it with a Lorentzian / DHO line shape. Resolving it requires the
dense 10 fs sampling (high Nyquist).

### 10.4 The conductivity contribution (electrolytes)

Even after decomposition, residual DC conductivity lifts the low-frequency
$\varepsilon''$:

$$
\varepsilon''_\text{cond}(\omega) = \frac{\sigma_\text{DC}}{\omega\,\varepsilon_0}.
$$

This is a **$1/\omega$ divergence** (slope −1 on a log–log plot). Either add it
as a fit term or subtract it first. Diagnose it by looking for the −1 slope at
the low-frequency end of $\varepsilon''$.

### 10.5 The conductivity "fit" = MSD slope

The `IonicConductivity` compute does this end-to-end: it fits the
$\mathbf{M}_J$ MSD over the diffusive window $[0.1,0.5]\cdot\text{max\_lag}$
and applies the §7.1 prefactor, returning $\sigma$ in S/m. Always verify
(a) the window lies in the linear diffusive regime, and (b) the window
sensitivity (§7.2).

---

## 11. Reading the results

| Quantity | Source | Physical meaning | Example value |
|---|---|---|---|
| $\varepsilon(0)$ | $\mathbf{M}_D$ fluctuation, Neumann | static permittivity / screening | ≈ 54 (SPC; low is normal) |
| $\varepsilon'(\omega)$ | EH spectrum, real | energy storage; $\varepsilon(0)$→$\varepsilon_\infty$ | 54 → 1 |
| $\varepsilon''(\omega)$ | EH spectrum, imag | loss / absorption; relaxation peak | peak at $\omega\approx1/\tau$ |
| $\tau$ | loss-peak $1/\omega_\text{peak}$ or HN fit | dipole relaxation time | ≈ 6.5 ps |
| librational peak | high-$\omega$ spectrum (dense sampling) | hindered-rotation resonance | tens of rad/ps |
| $\sigma$ | $\mathbf{M}_J$ MSD slope, Einstein–Helfand | DC ionic conductivity | ≈ 6 S/m (range 6–8.5) |

**Cross-checks.** Route I (dipole) and route II (current) must agree in the
Debye limit; the static $\varepsilon(0)$ must be reproduced both by the Neumann
formula and by the DC bin of the EH spectrum (enforced by tests). Disagreements
are almost always bookkeeping — unwrapping, units (nm vs Å, 298.15 vs 300 K),
volume, or the dipole grouping. Historically every discrepancy traced to one of
those, and matched setups agreed to ≈ 1 %.

---

## 12. Pitfalls checklist

1. **No unwrapping** → the dipole jumps by $q\cdot L$; the spectrum is garbage.
2. **Whole-system $\mathbf{M}_\text{tot}$ in the static formula** → $\varepsilon$
   diverges to thousands. Electrolytes *must* decompose ([§5](#5-the-electrolyte-step-decompose-the-dipole)).
3. **Hann/Blackman on the dielectric ACF** → kills $C(0)$. Use cos² ([§6.3](#63-windowing-which-window-and-why)).
4. **Forgetting the $\Delta t$ factor or nm→Å** → wrong magnitude by powers of ten.
5. **`max_lag` too large** → noise dominates the spectral tail. Keep ≤ ¼ of frames.
6. **Current row 0 is NaN** → must be skipped (the GK kernel does this).
7. **Reporting $\sigma$ as one exact number** → it is window-sensitive with few
   carriers; report a range ([§7.2](#72-an-honest-caveat-few-carriers-uncertain-sigma)).
8. **Wrong $T$ or $V$** (e.g. GROMACS internally using 300 K, $V=26.952$ nm³) →
   systematic offsets in $\varepsilon$ and $\sigma$.

---

## 13. References

- M. Neumann, *Mol. Phys.* **50**, 841 (1983) — dipole-fluctuation formula and boundary conditions (static $\varepsilon$).
- J.-M. Caillol, D. Levesque, J.-J. Weis, *J. Chem. Phys.* **85**, 6645 (1986) — Kubo relation; EH Eq. (30) and GK Eqs. (36)–(39).
- J.-P. Hansen, I. R. McDonald, *Theory of Simple Liquids*, Eq. 7.7.20 — $\sigma(\omega)\leftrightarrow\varepsilon(\omega)$.
- N. Wiener (1930) / A. Khinchin (1934) — autocorrelation ↔ power spectrum.
- W. H. Press et al., *Numerical Recipes* §13.2, §13.9 — FFT autocorrelation and discrete→continuous transform.
- S. Havriliak, S. Negami, *Polymer* **8**, 161 (1967) — non-Debye relaxation model.

## See also

- [Compute overview](index.md) — the Compute → Result pattern and other analyses.
- [API reference: Compute](../api/compute.md) — autodoc for the compute classes.
- [Concepts: Box and Periodicity](../tutorials/03_box_and_periodicity.md) — minimum-image conventions.
- [Concepts: Trajectory](../tutorials/05_trajectory.md) — frame sequences.
