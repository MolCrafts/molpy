# Spectra

This page is a self-contained, textbook-style introduction to predicting
**vibrational spectra** from molecular-dynamics trajectories — the
time-correlation route pioneered for *ab initio* MD analysis. Every spectrum
here is the Fourier transform of an autocorrelation (or cross-correlation) of a
fluctuating quantity, dressed with the appropriate prefactor: a velocity ACF
gives the vibrational density of states, a dipole-flux ACF gives the infrared
spectrum, a polarizability ACF gives Raman, and cross-correlations give the
chiral spectra (VCD, ROA).

The spectral transforms run in the high-performance backend. Unlike the
structural operators, they do **not** take frames — they take a *precomputed ACF*
and the sampling interval, and return the spectrum.

!!! note "Conventions used throughout"
    - The sampling interval `dt_fs` is in **femtoseconds**; output frequencies are
    in wavenumbers (cm⁻¹).
    - You supply the raw ACF (built from the relevant time series — velocities,
    dipole flux, polarizability). Build ACFs with
    `molpy.compute.signal.acf_fft` (scalar series) or the multi-DOF helpers
    documented in [vacf.md](vacf.md) and [msd.md](msd.md).
    - Classical MD intensities usually need a **harmonic quantum correction**
    $Q(\omega)$ before comparing to experiment (see §1.1).

---

## 1. Every vibrational spectrum is the Fourier transform of an ACF

The linear-response/Wiener–Khinchin result is that a spectral density is the
Fourier transform of the time-correlation function of the corresponding dynamical
variable $A(t)$:

$$
I(\omega) \;\propto\; Q(\omega)\int_{-\infty}^{\infty}\!\big\langle A(0)\,A(t)\big\rangle\,e^{-i\omega t}\,\mathrm{d}t,
$$

where $Q(\omega)$ is a method-specific prefactor (a harmonic quantum correction and,
for Raman, a frequency/temperature factor). The choice of $A$ is what distinguishes
the spectra:

| Spectrum | Dynamical variable $A$ | Operator |
|---|---|---|
| VDOS (power) | atomic velocities | `PowerSpectrum` |
| Infrared | total dipole derivative (flux) | `IRSpectrum` |
| Raman | polarizability (iso + aniso) | `RamanSpectrum` |
| VCD | electric ⊗ magnetic dipole | `VcdSpectrum` |
| ROA | ROA invariants (iso + aniso) | `RoaSpectrum` |
| Resonance Raman | resonant polarizability | `ResonanceRamanSpectrum` |

### 1.1 Prefactors, quantum corrections, and resolution

Three numerical facts control every spectrum below.

**Nyquist limit.** With sampling interval $\Delta t$ (fs), the highest
resolvable wavenumber is

$$
\tilde\nu_{\max}\;\approx\;\frac{1}{2c\,\Delta t}
\;\approx\;\frac{16678}{\Delta t/\mathrm{fs}\;\mathrm{cm}^{-1}.
$$

Reaching C–H stretches (~3000 cm⁻¹) needs $\Delta t\lesssim 2.5$ fs; sub-fs
dumps are typical for AIMD.

**Spectral resolution.** An ACF of length $T$ resolves features no narrower
than $\Delta\tilde\nu\sim 1/(cT)$. Truncating the ACF without a window produces
sinc ringing; apply a Hann / Blackman taper (`signal.apply_window`) before the
FFT when the ACF has not fully decayed.

**Quantum correction.** Classical $\langle A(0)A(t)\rangle$ underweights high
frequencies. A common harmonic factor is

$$
Q(\omega)=\frac{\beta\hbar\omega}{1-e^{-\beta\hbar\omega},
\qquad \beta=1/(k_B T),
$$

multiplied into $I(\omega)$ after the transform (and already partially
included by some Raman prefactors when `temperature_k` is set).



<figure id="fig-vdos" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:9"
mark:
  type: line
  strokeWidth: 2.2
  interpolate: monotone
data:
  values:
    - {nu: 0, I: 0.4}
    - {nu: 200, I: 0.5}
    - {nu: 400, I: 0.3}
    - {nu: 600, I: 1.2}
    - {nu: 800, I: 0.4}
    - {nu: 1000, I: 0.2}
    - {nu: 1600, I: 0.9}
    - {nu: 2000, I: 0.15}
    - {nu: 3000, I: 0.6}
    - {nu: 3500, I: 0.1}
encoding:
  x:
    field: nu
    type: quantitative
    title: ν̃ (cm⁻¹)
  y:
    field: I
    type: quantitative
    scale: {zero: false}
    title: intensity (arb.)
  color:
    value: "#0284c7"
```

</div>

**Figure 1.** Schematic VDOS: low-frequency intermolecular band, fingerprint region, and high-frequency intramolecular stretches.
</figure>

---

## 2. Vibrational density of states from velocities

The simplest spectrum is the **power spectrum** of the velocity ACF — the
vibrational density of states (VDOS):

$$
\boxed{\;
g(\omega)
\propto
\int_{-\infty}^{\infty}
\big\langle \mathbf{v}(0)\cdot\mathbf{v}(t)\big\rangle
\,e^{-i\omega t}\,\mathrm{d}t
\;}
$$

It needs no electronic information — only velocities — and locates **every**
vibrational mode, IR-active or not. The zero-frequency weight is proportional to
the diffusion coefficient (solids: $g(0)=0$ after COM removal). Full VACF theory
(cage effect, Green–Kubo $D$): [vacf.md](vacf.md). Sampling choices follow
[vacf.md §6](vacf.md#6-hyperparameter-effects).

```python
import numpy as np

rng = np.random.default_rng(0)
velocities = rng.standard_normal((512, 64, 3)) # (n_frames, n_atoms, 3)
dt, dt_fs = 0.5, 0.5 # sampling interval
from molpy.compute import PowerSpectrum, signal

# One scalar series per degree of freedom, averaged into one curve.
dofs = velocities.reshape(velocities.shape[0], -1)
vacf = np.mean(
 [
 signal.acf_fft(np.ascontiguousarray(dofs[:, k]), 256)
 for k in range(dofs.shape[1])
 ],
 axis=0,
) # raw velocity ACF
vdos = PowerSpectrum()(vacf, dt_fs=0.5) # -> {frequency (cm^-1), intensity}
```

---

## 3. Infrared spectrum from the dipole flux

Linear response relates the IR absorption to the Fourier transform of the
**total-dipole** autocorrelation. In practice classical MD uses the
**dipole flux** $\dot{\mathbf{M}(t)=\mathrm{d}\mathbf{M}/\mathrm{d}t$ (equivalent
up to $\omega^2$ factors and more convenient with noisy $\mathbf{M}$):

$$
\boxed{\;
I_\mathrm{IR}(\omega)
\propto
Q(\omega)\,
\frac{\beta}{3c\,V}
\int_{-\infty}^{\infty}
\big\langle \dot{\mathbf{M}(0)\cdot\dot{\mathbf{M}(t)\big\rangle
\,e^{-i\omega t}\,\mathrm{d}t
\;}
$$

Term by term:

- $\dot{\mathbf{M}$ — time derivative of the cell (or molecular) total dipole.
  For *ab initio* MD, per-molecule dipoles come from
  [Voronoi integration](voronoi.md) of the electron density; for classical force
  fields, $\mathbf{M}=\sum_i q_i\mathbf{r}_i$ with fixed partial charges.
- $Q(\omega)$ — harmonic quantum correction (§1.1).
- $\beta/(3cV)$ — thermal and geometric prefactors (implementation may absorb
  constants into arbitrary intensity units).

**Selection rule in disguise.** Modes that do not modulate $\mathbf{M}$ have
vanishing dipole flux and are IR-silent — exactly as in harmonic normal-mode
analysis, but here anharmonicity and finite temperature are built in.

```python
from molpy.compute import IRSpectrum, RamanSpectrum

# Build ACFs the same way as the VDOS curve — from dipole flux /
# polarizability time series instead of velocities.
dipole_flux_acf = acf_iso = acf_aniso = vacf

ir = IRSpectrum()(dipole_flux_acf, dt_fs=0.5)
```

---

## 4. Raman spectrum from the polarizability

Raman scattering is driven by fluctuations of the polarizability tensor
$\boldsymbol{\alpha}(t)$. Split into isotropic and anisotropic invariants:

$$
\alpha = \tfrac13\operatorname{tr}\boldsymbol{\alpha},
\qquad
\beta^2
= \tfrac12\bigl[
  (\alpha_{xx}-\alpha_{yy})^2 + (\alpha_{yy}-\alpha_{zz})^2
  + (\alpha_{zz}-\alpha_{xx})^2
  + 6(\alpha_{xy}^2+\alpha_{yz}^2+\alpha_{zx}^2)
\bigr].
$$

The corresponding spectral densities are Fourier transforms of
$\langle\alpha(0)\alpha(t)\rangle$ and $\langle\beta(0)\beta(t)\rangle$. The
**scattering intensity** multiplies by a frequency/temperature prefactor that
depends on the incident laser wavenumber $\tilde\nu_0$ and temperature $T$:

$$
I_\mathrm{Raman}(\tilde\nu)
\propto
\frac{(\tilde\nu_0-\tilde\nu)^4}{\tilde\nu\,(1-e^{-\beta h c\tilde\nu})}
\times
\bigl[
  I_\mathrm{iso}(\tilde\nu) + \tfrac{7}{45}I_\mathrm{aniso}(\tilde\nu)
\bigr]
$$

(exact prefactor grouping follows the chosen polarization geometry; MolPy’s
`RamanSpectrum` applies the standard bulk-phase form when
`incident_frequency_cm1` and `temperature_k` are set).

```python
raman = RamanSpectrum(incident_frequency_cm1=20000.0, temperature_k=300.0)(
    acf_iso, acf_aniso, dt_fs=0.5
)
```

Leave `incident_frequency_cm1=0` and `temperature_k=0` for the bare spectral
density without the scattering prefactor.

---

## 5. Chiral spectra: VCD, ROA and resonance Raman

Chiral spectra come from **cross**-correlations of response tensors.

### 5.1 Vibrational circular dichroism (VCD)

VCD is the differential absorption of left- vs right-circularly polarized IR
light. In the time-correlation picture it is the Fourier transform of the
electric-dipole ⊗ magnetic-dipole cross correlation:

$$
I_\mathrm{VCD}(\omega)
\propto
\int_{-\infty}^{\infty}
\big\langle
  \dot{\boldsymbol{\mu}(0)\cdot\dot{\mathbf{m}(t)
\big\rangle
\,e^{-i\omega t}\,\mathrm{d}t,
$$

where $\boldsymbol{\mu}$ and $\mathbf{m}$ are electric and magnetic dipole
moments (or their fluxes). The cross term changes sign under mirror reflection —
hence the optical activity of enantiomers.

### 5.2 Raman optical activity (ROA) and resonance Raman

ROA uses isotropic/anisotropic invariants built from the electric-dipole /
magnetic-dipole / electric-quadrupole polarizability tensors. Resonance Raman
enhances modes coupled to an electronic excitation; MolPy’s
`ResonanceRamanSpectrum` takes the same iso/aniso ACF pair with an incident
frequency near resonance.

```python
from molpy.compute import VcdSpectrum, RoaSpectrum, ResonanceRamanSpectrum

electric_magnetic_acf = vacf

vcd = VcdSpectrum()(electric_magnetic_acf, dt_fs=0.5)
roa = RoaSpectrum(averaged=True)(acf_iso, acf_aniso, dt_fs=0.5)
rr = ResonanceRamanSpectrum(incident_frequency_cm1=20000.0)(
    acf_iso, acf_aniso, dt_fs=0.5
)
```

These APIs implement bulk-phase chiral spectroscopy from MD trajectories (Brehm
& Thomas, 2017).

---

## 6. Pitfalls checklist

1. **Sampling interval too coarse** → Nyquist
   $\tilde\nu_\mathrm{max}\approx 16678/(\Delta t/\mathrm{fs})$ cm⁻¹ must exceed
   the highest mode; sub-fs dumps for C–H stretches (~3000 cm⁻¹).
2. **ACF too short** → resolution $\propto 1/T_\mathrm{ACF}$; window before FFT
   to suppress truncation ringing.
3. **Missing quantum correction** → classical intensities need $Q(\omega)$ for
   quantitative comparison with experiment.
4. **Wrong dynamical variable** → IR needs the dipole *flux*, not raw $\mathbf{M}$
   without the derivative convention; VDOS needs velocities.
5. **Unconverged molecular dipoles/polarizabilities** → verify
   [Voronoi](voronoi.md) charges before trusting IR intensities.
6. **COM / total-dipole drift** → produces a huge $\omega=0$ spike; remove bulk
   translation and check neutrality.
7. **Thermostat noise** → sample spectra from NVE (or weakly thermostatted)
   production after equilibration.

---

## 7. References

- D. A. McQuarrie, *Statistical Mechanics*, Harper & Row (1976) — time-correlation
  functions and spectral densities.
- R. G. Gordon, *Adv. Magn. Reson.* **3**, 1 (1968) — correlation-function view
  of IR/Raman band shapes.
- M. Thomas, M. Brehm, R. Fligg, P. Vöhringer, B. Kirchner, *Phys. Chem. Chem.
  Phys.* **15**, 6608 (2013) — IR and Raman from AIMD via TCFs.
- M. Brehm, M. Thomas, *J. Phys. Chem. Lett.* **8**, 3409 (2017) — VCD, ROA and
  resonance Raman from MD.
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105
  (2020) — AIMD analysis feature set.

## See also

- [Voronoi](voronoi.md) — molecular dipoles for IR intensities.
- [VACF](vacf.md) — VACF theory and sampling.
- [Van Hove & Reorientational Dynamics](van_hove.md) — lineshape-related dynamics.
- [Dielectric](dielectric.md) — dipole-fluctuation route to
  $\varepsilon^*(\omega)$.
- [API reference: Compute](../api/compute.md).
