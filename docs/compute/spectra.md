# Vibrational Spectra from MD: IR, Raman, VDOS, VCD & ROA

This page is a self-contained, textbook-style introduction to predicting
**vibrational spectra** from molecular-dynamics trajectories — the time-correlation
route that the reference implementation pioneered for *ab initio* MD. Every spectrum here is the Fourier
transform of an autocorrelation function (ACF) of a fluctuating quantity, dressed
with the appropriate prefactor: a velocity ACF gives the vibrational density of
states, a dipole-derivative ACF gives the infrared spectrum, a polarizability ACF
gives Raman, and cross-correlations give the chiral spectra (VCD, ROA).

The spectral transforms run in Rust (`molrs`). Unlike the structural operators,
they do **not** take frames — they take a *precomputed ACF* and the sampling
interval, and return the spectrum.

!!! note "Conventions used throughout"
    - The sampling interval `dt_fs` is in **femtoseconds**; output frequencies are
      in wavenumbers (cm⁻¹).
    - You supply the raw ACF (built from the relevant time series — velocities,
      dipole flux, polarizability). MolPy's
      [`compute_acf`/`ACFAnalyzer`](transport.md) build ACFs from a column.

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

---

## 2. Vibrational density of states from velocities

The simplest spectrum is the **power spectrum** of the velocity ACF, the
vibrational density of states (VDOS). It needs no electronic information — just the
velocities — and locates every vibrational mode, IR-active or not. See
[Velocity Autocorrelation & VDOS](vacf.md) for the full theory (cage effect,
Green–Kubo diffusion); the `dt_fs` and `cache_size` choices below follow
[vacf.md §6 — Hyperparameter effects](vacf.md#6-hyperparameter-effects).

```python
from molpy.compute import PowerSpectrum, compute_acf

# velocities: (n_frames, n_atoms, 3); cache_size = max lag in frames
vacf = compute_acf(velocities, cache_size=4096)  # raw velocity autocorrelation
vdos = PowerSpectrum()(vacf, dt_fs=0.5)          # -> {frequency (cm^-1), intensity}
```

---

## 3. Infrared and Raman spectra

The **infrared** spectrum is the Fourier transform of the autocorrelation of the
cell's total dipole derivative ($\dot{\mathbf M}$, the dipole "flux"); its
intensities require the molecular dipoles, which for *ab initio* MD come from
[Voronoi integration](voronoi.md) of the electron density. The **Raman** spectrum
uses the molecular polarizability, split into isotropic and anisotropic parts:

```python
from molpy.compute import IRSpectrum, RamanSpectrum

ir = IRSpectrum()(dipole_flux_acf, dt_fs=0.5)

raman = RamanSpectrum(incident_frequency_cm1=20000.0, temperature_k=300.0)(
    acf_iso, acf_aniso, dt_fs=0.5
)
```

The `incident_frequency_cm1` and `temperature_k` apply the Raman scattering
prefactor; leave them at `0.0` to obtain the bare spectral density.

---

## 4. Chiral spectra: VCD, ROA and resonance Raman

The chiral spectra come from *cross*-correlations. **Vibrational circular
dichroism** (VCD) is the transform of the electric-dipole ⊗ magnetic-dipole
cross-ACF; **Raman optical activity** (ROA) and **resonance Raman** use isotropic
and anisotropic invariants of the relevant response tensors:

```python
from molpy.compute import VcdSpectrum, RoaSpectrum, ResonanceRamanSpectrum

vcd = VcdSpectrum()(electric_magnetic_acf, dt_fs=0.5)
roa = RoaSpectrum(averaged=True)(acf_iso, acf_aniso, dt_fs=0.5)
rr  = ResonanceRamanSpectrum(incident_frequency_cm1=20000.0)(acf_iso, acf_aniso, dt_fs=0.5)
```

These reproduce the reference implementation's bulk-phase chiral-spectroscopy predictions, the first of
their kind from MD.

---

## 5. Pitfalls checklist

1. **Sampling interval too coarse** → the Nyquist limit $\tilde\nu_\text{max} =
   1/(2c\,\Delta t)$ must exceed the highest mode; sub-femtosecond `dt_fs` is
   needed to reach C–H stretches (~3000 cm⁻¹).
2. **ACF too short** → spectral resolution is set by the ACF length; a short ACF
   gives broad, unresolved peaks. Window the ACF to suppress truncation ringing.
3. **Missing quantum correction** → classical intensities need a harmonic quantum
   correction factor for quantitative comparison with experiment.
4. **Wrong variable for the spectrum** → IR needs the dipole *flux* (derivative),
   not the dipole; VDOS needs velocities, not positions.
5. **Unconverged molecular dipoles/polarizabilities** → garbage in, garbage out:
   verify the [Voronoi](voronoi.md) charges before trusting IR intensities.

---

## 6. References

- D. A. McQuarrie, *Statistical Mechanics*, Harper & Row (1976) — time-correlation
  functions and spectral densities.
- M. Thomas, M. Brehm, R. Fligg, P. Vöhringer, B. Kirchner, *Phys. Chem. Chem.
  Phys.* **15**, 6608 (2013) — IR and Raman spectra from AIMD via TCFs.
- M. Brehm, M. Thomas, *J. Phys. Chem. Lett.* **8**, 3409 (2017) — VCD, ROA and
  resonance Raman from MD.
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105
  (2020) — reference implementation.

## See also

- [Radical Voronoi](voronoi.md) — supplies the molecular dipoles IR intensities need.
- [Van Hove & Reorientational Dynamics](van-hove.md) — the dynamical correlations behind lineshapes.
- [Dielectric Spectroscopy](dielectric.md) — the dipole-fluctuation route to the dielectric response.
- [API reference: Compute](../api/compute.md).
