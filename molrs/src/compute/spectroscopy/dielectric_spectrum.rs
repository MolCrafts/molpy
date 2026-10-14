//! Dielectric ε(ω) transform [`Fit`] impls: [`EinsteinHelfandSpectrum`] and
//! [`GreenKuboSpectrum`].
//!
//! The window + one-sided-FFT machinery (`piecewise_linear_onesided_ft`,
//! `taper_derivative_spectrum`, `windowed_acf_spectrum`) was relocated here
//! from `compute::dielectric` in compute-fit-04-dielectric: windowing +
//! transforming a raw ACF into ε(ω) is a *fit*, so it belongs in the [`Fit`]
//! layer. Window coefficients always route through [`molrs::signal`] (never
//! reimplemented).

use ndarray::Array1;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

use super::windowed_acf;
use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::{Check, Fit, Verdict};
use crate::signal::forward_fft_onesided;
use molrs::signal as sig;

/// Zero-padding multiplier for the dielectric one-sided FT.
///
/// Matches the DRS / trungle `analyze_*.py` convention (`PAD_FACTOR = 4`):
/// `n_fft = next_pow2(max(n, pad_factor · n))`. Larger pad improves frequency
/// resolution without changing the continuous piecewise-linear integral.
const DIELECTRIC_PAD_FACTOR: usize = 4;

// ── Physical constants (MD real units: kcal, mol, Angstrom, e, K) ─────────────

use molrs::core::constants::COULOMB_REAL as KAPPA;
use molrs::core::{FOUR_PI, FOUR_THIRDS_PI, UnitPreset};

/// Result of a dielectric ε(ω) spectrum transform.
///
/// The complex permittivity is `ε*(ω) = ε′(ω) − i·ε″(ω)`. `eps_imag`
/// stores `ε″(ω)` with the **positive-loss** convention (≥ 0 for stable
/// causal systems). FT convention throughout: `X(ω) = ∫₀^∞ f(t) e^{−iωt} dt`.
///
/// Each of [`EinsteinHelfandSpectrum`] and [`GreenKuboSpectrum`] — a
/// [`Fit`] — consumes a **raw autocorrelation function** (the fluctuation
/// dipole ACF for the Einstein–Helfand route; the current ACF for the
/// Green–Kubo route) plus the physical metadata (`dt`, `V`, `T`, `ε_∞`, and —
/// for EH — the zero-lag variance ⟨|δM|²⟩) and applies window + FFT +
/// prefactors to produce this frequency-dependent permittivity.
///
/// The raw, unwindowed ACFs these fits consume come from the raw computes
/// [`DebyeRelaxation`](crate::compute::DebyeRelaxation) (fluctuation dipole ACF +
/// ⟨M(0)²⟩ + V/T/Ewald-BC) and
/// [`GreenKuboConductivity`](crate::compute::GreenKuboConductivity) (current ACF).
///
/// # Units
///
/// All inputs and outputs use LAMMPS *real* units throughout:
///
/// | quantity        | unit                |
/// |-----------------|---------------------|
/// | length          | Å                   |
/// | charge          | e                   |
/// | time / dt       | ps                  |
/// | temperature     | K                   |
/// | volume          | Å³                  |
/// | dipole moment   | e · Å               |
/// | current density | e · Å⁻² · ps⁻¹      |
/// | angular ω       | rad · ps⁻¹          |
/// | ε permittivity  | dimensionless       |
#[derive(Debug, Clone)]
pub struct DielectricSpectrumResult {
    /// Angular frequency grid, rad·ps⁻¹, length `n_pad/2 + 1` with
    /// `n_pad = (2·n_correlation_steps).next_power_of_two()`.
    /// Bin 0 is the DC bin; bin 1 is Δω = 2π/(n_pad·dt); the last bin
    /// is the Nyquist frequency π/dt.
    pub frequencies: Array1<f64>,
    /// Real part ε′(ω), dimensionless.
    pub eps_real: Array1<f64>,
    /// Loss spectrum ε″(ω), dimensionless (positive sign convention).
    pub eps_imag: Array1<f64>,
}

impl ComputeResult for DielectricSpectrumResult {}

// ── Shared transform helpers (relocated from compute::dielectric) ────────────

/// `(frequencies, spec_re, spec_im)` triple — the real and imaginary parts of
/// the continuous one-sided Fourier transform `X(ω) = ∫₀^∞ C(t) e^{−iωt} dt`
/// of a (possibly windowed / differentiated) ACF on the rfft frequency grid.
/// No physical prefactor (β, V, k_B T, ε₀, …) is applied.
type RawSpectrum = (Array1<f64>, Array1<f64>, Array1<f64>);

/// Continuous one-sided FT of the **piecewise-linear interpolant** of a
/// discrete ACF (or differentiated ACF).
///
/// For samples `y[k]` at times `k·dt`, this evaluates
///
/// ```text
///     X(ω) = ∫₀^T y_lin(t) e^{−iωt} dt
/// ```
///
/// analytically (closed form via the first-difference rFFT), matching the
/// DRS / trungle `one_sided_linear_transform` used for JACF → χ(ω).
///
/// Rectangle-rule `rfft(y)·dt` systematically inflates the low-frequency
/// loss spectrum ε″ when the ACF is truncated with `window="none"` (factors
/// of 3–15 vs literature at 0.01–1 THz on neat water). The piecewise-linear
/// integral removes that bias; unit prefactors (β, V, 4π·κ) are unchanged.
///
/// `pad_factor` controls zero-padding of the difference array before the
/// rFFT (`n_fft = next_pow2(max(n, pad_factor·n))`).
fn piecewise_linear_onesided_ft(y: &Array1<f64>, dt: f64, pad_factor: usize) -> RawSpectrum {
    let n = y.len();
    debug_assert!(n >= 2 && dt > 0.0);
    let n_fft = (n.max(pad_factor.saturating_mul(n).max(1))).next_power_of_two();

    // First differences of the discrete samples (length n-1).
    let mut dy = Vec::with_capacity(n.saturating_sub(1));
    for k in 0..n.saturating_sub(1) {
        dy.push(y[k + 1] - y[k]);
    }

    let mut planner = FftPlanner::new();
    let dy_bins = forward_fft_onesided(&mut planner, &dy, n_fft);

    let frequencies = sig::frequency_grid(n_fft, dt);
    let n_freq = frequencies.len();
    let mut spec_re = Array1::zeros(n_freq);
    let mut spec_im = Array1::zeros(n_freq);

    // DC: trapezoidal integral of the piecewise-linear interpolant.
    let mut trap = 0.0;
    for k in 0..n.saturating_sub(1) {
        trap += 0.5 * (y[k] + y[k + 1]) * dt;
    }
    spec_re[0] = trap;
    spec_im[0] = 0.0;

    let y0 = y[0];
    let y_last = y[n - 1];
    let t_end = (n - 1) as f64 * dt;

    for j in 1..n_freq {
        let omega = frequencies[j];
        let theta = omega * dt;
        // e^{−iωT}
        let exp_iwt = Complex64::new((-omega * t_end).cos(), (-omega * t_end).sin());
        // (y₀ − y_last e^{−iωT}) / (iω)
        let term1 = (Complex64::new(y0, 0.0) - Complex64::new(y_last, 0.0) * exp_iwt)
            / Complex64::new(0.0, omega);
        // (1 − e^{−iθ}) / (ω² dt) · rfft(dy)[j]
        let exp_ith = Complex64::new((-theta).cos(), (-theta).sin());
        let pre = (Complex64::new(1.0, 0.0) - exp_ith) / (omega * omega * dt);
        let out = term1 - pre * dy_bins[j];
        spec_re[j] = out.re;
        spec_im[j] = out.im;
    }

    (frequencies, spec_re, spec_im)
}

/// One-sided cosine² taper → central-difference derivative → one-sided FT of a
/// raw fluctuation dipole ACF.
///
/// Returns `(frequencies, dre, dim)` where `dre + i·dim ≈ Ĉ′(ω) =
/// ∫₀^∞ C′(t) e^{−iωt} dt`. Transforming the ACF *derivative* — rather than
/// forming `ω·X(ω)` from the transform `X(ω)` of the ACF itself — is what keeps
/// the loss spectrum finite. `C′(0) = 0` (the ACF is even) and the cos² taper
/// (1 at `C(0)`, 0 at `C(L)`) drives `C(L) → 0`, so `Ĉ′(ω)` decays toward the
/// Nyquist frequency instead of diverging.
///
/// The input `acf` must be the **fluctuation** (mean-subtracted) dipole ACF
/// `C(k) = ⟨δM(0)·δM(k·dt)⟩` summed over the 3 Cartesian components — exactly
/// the [`DebyeRelaxationResult.acf`](crate::compute::DebyeRelaxationResult::acf).
fn taper_derivative_spectrum(acf: &Array1<f64>, dt: f64) -> RawSpectrum {
    let max_lag = acf.len() - 1;

    // One-sided cosine² taper: 1 at C(0), 0 at C(max_lag).
    let tapered = windowed_acf(acf, sig::WindowType::CosineSq);

    // Central finite-difference derivative of the tapered ACF. C′(0) = 0
    // exactly because the autocorrelation is even; the tail uses a one-sided
    // stencil (the taper has already driven C(max_lag) to zero).
    let mut deriv = Array1::<f64>::zeros(max_lag + 1);
    for k in 1..max_lag {
        deriv[k] = (tapered[k + 1] - tapered[k - 1]) / (2.0 * dt);
    }
    if max_lag >= 1 {
        deriv[max_lag] = (tapered[max_lag] - tapered[max_lag - 1]) / dt;
    }

    piecewise_linear_onesided_ft(&deriv, dt, DIELECTRIC_PAD_FACTOR)
}

/// [`WindowType::from_name`](sig::WindowType::from_name) with the error
/// mapped onto the `window_type` field.
fn parse_window_type(s: &str) -> Result<sig::WindowType, ComputeError> {
    sig::WindowType::from_name(s).map_err(|_| ComputeError::OutOfRange {
        field: "window_type",
        value: s.into(),
    })
}

/// Window a raw current ACF and take its continuous one-sided FT
/// (piecewise-linear interpolant).
///
/// Returns `(frequencies, spec_re, spec_im)` where `spec_re + i·spec_im` is
/// `X(ω) = ∫₀^T C_win(t) e^{−iωt} dt`. The input `acf` must be the unbiased
/// current ACF `C(k) = ⟨J(0)·J(k·dt)⟩` summed over the 3 Cartesian components
/// — exactly the
/// [`GreenKuboConductivityResult.jacf`](crate::compute::GreenKuboConductivityResult::jacf).
fn windowed_acf_spectrum(
    acf: &Array1<f64>,
    dt: f64,
    window_type: &str,
) -> Result<RawSpectrum, ComputeError> {
    let wt = parse_window_type(window_type)?;
    let windowed_1d = windowed_acf(acf, wt);

    Ok(piecewise_linear_onesided_ft(
        &windowed_1d,
        dt,
        DIELECTRIC_PAD_FACTOR,
    ))
}

// ── Einstein–Helfand ε(ω) transform ──────────────────────────────────────────

/// Einstein–Helfand (dipole-ACF) route to the frequency-dependent permittivity.
///
/// Consumes the **raw fluctuation dipole ACF** `C(k) = ⟨δM(0)·δM(k·dt)⟩`
/// (summed over the 3 Cartesian components) — the
/// [`DebyeRelaxationResult.acf`](crate::compute::DebyeRelaxationResult::acf) — together
/// with `dt`, `V`, `T`, `ε_∞`, and the zero-lag variance ⟨|δM|²⟩ (the
/// [`zero_lag_variance`](crate::compute::DebyeRelaxationResult::zero_lag_variance), which
/// pins the exact DC bin).
///
/// Implements the integration-by-parts form of Caillol-Levesque-Weis Eq. (30):
///
/// ```text
///     ε*(ω) − ε_∞ = −A · Ĉ′(ω),     A = 4π·KAPPA / (3·V·k_B·T)
///     Ĉ′(ω) = ∫₀^∞ C′(t) e^{−iωt} dt
/// ```
///
/// Under the positive-loss convention `ε* = ε′ − i·ε″`, with
/// `Ĉ′ = Ĉ′_re + i·Ĉ′_im`:
///
/// ```text
///     ε′(ω) − ε_∞ = −A · Ĉ′_re(ω)
///     ε″(ω)       =  A · Ĉ′_im(ω)
/// ```
///
/// At `ω = 0` the result contracts to the Neumann static formula
/// `ε(0) − ε_∞ = A·⟨|δM|²⟩`; the DC bin is set to that exact value, matching
/// [`static_dielectric_constant`](crate::compute::static_dielectric_constant).
///
/// # References
/// - Caillol, Levesque & Weis, *J. Chem. Phys.* **85**, 6645 (1986), Eq. (30).
/// - Neumann, *Mol. Phys.* **50**, 841 (1983), static-limit identity.
#[derive(Debug, Clone, Copy)]
pub struct EinsteinHelfandSpectrum {
    /// Frame spacing, **ps**, > 0. Sets the frequency resolution
    /// `Δω = 2π / (n_pad · dt)`.
    pub dt: f64,
    /// System volume V, **Å³**, > 0.
    pub volume: f64,
    /// Temperature T, **K**, > 0.
    pub temperature: f64,
    /// High-frequency / electronic permittivity ε_∞, dimensionless.
    pub epsilon_inf: f64,
    /// Zero-lag variance ⟨|δM|²⟩ = `acf[0]`, **(e·Å)²** — the exact DC term
    /// (the [`DebyeRelaxationResult.zero_lag_variance`](crate::compute::DebyeRelaxationResult::zero_lag_variance)).
    pub zero_lag_variance: f64,
}

impl Fit for EinsteinHelfandSpectrum {
    /// The raw fluctuation dipole ACF (1D, length `max_lag + 1`).
    type Input<'a> = &'a Array1<f64>;
    type Output = DielectricSpectrumResult;

    /// Taper + derivative + FFT the raw dipole ACF into ε(ω).
    ///
    /// # Errors
    /// * [`ComputeError::EmptyInput`] if the ACF has fewer than 2 lags.
    /// * [`ComputeError::OutOfRange`] if `dt`/`volume`/`temperature` ≤ 0.
    fn fit<'a>(&self, acf: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        if acf.len() < 2 {
            return Err(ComputeError::EmptyInput);
        }
        validate_thermo(self.dt, self.volume, self.temperature)?;

        let (frequencies, dre, dim) = taper_derivative_spectrum(acf, self.dt);

        // ε*(ω) − ε_∞ = −A·Ĉ′(ω), with ε* = ε′ − i·ε″ (positive-loss convention).
        let prefactor = FOUR_THIRDS_PI * KAPPA
            / (self.volume * UnitPreset::real().boltzmann() * self.temperature);
        let n_freq = frequencies.len();
        let mut eps_real = Array1::zeros(n_freq);
        let mut eps_imag = Array1::zeros(n_freq);
        for j in 0..n_freq {
            eps_real[j] = self.epsilon_inf - prefactor * dre[j];
            eps_imag[j] = prefactor * dim[j];
        }
        // DC bin: contract to the exact Neumann static limit. The discrete
        // derivative transform reproduces it only up to O(dt) truncation error.
        eps_real[0] = self.epsilon_inf + prefactor * self.zero_lag_variance;
        eps_imag[0] = 0.0;

        Ok(DielectricSpectrumResult {
            frequencies,
            eps_real,
            eps_imag,
        })
    }
}

// ── Green–Kubo ε(ω) transform ─────────────────────────────────────────────────

/// Green–Kubo (current-ACF) route to the frequency-dependent permittivity.
///
/// Consumes the **raw current ACF** `C(k) = ⟨J(0)·J(k·dt)⟩` (summed over the 3
/// Cartesian components) — the
/// [`GreenKuboConductivityResult.jacf`](crate::compute::GreenKuboConductivityResult::jacf)
/// — together with `dt`, `V`, `T`, `ε_∞`, and the window choice.
///
/// Implements
///
/// ```text
///     σ(ω) = (V / (3·k_B·T)) · ∫₀^∞ ⟨J(0)·J(t)⟩ e^{−iωt} dt = σ′ + i·σ″
///     ε*(ω) − ε_∞ = −i · σ(ω) / (ω · ε₀)
/// ```
///
/// (Hansen & McDonald, *Theory of Simple Liquids*, Eq. 7.7.20; equivalent to
/// Eq. (39) of Caillol-Levesque-Weis after substituting `J = Ṁ/V`).
/// Separating real/imaginary parts under `ε* = ε′ − i·ε″`:
///
/// ```text
///     ε′(ω) − ε_∞ = σ″(ω) / (ω · ε₀)
///     ε″(ω)       = σ′(ω) / (ω · ε₀)
/// ```
///
/// with `1/ε₀ = 4π·KAPPA` in LAMMPS real units. The DC bin (`ω = 0`) is
/// regularized to `(ε_∞, 0)`; the physical static limit must be taken from
/// [`static_dielectric_constant`](crate::compute::static_dielectric_constant)
/// for insulators or extrapolated from low-ω bins for conductors.
///
/// # References
/// - Caillol, Levesque & Weis, *J. Chem. Phys.* **85**, 6645 (1986), Eqs.
///   (36)–(39).
/// - Hansen & McDonald, *Theory of Simple Liquids*, Eq. 7.7.20.
#[derive(Debug, Clone)]
pub struct GreenKuboSpectrum {
    /// Frame spacing, **ps**, > 0.
    pub dt: f64,
    /// System volume V, **Å³**, > 0.
    pub volume: f64,
    /// Temperature T, **K**, > 0.
    pub temperature: f64,
    /// High-frequency / electronic permittivity ε_∞, dimensionless.
    pub epsilon_inf: f64,
    /// Window applied to the one-sided current ACF: `"cosine_sq"`, `"hann"`, or
    /// `"blackman"`.
    pub window_type: String,
}

impl Fit for GreenKuboSpectrum {
    /// The raw current ACF (1D, length `max_lag + 1`).
    type Input<'a> = &'a Array1<f64>;
    type Output = DielectricSpectrumResult;

    /// Window + FFT the raw current ACF and convert σ(ω) → ε(ω).
    ///
    /// # Errors
    /// * [`ComputeError::EmptyInput`] if the ACF has fewer than 2 lags.
    /// * [`ComputeError::OutOfRange`] if `dt`/`volume`/`temperature` ≤ 0 or the
    ///   window type is unknown.
    fn fit<'a>(&self, acf: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        if acf.len() < 2 {
            return Err(ComputeError::EmptyInput);
        }
        validate_thermo(self.dt, self.volume, self.temperature)?;

        let (frequencies, spec_re, spec_im) =
            windowed_acf_spectrum(acf, self.dt, &self.window_type)?;

        // σ(ω) = sigma_prefactor · (spec_re + i·spec_im) under e^{−iωt}.
        // ε*(ω) − ε_∞ = −i·σ(ω)/(ω·ε₀):
        //   ε′(ω) − ε_∞ = σ″/(ω·ε₀) = (1/ε₀)·sigma_prefactor·spec_im/ω
        //   ε″(ω)       = σ′/(ω·ε₀) = (1/ε₀)·sigma_prefactor·spec_re/ω
        // With J(t) = Ṁ(t)/V: ⟨Ṁ·Ṁ⟩ = V²·⟨J·J⟩, so the textbook
        // 1/(3·V·k_B·T) prefactor for Ṁ becomes V/(3·k_B·T) for J.
        // 1/ε₀ = 4π·KAPPA in MD real units.
        let sigma_prefactor =
            self.volume / (3.0 * UnitPreset::real().boltzmann() * self.temperature);
        let eps0_factor = FOUR_PI * KAPPA;
        let n_freq = frequencies.len();
        let mut eps_real = Array1::zeros(n_freq);
        let mut eps_imag = Array1::zeros(n_freq);
        for j in 0..n_freq {
            let omega = frequencies[j];
            if omega == 0.0 {
                // DC bin: σ/ω is 0/0. The true static limit lives in
                // `static_dielectric_constant`; this routine regularizes the
                // bin to (ε_∞, 0) and the caller must consult the static helper.
                eps_real[j] = self.epsilon_inf;
                eps_imag[j] = 0.0;
            } else {
                let sigma_re = sigma_prefactor * spec_re[j];
                let sigma_im = sigma_prefactor * spec_im[j];
                eps_real[j] = self.epsilon_inf + eps0_factor * sigma_im / omega;
                eps_imag[j] = eps0_factor * sigma_re / omega;
            }
        }

        Ok(DielectricSpectrumResult {
            frequencies,
            eps_real,
            eps_imag,
        })
    }
}

// ── Dipole–rate / polarization cross-correlation route ────────────────────────

/// Permittivity from the **dipole-rate × dipole** cross-correlation
/// `C_{ṀM}(t) = Σ_α ⟨δṀ_α(0) δM_α(t)⟩` (e·Å/ps · e·Å).
///
/// Linear-response identity (conducting BC, real units):
///
/// ```text
///     χ(ω) = A · Ĉ_{ṀM}(ω),     A = 4π·KAPPA / (3·V·k_B·T)
///     ε*(ω) − ε_∞ = χ(ω)
/// ```
///
/// with one-sided continuous FT of the (optionally windowed) correlator
/// (piecewise-linear interpolant, same kernel as [`GreenKuboSpectrum`]).
/// Positive-loss convention: `ε* = ε′ − i·ε″` ⇒
/// `ε′ = ε_∞ + Re χ`, `ε″ = −Im χ`.
///
/// This is the MD realization of the ⟨Ṗ(0)·P(t)⟩ susceptibility route
/// (notes / DRS_MD_v2; Caillol–Levesque–Weis cross form). Numerically
/// distinct from [`GreenKuboSpectrum`] (which uses ⟨J·J⟩) even though
/// `J = Ṁ/V` at fixed V: the correlator and prefactor arrangement differ.
///
/// # Input
/// Unbiased 1-D lag series of length `max_lag+1`, Cartesian sum already
/// performed. `Ṁ` is typically a finite-difference of the unwrapped dipole
/// (not the velocity current used for JACF).
#[derive(Debug, Clone)]
pub struct DipoleRateCrossSpectrum {
    /// Frame spacing, **ps**, > 0.
    pub dt: f64,
    /// System volume V, **Å³**, > 0.
    pub volume: f64,
    /// Temperature T, **K**, > 0.
    pub temperature: f64,
    /// High-frequency permittivity ε_∞.
    pub epsilon_inf: f64,
    /// Window on the one-sided cross correlator: `"none"`, `"cosine_sq"`, …
    pub window_type: String,
}

impl Fit for DipoleRateCrossSpectrum {
    type Input<'a> = &'a Array1<f64>;
    type Output = DielectricSpectrumResult;

    fn fit<'a>(&self, cross: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        if cross.len() < 2 {
            return Err(ComputeError::EmptyInput);
        }
        validate_thermo(self.dt, self.volume, self.temperature)?;

        let (frequencies, re, im) = windowed_acf_spectrum(cross, self.dt, &self.window_type)?;
        let prefactor = FOUR_THIRDS_PI * KAPPA
            / (self.volume * UnitPreset::real().boltzmann() * self.temperature);
        let n_freq = frequencies.len();
        let mut eps_real = Array1::zeros(n_freq);
        let mut eps_imag = Array1::zeros(n_freq);
        for j in 0..n_freq {
            // χ = A · (re + i·im); ε′=ε∞+Reχ, ε″=−Imχ
            eps_real[j] = self.epsilon_inf + prefactor * re[j];
            eps_imag[j] = -prefactor * im[j];
        }
        Ok(DielectricSpectrumResult {
            frequencies,
            eps_real,
            eps_imag,
        })
    }
}

// ── Dipole autocorrelation (PACF) route ───────────────────────────────────────

/// Permittivity from the **fluctuation dipole autocorrelation**
/// `C(t) = Σ_α ⟨δM_α(0) δM_α(t)⟩` via the Fourier identity
///
/// ```text
///     χ(ω) = A · [ C(0) − iω Ĉ(ω) ],     A = 4π·KAPPA / (3·V·k_B·T)
///     ε*(ω) − ε_∞ = χ(ω)
/// ```
///
/// (integration-by-parts form of the PACF susceptibility; DRS notes / Trung
/// `polarization_acf` path). Distinct from [`EinsteinHelfandSpectrum`], which
/// uses a cos² taper of `C′(t)` instead of the `C(0)−iωĈ` algebra.
///
/// # Conductive plateau
///
/// For ionic conductors the PACF often plateaus (`C(∞)/C(0) → O(1)`): the
/// total cell dipole does not relax. The bare formula then yields a huge
/// false static χ. When [`Self::subtract_plateau`] is `Some(true)`, or
/// `None` and `|C(∞)/C(0)| > 0.1`, the fit applies the **same** formula to
/// `C′(t) = C(t) − C(∞)` with `C′(0) = C(0) − C(∞)` (relaxational PACF).
/// Water (`C∞/C0 ∼ 0`) is unchanged under auto mode.
#[derive(Debug, Clone)]
pub struct DipoleAutocorrelationSpectrum {
    /// Frame spacing, **ps**, > 0.
    pub dt: f64,
    /// System volume V, **Å³**, > 0.
    pub volume: f64,
    /// Temperature T, **K**, > 0.
    pub temperature: f64,
    /// High-frequency permittivity ε_∞.
    pub epsilon_inf: f64,
    /// Window on the PACF before the FT (`"none"` for DRS slides).
    pub window_type: String,
    /// Plateau handling: `None` = auto if `|C∞/C0| > 0.1`; `Some(true/false)` force.
    pub subtract_plateau: Option<bool>,
}

impl Fit for DipoleAutocorrelationSpectrum {
    type Input<'a> = &'a Array1<f64>;
    type Output = DielectricSpectrumResult;

    fn fit<'a>(&self, acf: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        if acf.len() < 2 {
            return Err(ComputeError::EmptyInput);
        }
        validate_thermo(self.dt, self.volume, self.temperature)?;

        let c0_raw = acf[0];
        let c_inf = acf[acf.len() - 1];
        let ratio = if c0_raw.abs() > 0.0 {
            c_inf / c0_raw
        } else {
            0.0
        };
        let do_sub = match self.subtract_plateau {
            Some(v) => v,
            None => ratio.abs() > 0.1,
        };

        let (c0, series): (f64, Array1<f64>) = if do_sub {
            let mut s = acf.to_owned();
            for v in s.iter_mut() {
                *v -= c_inf;
            }
            (c0_raw - c_inf, s)
        } else {
            (c0_raw, acf.to_owned())
        };

        let (frequencies, re, im) = windowed_acf_spectrum(&series, self.dt, &self.window_type)?;
        let prefactor = FOUR_THIRDS_PI * KAPPA
            / (self.volume * UnitPreset::real().boltzmann() * self.temperature);
        let n_freq = frequencies.len();
        let mut eps_real = Array1::zeros(n_freq);
        let mut eps_imag = Array1::zeros(n_freq);
        for j in 0..n_freq {
            let omega = frequencies[j];
            // χ = A · (c0 − iω Ĉ), Ĉ = re + i·im
            // Re χ = A · (c0 + ω·im),  Im χ = A · (−ω·re)
            // ε′ = ε∞ + Reχ,  ε″ = −Imχ = A·ω·re
            if omega == 0.0 {
                eps_real[j] = self.epsilon_inf + prefactor * c0;
                eps_imag[j] = 0.0;
            } else {
                let chi_re = prefactor * (c0 + omega * im[j]);
                let chi_im = prefactor * (-omega * re[j]);
                eps_real[j] = self.epsilon_inf + chi_re;
                eps_imag[j] = -chi_im;
            }
        }
        Ok(DielectricSpectrumResult {
            frequencies,
            eps_real,
            eps_imag,
        })
    }
}

/// Validate the shared `dt`/`volume`/`temperature` > 0 contract.
fn validate_thermo(dt: f64, volume: f64, temperature: f64) -> Result<(), ComputeError> {
    if dt <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "dt",
            value: dt.to_string(),
        });
    }
    if volume <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "volume",
            value: volume.to_string(),
        });
    }
    if temperature <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "temperature",
            value: temperature.to_string(),
        });
    }
    Ok(())
}

// ─── Consistency checks ──────────────────────────────────────────────────
// The Check stage for the spectra above. `EinsteinHelfandSpectrum` and
// `GreenKuboSpectrum` produce ε(ω) / σ(ω); these judge that output against
// physics. compute, fit and check for one physical quantity live in one file.

/// Outcome of the Kramers-Kronig consistency check.
#[derive(Debug, Clone)]
pub struct KramersKronigCheck {
    /// ε'(ω) recovered from ε''(ω).
    pub recovered: Array1<f64>,
    /// Mean absolute error vs the supplied real part.
    pub mae: f64,
    /// Whether `mae` is within the (dynamic-range-scaled) tolerance.
    pub passed: bool,
}

/// Recover ε'(ω) from ε''(ω) by a discrete Kramers-Kronig integral with
/// trapezoidal weights, offset by ε∞ (the principal value skips j == i).
fn discrete_kramers_kronig(
    omega: &Array1<f64>,
    eps_imag: &Array1<f64>,
    eps_inf: f64,
) -> Array1<f64> {
    let n = omega.len();
    let mut recovered = Array1::<f64>::from_elem(n, eps_inf);
    for i in 0..n {
        let omega_i = omega[i];
        let mut acc = 0.0;
        for j in 0..n {
            if j == i {
                continue;
            }
            let omega_j = omega[j];
            let denom = omega_j * omega_j - omega_i * omega_i;
            if denom.abs() < 1e-30 {
                continue;
            }
            let dw = if j == 0 {
                omega[1] - omega[0]
            } else if j == n - 1 {
                omega[n - 1] - omega[n - 2]
            } else {
                0.5 * (omega[j + 1] - omega[j - 1])
            };
            acc += eps_imag[j] * omega_j / denom * dw;
        }
        recovered[i] += (2.0 / std::f64::consts::PI) * acc;
    }
    recovered
}

/// Kramers-Kronig causality of a dielectric spectrum.
///
/// `eps_inf` is the high-frequency limit ε∞ that offsets the recovered real
/// part. Requires ≥ 3 frequency points and equal-length arrays.
#[derive(Debug, Clone, Copy)]
pub struct KramersKronig {
    /// High-frequency limit ε∞.
    pub eps_inf: f64,
}

impl Check for KramersKronig {
    /// `(omega, eps_real, eps_imag)`.
    type Input<'a> = (&'a Array1<f64>, &'a Array1<f64>, &'a Array1<f64>);
    type Output = KramersKronigCheck;

    fn check<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        let (omega, eps_real, eps_imag) = input;
        let eps_inf = self.eps_inf;
        require_same_len(
            "KramersKronig eps_real vs frequency",
            omega.len(),
            eps_real.len(),
        )?;
        require_same_len(
            "KramersKronig eps_imag vs frequency",
            omega.len(),
            eps_imag.len(),
        )?;
        if omega.len() < 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: omega.len(),
                what: "KramersKronig needs at least 3 frequency points",
            });
        }

        let recovered = discrete_kramers_kronig(omega, eps_imag, eps_inf);
        let sum_abs: f64 = recovered
            .iter()
            .zip(eps_real.iter())
            .map(|(r, e)| (r - e).abs())
            .sum();
        let mae = sum_abs / (eps_real.len() as f64);
        // Tolerance scales with the dynamic range of the real part so small
        // absolute residuals on bounded dielectrics are forgiven.
        let dynamic_range = eps_real.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - eps_real.iter().copied().fold(f64::INFINITY, f64::min);
        let tol = (dynamic_range.abs() * 0.1).max(1e-2);
        Ok(KramersKronigCheck {
            recovered,
            mae,
            passed: mae < tol,
        })
    }
}

/// Outcome of the conductivity sum-rule check.
#[derive(Debug, Clone, Copy)]
pub struct SumRuleCheck {
    /// Trapezoidal ∫ σ(ω) dω.
    pub integral: f64,
    /// (π/2)·⟨J²⟩/(3 V k_B T).
    pub expected: f64,
    /// `(integral - expected) / |expected|`.
    pub relative_error: f64,
    /// Whether `|relative_error| < 0.05`.
    pub passed: bool,
}

/// Conductivity sum rule ∫₀^∞ σ(ω) dω = (π/2)·⟨J²⟩/(3 V k_B T).
///
/// Requires ≥ 2 frequency points, positive finite `volume` / `temperature`,
/// and a finite non-negative `current_sq_mean`.
#[derive(Debug, Clone, Copy)]
pub struct ConductivitySumRule {
    /// ⟨J²⟩, the mean squared current.
    pub current_sq_mean: f64,
    /// Cell volume.
    pub volume: f64,
    /// Temperature.
    pub temperature: f64,
}

impl Check for ConductivitySumRule {
    /// `(omega, sigma)`.
    type Input<'a> = (&'a Array1<f64>, &'a Array1<f64>);
    type Output = SumRuleCheck;

    fn check<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        let (omega, sigma) = input;
        let Self {
            current_sq_mean,
            volume,
            temperature,
        } = *self;
        require_same_len(
            "sum_rule conductivity vs frequency",
            omega.len(),
            sigma.len(),
        )?;
        require_positive("volume", volume)?;
        require_positive("temperature", temperature)?;
        if !current_sq_mean.is_finite() || current_sq_mean < 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "current_sq_mean",
                value: current_sq_mean.to_string(),
            });
        }
        if omega.len() < 2 {
            return Err(ComputeError::DimensionMismatch {
                expected: 2,
                got: omega.len(),
                what: "conductivity sum rule needs at least 2 frequency points",
            });
        }

        let mut integral = 0.0;
        for i in 1..omega.len() {
            let dw = omega[i] - omega[i - 1];
            integral += 0.5 * (sigma[i] + sigma[i - 1]) * dw;
        }
        let expected = std::f64::consts::PI * 0.5 * current_sq_mean
            / (3.0 * volume * UnitPreset::real().boltzmann() * temperature);
        let denom = expected.abs().max(1e-30);
        let relative_error = (integral - expected) / denom;
        Ok(SumRuleCheck {
            integral,
            expected,
            relative_error,
            passed: relative_error.abs() < 0.05,
        })
    }
}

/// Outcome of the route-agreement check.
#[derive(Debug, Clone)]
pub struct RouteAgreementCheck {
    /// `(pair_label, relative_rms)` for each unordered pair, in input order.
    pub pairwise: Vec<(String, f64)>,
    /// Largest pairwise relative RMS.
    pub max_rms: f64,
    /// Whether `max_rms < 0.10`.
    pub passed: bool,
}

/// Pairwise relative-RMS agreement between named ε(ω) arrays — e.g. the same
/// quantity computed by the Einstein-Helfand and Green-Kubo routes.
///
/// Requires ≥ 2 equal-length arrays.
#[derive(Debug, Clone, Copy, Default)]
pub struct RouteAgreement;

impl Check for RouteAgreement {
    /// Named result arrays, one per route.
    type Input<'a> = &'a [(String, Array1<f64>)];
    type Output = RouteAgreementCheck;

    fn check<'a>(&self, entries: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        if entries.len() < 2 {
            return Err(ComputeError::DimensionMismatch {
                expected: 2,
                got: entries.len(),
                what: "route agreement needs at least two named result arrays",
            });
        }
        let expected_len = entries[0].1.len();
        for (_name, arr) in entries {
            require_same_len("RouteAgreement array lengths", expected_len, arr.len())?;
        }

        let mut pairwise = Vec::new();
        let mut max_rms = 0.0_f64;
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let (name_i, arr_i) = &entries[i];
                let (name_j, arr_j) = &entries[j];
                let mut sum_sq = 0.0;
                let mut norm = 0.0;
                for k in 0..expected_len {
                    let diff = arr_i[k] - arr_j[k];
                    sum_sq += diff * diff;
                    norm += 0.5 * (arr_i[k].abs() + arr_j[k].abs());
                }
                let rms_abs = (sum_sq / (expected_len as f64)).sqrt();
                let scale = (norm / (expected_len as f64)).max(1e-30);
                let rms_rel = rms_abs / scale;
                max_rms = max_rms.max(rms_rel);
                pairwise.push((format!("{name_i}_vs_{name_j}"), rms_rel));
            }
        }
        Ok(RouteAgreementCheck {
            pairwise,
            max_rms,
            passed: max_rms < 0.10,
        })
    }
}

fn require_same_len(what: &'static str, expected: usize, got: usize) -> Result<(), ComputeError> {
    if expected != got {
        return Err(ComputeError::DimensionMismatch {
            expected,
            got,
            what,
        });
    }
    Ok(())
}

fn require_positive(field: &'static str, value: f64) -> Result<(), ComputeError> {
    if !(value.is_finite() && value > 0.0) {
        return Err(ComputeError::OutOfRange {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

impl Verdict for KramersKronigCheck {
    fn passed(&self) -> bool {
        self.passed
    }
}
impl ComputeResult for KramersKronigCheck {}

impl Verdict for SumRuleCheck {
    fn passed(&self) -> bool {
        self.passed
    }
}
impl ComputeResult for SumRuleCheck {}

impl Verdict for RouteAgreementCheck {
    fn passed(&self) -> bool {
        self.passed
    }
}
impl ComputeResult for RouteAgreementCheck {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::Compute;
    use crate::compute::{DebyeRelaxation, EwaldBoundary, GreenKuboConductivity};
    use molrs::core::Frame;
    use ndarray::Array2;
    use rustfft::FftPlanner;

    /// Empty frame slice for the series-based raw computes.
    fn no_frames() -> Vec<&'static Frame> {
        Vec::new()
    }

    fn rng_dipole(n: usize, seed: u64) -> Array2<f64> {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut s = Array2::zeros((n, 3));
        for t in 0..n {
            for d in 0..3 {
                s[[t, d]] = rng.random::<f64>() - 0.5;
            }
        }
        s
    }

    // ── Reference implementations ──────────────────────────────────────────
    // Each spectrum written out inline in one function, an independent oracle
    // the raw-compute + Fit composition is locked to bit-for-bit (ac-001).

    /// Einstein–Helfand spectrum, inline: fluctuation ACF, cos² taper,
    /// derivative, one-sided FT, prefactor.
    #[allow(clippy::too_many_arguments)]
    fn reference_einstein_helfand(
        dipole_moments: &Array2<f64>,
        dt: f64,
        volume: f64,
        temperature: f64,
        epsilon_inf: f64,
        max_correlation_time: usize,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let n_frames = dipole_moments.shape()[0];
        let max_lag = max_correlation_time.min(n_frames - 1);

        // ⟨|δM|²⟩ — the exact static (ω = 0) term.
        let n = n_frames as f64;
        let mut mean_m = [0.0_f64; 3];
        for t in 0..n_frames {
            for d in 0..3 {
                mean_m[d] += dipole_moments[[t, d]];
            }
        }
        for m in mean_m.iter_mut() {
            *m /= n;
        }
        let mut m_sq = 0.0;
        for t in 0..n_frames {
            for d in 0..3 {
                let dev = dipole_moments[[t, d]] - mean_m[d];
                m_sq += dev * dev;
            }
        }
        m_sq /= n;

        // Fluctuation ACF, summed over components, unbiased estimator.
        let mut planner = FftPlanner::new();
        let mut acf = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..3 {
            let col: Array1<f64> = (0..n_frames)
                .map(|t| dipole_moments[[t, d]] - mean_m[d])
                .collect();
            let component = sig::acf_fft_with_planner(&mut planner, &col, max_lag).unwrap();
            for k in 0..=max_lag {
                acf[k] += component[k];
            }
        }
        for k in 0..=max_lag {
            acf[k] /= (n_frames - k) as f64;
        }

        // One-sided cos² taper.
        let denom = 2.0 * max_lag.max(1) as f64;
        for k in 0..=max_lag {
            let angle = std::f64::consts::PI * k as f64 / denom;
            acf[k] *= angle.cos().powi(2);
        }
        // Central finite-difference derivative.
        let mut deriv = Array1::<f64>::zeros(max_lag + 1);
        for k in 1..max_lag {
            deriv[k] = (acf[k + 1] - acf[k - 1]) / (2.0 * dt);
        }
        if max_lag >= 1 {
            deriv[max_lag] = (acf[max_lag] - acf[max_lag - 1]) / dt;
        }
        let (frequencies, dre, dim) =
            piecewise_linear_onesided_ft(&deriv, dt, DIELECTRIC_PAD_FACTOR);

        let prefactor =
            FOUR_THIRDS_PI * KAPPA / (volume * UnitPreset::real().boltzmann() * temperature);
        let n_freq = frequencies.len();
        let mut eps_real = Array1::zeros(n_freq);
        let mut eps_imag = Array1::zeros(n_freq);
        for j in 0..n_freq {
            eps_real[j] = epsilon_inf - prefactor * dre[j];
            eps_imag[j] = prefactor * dim[j];
        }
        eps_real[0] = epsilon_inf + prefactor * m_sq;
        eps_imag[0] = 0.0;
        (frequencies, eps_real, eps_imag)
    }

    /// The unbiased current ACF, FFT-based, over the post-NaN `start=1`
    /// series. Returned so a test can lock the Fit's transform tail to the
    /// reference tail on the *same* ACF (bit-for-bit), independent of the
    /// raw-compute estimator.
    fn reference_gk_acf(current: &Array2<f64>, max_lag: usize) -> Array1<f64> {
        let n_frames = current.shape()[0];
        let start = 1;
        let n_eff = n_frames - start;
        let mut planner = FftPlanner::new();
        let mut acf_sum = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..3 {
            let col: Array1<f64> = (start..n_frames).map(|t| current[[t, d]]).collect();
            let acf = sig::acf_fft_with_planner(&mut planner, &col, max_lag).unwrap();
            for k in 0..=max_lag {
                acf_sum[k] += acf[k];
            }
        }
        for k in 0..=max_lag {
            acf_sum[k] /= (n_eff - k) as f64;
        }
        acf_sum
    }

    /// Green–Kubo spectrum tail on a pre-built ACF, inline: window, one-sided
    /// FT, σ→ε.
    fn reference_green_kubo_from_acf(
        acf: &Array1<f64>,
        dt: f64,
        volume: f64,
        temperature: f64,
        epsilon_inf: f64,
        window_type: &str,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let max_lag = acf.len() - 1;
        let wt = parse_window_type(window_type).unwrap();
        let acf_dyn =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[max_lag + 1]), acf.to_vec()).unwrap();
        let windowed = sig::apply_window(&acf_dyn, wt, 0).unwrap();
        let windowed_1d: Array1<f64> = windowed.iter().copied().collect();
        let (frequencies, spec_re, spec_im) =
            piecewise_linear_onesided_ft(&windowed_1d, dt, DIELECTRIC_PAD_FACTOR);

        let sigma_prefactor = volume / (3.0 * UnitPreset::real().boltzmann() * temperature);
        let eps0_factor = FOUR_PI * KAPPA;
        let n_freq = frequencies.len();
        let mut eps_real = Array1::zeros(n_freq);
        let mut eps_imag = Array1::zeros(n_freq);
        for j in 0..n_freq {
            let omega = frequencies[j];
            if omega == 0.0 {
                eps_real[j] = epsilon_inf;
                eps_imag[j] = 0.0;
            } else {
                let sigma_re = sigma_prefactor * spec_re[j];
                let sigma_im = sigma_prefactor * spec_im[j];
                eps_real[j] = epsilon_inf + eps0_factor * sigma_im / omega;
                eps_imag[j] = eps0_factor * sigma_re / omega;
            }
        }
        (frequencies, eps_real, eps_imag)
    }

    /// FFT / fused-scale paths may differ by a few ULP from the hand-rolled
    /// reference; keep a tight absolute tolerance.
    fn assert_close_1d(got: &Array1<f64>, expected: &Array1<f64>, tol: f64, what: &str) {
        assert_eq!(got.len(), expected.len(), "{what} length");
        for k in 0..got.len() {
            let d = (got[k] - expected[k]).abs();
            assert!(
                d <= tol,
                "{what}[{k}]: |{got} − {exp}| = {d} > {tol}",
                got = got[k],
                exp = expected[k],
            );
        }
    }

    #[test]
    fn eh_fit_matches_the_reference_bit_for_bit() {
        // ac-001: DebyeRelaxation raw ACF + EinsteinHelfandSpectrum matches the
        // inline reference (tight ULP tolerance).
        let n = 256;
        let dt = 0.001;
        let (vol, temp, eps_inf) = (1000.0, 300.0, 1.5);
        let mct = 50;
        let dm = rng_dipole(n, 42);

        let (freq_l, re_l, im_l) = reference_einstein_helfand(&dm, dt, vol, temp, eps_inf, mct);

        let raw = DebyeRelaxation {
            volume: vol,
            temperature: temp,
            boundary: EwaldBoundary::TinFoil,
        }
        .compute(&no_frames(), (&dm, dt, mct))
        .unwrap();
        let fit = EinsteinHelfandSpectrum {
            dt,
            volume: vol,
            temperature: temp,
            epsilon_inf: eps_inf,
            zero_lag_variance: raw.zero_lag_variance,
        }
        .fit(&raw.acf)
        .unwrap();

        assert_eq!(fit.frequencies, freq_l);
        // Shared FFT primitives may reassociate by a few ULP vs the inlined
        // reference; 1e-12 absolute is still machine-precision tight.
        assert_close_1d(&fit.eps_real, &re_l, 1e-12, "eps_real");
        assert_close_1d(&fit.eps_imag, &im_l, 1e-12, "eps_imag");
    }

    #[test]
    fn gk_fit_matches_the_reference_bit_for_bit() {
        // ac-001: on the SAME raw current ACF, GreenKuboSpectrum matches the
        // reference transform tail bit-for-bit. The raw ACF is the unbiased
        // current ACF over the start=1 (post-NaN) series.
        let n = 256;
        let dt = 0.001;
        let (vol, temp, eps_inf) = (1000.0, 300.0, 1.0);
        let mct = 50;
        // Current density with NaN row 0 (as current_density emits).
        let mut current = rng_dipole(n, 7);
        for d in 0..3 {
            current[[0, d]] = f64::NAN;
        }
        let start = 1;
        let effective_len = n - start;
        let max_lag = mct.min(effective_len.saturating_sub(1));
        let raw_acf = reference_gk_acf(&current, max_lag);

        for window in ["hann", "blackman", "cosine_sq"] {
            let (freq_l, re_l, im_l) =
                reference_green_kubo_from_acf(&raw_acf, dt, vol, temp, eps_inf, window);

            let fit = GreenKuboSpectrum {
                dt,
                volume: vol,
                temperature: temp,
                epsilon_inf: eps_inf,
                window_type: window.to_string(),
            }
            .fit(&raw_acf)
            .unwrap();

            assert_eq!(fit.frequencies, freq_l, "window={window}");
            assert_eq!(fit.eps_real, re_l, "window={window}");
            assert_eq!(fit.eps_imag, im_l, "window={window}");
        }
    }

    #[test]
    fn gk_raw_compute_acf_matches_the_reference_fft_acf() {
        // The GreenKuboConductivity raw compute (direct-summation estimator) and
        // the reference FFT-based ACF agree to FP tolerance on the same series,
        // so composing the raw compute with GreenKuboSpectrum matches the
        // reference ε(ω) within that tolerance.
        let n = 256;
        let dt = 0.001;
        let mct = 50;
        let mut current = rng_dipole(n, 7);
        for d in 0..3 {
            current[[0, d]] = f64::NAN;
        }
        let start = 1;
        let effective_len = n - start;
        let max_lag = mct.min(effective_len.saturating_sub(1));
        let reference = reference_gk_acf(&current, max_lag);

        let post: Array2<f64> = current.slice(ndarray::s![start.., ..]).to_owned();
        let raw = GreenKuboConductivity
            .compute(&no_frames(), (&post, dt, max_lag))
            .unwrap();
        assert_eq!(raw.jacf.len(), reference.len());
        for k in 0..reference.len() {
            assert!((raw.jacf[k] - reference[k]).abs() < 1e-9, "k={k}");
        }
    }

    #[test]
    fn eh_dc_bin_recovers_neumann_static() {
        // ac-006: ε(ω=0) recovers the Neumann static dielectric constant.
        use crate::compute::static_dielectric_constant;
        let n = 256;
        let dt = 0.001;
        let (vol, temp, eps_inf) = (1000.0, 300.0, 1.5);
        let dm = rng_dipole(n, 99);
        let static_eps = static_dielectric_constant(&dm, vol, temp, eps_inf).unwrap();

        let raw = DebyeRelaxation {
            volume: vol,
            temperature: temp,
            boundary: EwaldBoundary::TinFoil,
        }
        .compute(&no_frames(), (&dm, dt, 50))
        .unwrap();
        let fit = EinsteinHelfandSpectrum {
            dt,
            volume: vol,
            temperature: temp,
            epsilon_inf: eps_inf,
            zero_lag_variance: raw.zero_lag_variance,
        }
        .fit(&raw.acf)
        .unwrap();
        assert!((fit.eps_real[0] - static_eps).abs() < 1e-10);
        assert_eq!(fit.eps_imag[0], 0.0);
    }

    #[test]
    fn eh_loss_finite_to_nyquist() {
        // ac-006: the loss spectrum ε″ stays finite to Nyquist (derivative-FT).
        let n = 2048;
        let dt = 1.0;
        let dm = rng_dipole(n, 11);
        let raw = DebyeRelaxation {
            volume: 1000.0,
            temperature: 300.0,
            boundary: EwaldBoundary::TinFoil,
        }
        .compute(&no_frames(), (&dm, dt, 100))
        .unwrap();
        let fit = EinsteinHelfandSpectrum {
            dt,
            volume: 1000.0,
            temperature: 300.0,
            epsilon_inf: 1.0,
            zero_lag_variance: raw.zero_lag_variance,
        }
        .fit(&raw.acf)
        .unwrap();
        assert!(fit.eps_imag.iter().all(|x| x.is_finite()));
        assert!(fit.eps_real.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn fits_reject_short_acf() {
        let tiny = Array1::from_vec(vec![1.0]);
        assert!(matches!(
            EinsteinHelfandSpectrum {
                dt: 1.0,
                volume: 1.0,
                temperature: 1.0,
                epsilon_inf: 1.0,
                zero_lag_variance: 1.0,
            }
            .fit(&tiny),
            Err(ComputeError::EmptyInput)
        ));
        assert!(matches!(
            GreenKuboSpectrum {
                dt: 1.0,
                volume: 1.0,
                temperature: 1.0,
                epsilon_inf: 1.0,
                window_type: "hann".into(),
            }
            .fit(&tiny),
            Err(ComputeError::EmptyInput)
        ));
    }

    #[test]
    fn gk_rejects_bad_window() {
        let acf = Array1::from_vec(vec![1.0, 0.5, 0.25]);
        assert!(matches!(
            GreenKuboSpectrum {
                dt: 1.0,
                volume: 1.0,
                temperature: 1.0,
                epsilon_inf: 1.0,
                window_type: "nope".into(),
            }
            .fit(&acf),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    // ── consistency checks ──

    #[test]
    fn kk_recovers_eps_inf_for_zero_loss() {
        // ε'' ≡ 0 ⇒ the KK integral is zero, so the recovered real part is ε∞
        // everywhere; matching eps_real ⇒ mae 0, passed.
        let omega = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let eps_imag = Array1::zeros(4);
        let eps_real = Array1::from_elem(4, 2.5);
        let out = KramersKronig { eps_inf: 2.5 }
            .check((&omega, &eps_real, &eps_imag))
            .unwrap();
        assert!(out.mae < 1e-12, "mae {}", out.mae);
        assert!(out.passed);
    }

    #[test]
    fn kk_requires_three_points() {
        let omega = Array1::from_vec(vec![1.0, 2.0]);
        let z = Array1::zeros(2);
        assert!(
            KramersKronig { eps_inf: 1.0 }
                .check((&omega, &z, &z))
                .is_err()
        );
    }

    #[test]
    fn sum_rule_matches_constructed_integral() {
        // Constant σ over [0, 4]: trapezoidal integral = σ·4 = 8. Pick the
        // physical inputs so `expected` equals 8 ⇒ relative_error 0, passed.
        let omega = Array1::from_vec(vec![0.0, 2.0, 4.0]);
        let sigma = Array1::from_elem(3, 2.0);
        let volume = 100.0;
        let temperature = 300.0;
        let expected = 8.0;
        // expected = π/2 · ⟨J²⟩ / (3 V k_B T)  ⇒  ⟨J²⟩ = expected·3 V k_B T / (π/2)
        let current_sq_mean =
            expected * 3.0 * volume * UnitPreset::real().boltzmann() * temperature
                / (std::f64::consts::PI * 0.5);
        let out = ConductivitySumRule {
            current_sq_mean,
            volume,
            temperature,
        }
        .check((&omega, &sigma))
        .unwrap();
        assert!((out.integral - 8.0).abs() < 1e-12);
        assert!(out.relative_error.abs() < 1e-12, "{}", out.relative_error);
        assert!(out.passed);
    }

    #[test]
    fn sum_rule_rejects_nonpositive_volume() {
        let omega = Array1::from_vec(vec![0.0, 1.0]);
        let sigma = Array1::from_elem(2, 1.0);
        assert!(
            ConductivitySumRule {
                current_sq_mean: 1.0,
                volume: 0.0,
                temperature: 300.0
            }
            .check((&omega, &sigma))
            .is_err()
        );
    }

    #[test]
    fn route_agreement_identical_is_zero() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let entries = vec![("r1".to_string(), a.clone()), ("r2".to_string(), a)];
        let out = RouteAgreement.check(&entries).unwrap();
        assert_eq!(out.pairwise.len(), 1);
        assert_eq!(out.pairwise[0].0, "r1_vs_r2");
        assert!(out.max_rms < 1e-12);
        assert!(out.passed);
    }

    #[test]
    fn route_agreement_flags_divergence() {
        let a = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let b = Array1::from_vec(vec![2.0, 2.0, 2.0]);
        let entries = vec![("a".to_string(), a), ("b".to_string(), b)];
        let out = RouteAgreement.check(&entries).unwrap();
        assert!(out.max_rms > 0.1);
        assert!(!out.passed);
    }
}
