//! The frequency-dependent permittivity ε(ω): four routes from a dipole or
//! current correlation, one shared continuous transform.

use molrs::core::UnitPreset;
use molrs::op::F;
use ndarray::{Array1, ArrayView1, Axis, s};
use num_complex::Complex64;

use crate::compute::dielectric::DielectricPrefactor;
use crate::compute::{
    Compute, ComputeError, CumulativeTrapezoid, Derivative, DerivativeInput, EwaldBoundary,
    FrequencyGrid, Plateau, Spectrum, Stencil, TimeCorrelation, Window,
};

/// The frequency-dependent permittivity `ε*(ω) = ε′(ω) − i·ε″(ω)`.
///
/// `eps_imag` stores `ε″(ω)` with the **positive-loss** convention (≥ 0 for
/// stable causal systems). FT convention throughout:
/// `X(ω) = ∫₀^∞ f(t) e^{−iωt} dt`.
///
/// # Units
///
/// ε is dimensionless. The angular frequencies are in radians per unit of
/// the correlation's lag spacing: no prefactor carries a time unit, so any
/// consistent time unit (fs in the `real` preset, ps, …) gives the same ε at
/// the same physical frequency. Volume, dipole and charge are in the units of
/// the route's [`UnitPreset`] (`real`: Å³, e·Å, e).
#[derive(Debug, Clone)]
pub struct DielectricSpectrumResult {
    /// Angular frequency grid, `n_fft / 2 + 1` points from DC to Nyquist
    /// `π / dt`, with `n_fft = (pad · n_lags).next_power_of_two()`.
    pub frequencies: Array1<F>,
    /// Real part ε′(ω).
    pub eps_real: Array1<F>,
    /// Loss spectrum ε″(ω), positive-loss convention.
    pub eps_imag: Array1<F>,
    /// The high-frequency permittivity ε_∞ the spectrum is offset by.
    pub epsilon_inf: F,
}

/// The correlation and thermodynamic state an ε(ω) route transforms.
#[derive(Debug, Clone, Copy)]
pub struct DielectricSpectrumInput<'a> {
    /// The route's correlation, one channel (see each route).
    pub correlation: &'a TimeCorrelation,
    /// Cell volume.
    pub volume: F,
    /// Temperature of the state.
    pub temperature: F,
    /// Ewald boundary the trajectory was sampled under.
    pub boundary: EwaldBoundary,
}

/// `(frequencies, re, im)` of a continuous one-sided transform.
type RawSpectrum = (Array1<F>, Array1<F>, Array1<F>);

/// The configuration every route shares: the window and zero padding of the
/// transform, the unit system of the prefactor, and ε_∞.
#[derive(Debug, Clone)]
struct RouteConfig {
    spectrum: Spectrum,
    units: UnitPreset,
    epsilon_inf: F,
}

/// The validated pieces of a route's input.
struct Prepared<'a> {
    acf: ArrayView1<'a, F>,
    dt: F,
    prefactor: DielectricPrefactor,
}

impl RouteConfig {
    /// 4× zero padding, ε_∞ = 1.
    fn new(window: Window, units: UnitPreset) -> Self {
        Self {
            spectrum: Spectrum::new(window, 4).expect("pad 4 is valid"),
            units,
            epsilon_inf: 1.0,
        }
    }

    fn with_pad(&mut self, pad: usize) -> Result<(), ComputeError> {
        self.spectrum = Spectrum::new(self.spectrum.window(), pad)?;
        Ok(())
    }

    fn prepare<'a>(
        &self,
        input: &DielectricSpectrumInput<'a>,
    ) -> Result<Prepared<'a>, ComputeError> {
        let acf = input.correlation.require_single_channel()?;
        if acf.len() < 2 {
            return Err(ComputeError::EmptyInput);
        }
        let dt = input.correlation.sample_interval()?;
        if !self.epsilon_inf.is_finite() {
            return Err(ComputeError::NonFinite {
                where_: "epsilon_inf",
                index: 0,
            });
        }
        input.boundary.require_supported()?;
        let prefactor = DielectricPrefactor::new(&self.units, input.volume, input.temperature)?;
        Ok(Prepared { acf, dt, prefactor })
    }

    /// The configured window applied to `acf`, then the continuous
    /// transform.
    fn windowed_transform(
        &self,
        acf: ArrayView1<'_, F>,
        dt: F,
    ) -> Result<RawSpectrum, ComputeError> {
        let windowed = self.spectrum.window().apply(acf);
        self.continuous_transform(&windowed, dt)
    }

    /// Continuous one-sided FT of the **piecewise-linear interpolant** of the
    /// samples `y[k]` at `k · dt`,
    ///
    /// ```text
    ///     X(ω) = ∫₀^T y_lin(t) e^{−iωt} dt
    /// ```
    ///
    /// in closed form through the FFT of the first differences (the DRS
    /// `one_sided_linear_transform`). The rectangle rule `rfft(y)·dt`
    /// inflates the low-frequency loss of a truncated correlation by factors
    /// of 3–15 on neat water; the piecewise-linear integral removes that bias.
    /// The DC bin is the trapezoid integral. No physical prefactor applied.
    fn continuous_transform(&self, y: &Array1<F>, dt: F) -> Result<RawSpectrum, ComputeError> {
        let n = y.len();
        let n_fft = (self.spectrum.pad() * n).next_power_of_two();
        let dy: Vec<F> = (0..n - 1).map(|k| y[k + 1] - y[k]).collect();
        let dy_bins = Spectrum::onesided(&dy, n_fft);
        let frequencies = FrequencyGrid::new(n_fft, dt)?.angular();
        let n_freq = frequencies.len();
        let mut spec_re = Array1::zeros(n_freq);
        let mut spec_im = Array1::zeros(n_freq);

        spec_re[0] = CumulativeTrapezoid::total(y.as_slice().expect("contiguous"), dt);

        let y0 = y[0];
        let y_last = y[n - 1];
        let t_end = (n - 1) as F * dt;
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
        Ok((frequencies, spec_re, spec_im))
    }
}

/// The builder methods every route shares.
macro_rules! route_config_methods {
    () => {
        /// Zero-pad the transform to `pad` times the correlation length
        /// (rounded up to a power of two). Default 4.
        ///
        /// # Errors
        /// [`ComputeError::OutOfRange`] for `pad == 0`.
        pub fn with_pad(mut self, pad: usize) -> Result<Self, ComputeError> {
            self.config.with_pad(pad)?;
            Ok(self)
        }

        /// The high-frequency (electronic) permittivity ε_∞. Default 1.
        pub fn with_epsilon_inf(mut self, epsilon_inf: F) -> Self {
            self.config.epsilon_inf = epsilon_inf;
            self
        }

        /// The window and zero padding.
        pub fn spectrum(&self) -> Spectrum {
            self.config.spectrum
        }

        /// The unit system of the prefactor.
        pub fn units(&self) -> &UnitPreset {
            &self.config.units
        }

        /// ε_∞.
        pub fn epsilon_inf(&self) -> F {
            self.config.epsilon_inf
        }
    };
}

// ── Einstein–Helfand route ───────────────────────────────────────────────────

/// Einstein–Helfand (dipole-ACF) route to ε(ω).
///
/// Its correlation is the **fluctuation dipole ACF** `C(t) = ⟨δM(0)·δM(t)⟩`
/// summed over the Cartesian components — the
/// [`DipoleAcf`](crate::compute::DipoleAcf) curve. It implements the
/// integration-by-parts form of Caillol–Levesque–Weis Eq. (30):
///
/// ```text
///     ε*(ω) − ε_∞ = −A · Ĉ′(ω),     A = 4π κ / (3 V k_B T)
///     Ĉ′(ω) = ∫₀^∞ C′(t) e^{−iωt} dt
/// ```
///
/// so `ε′ − ε_∞ = −A Ĉ′_re` and `ε″ = A Ĉ′_im`. The correlation is tapered
/// by the window (use [`Window::CosineSq`]: 1 at `C(0)`, 0 at the last lag,
/// which keeps `Ĉ′` decaying toward Nyquist) and differentiated with the
/// [`Derivative`] stencils — `C′(0) = 0` because the ACF is even, central
/// differences inside, a backward difference at the last lag. The DC bin is
/// the exact Neumann static limit `ε_∞ + A·C(0)`, matching
/// [`StaticDielectric`](crate::compute::StaticDielectric).
///
/// References: Caillol, Levesque & Weis, *J. Chem. Phys.* **85**, 6645
/// (1986), Eq. (30); Neumann, *Mol. Phys.* **50**, 841 (1983).
#[derive(Debug, Clone)]
pub struct EinsteinHelfandSpectrum {
    config: RouteConfig,
}

impl EinsteinHelfandSpectrum {
    /// The route with taper `window`, its prefactor in `units`.
    pub fn new(window: Window, units: UnitPreset) -> Self {
        Self {
            config: RouteConfig::new(window, units),
        }
    }

    route_config_methods!();

    /// Taper, then `C′(t)` on the [`Derivative`] stencils.
    fn tapered_derivative(&self, acf: ArrayView1<'_, F>, dt: F) -> Result<Array1<F>, ComputeError> {
        let tapered = self.config.spectrum.window().apply(acf);
        let n = tapered.len();
        let mut deriv = Array1::<F>::zeros(n);
        if n >= 3 {
            let central = Derivative::new(Stencil::Central).compute(DerivativeInput {
                series: tapered.view().insert_axis(Axis(1)),
                dt,
            })?;
            deriv
                .slice_mut(s![1..n - 1])
                .assign(&central.index_axis(Axis(1), 0));
        }
        let tail = Derivative::new(Stencil::Backward).compute(DerivativeInput {
            series: tapered.slice(s![n - 2..]).insert_axis(Axis(1)),
            dt,
        })?;
        deriv[n - 1] = tail[[0, 0]];
        Ok(deriv)
    }
}

impl Compute for EinsteinHelfandSpectrum {
    type Input<'a> = DielectricSpectrumInput<'a>;
    type Output = DielectricSpectrumResult;

    /// # Errors
    /// * [`ComputeError::EmptyInput`] for fewer than two lags.
    /// * [`ComputeError::DimensionMismatch`] for a correlation with more than
    ///   one channel.
    /// * [`ComputeError::OutOfRange`] for a non-positive volume, temperature
    ///   or lag spacing, or a vacuum boundary.
    fn compute(
        &self,
        input: DielectricSpectrumInput<'_>,
    ) -> Result<DielectricSpectrumResult, ComputeError> {
        let Prepared { acf, dt, prefactor } = self.config.prepare(&input)?;
        let deriv = self.tapered_derivative(acf, dt)?;
        let (frequencies, dre, dim) = self.config.continuous_transform(&deriv, dt)?;
        let a = prefactor.fluctuation();
        let eps_inf = self.config.epsilon_inf;
        let mut eps_real = dre.mapv(|re| eps_inf - a * re);
        let mut eps_imag = dim.mapv(|im| a * im);
        // DC bin: the exact Neumann static limit; the discrete derivative
        // transform reproduces it only up to O(dt).
        eps_real[0] = eps_inf + a * acf[0];
        eps_imag[0] = 0.0;
        Ok(DielectricSpectrumResult {
            frequencies,
            eps_real,
            eps_imag,
            epsilon_inf: eps_inf,
        })
    }
}

// ── Green–Kubo route ─────────────────────────────────────────────────────────

/// Green–Kubo (current-ACF) route to ε(ω).
///
/// Its correlation is the **current-density ACF** `⟨J(0)·J(t)⟩`, `J = Ṁ / V`,
/// summed over the Cartesian components — the
/// [`CurrentAcf`](crate::compute::CurrentAcf) of the
/// [`Derivative`] of the [`DipoleMoment`](crate::compute::DipoleMoment)
/// series divided by the volume. It implements
///
/// ```text
///     σ(ω) = (V / (3 k_B T)) · ∫₀^∞ ⟨J(0)·J(t)⟩ e^{−iωt} dt = σ′ + i σ″
///     ε*(ω) − ε_∞ = −i σ(ω) / (ω ε₀)
/// ```
///
/// so `ε′ − ε_∞ = σ″ / (ω ε₀)` and `ε″ = σ′ / (ω ε₀)`, `1/ε₀ = 4π κ`. The DC
/// bin (`σ/ω = 0/0`) is `(ε_∞, 0)`; the static limit is
/// [`StaticDielectric`](crate::compute::StaticDielectric) for insulators.
///
/// References: Hansen & McDonald, *Theory of Simple Liquids*, Eq. 7.7.20;
/// Caillol, Levesque & Weis, *J. Chem. Phys.* **85**, 6645 (1986), Eqs.
/// (36)–(39).
#[derive(Debug, Clone)]
pub struct GreenKuboSpectrum {
    config: RouteConfig,
}

impl GreenKuboSpectrum {
    /// The route with `window` on the correlation, its prefactor in `units`.
    pub fn new(window: Window, units: UnitPreset) -> Self {
        Self {
            config: RouteConfig::new(window, units),
        }
    }

    route_config_methods!();
}

impl Compute for GreenKuboSpectrum {
    type Input<'a> = DielectricSpectrumInput<'a>;
    type Output = DielectricSpectrumResult;

    /// # Errors
    /// As [`EinsteinHelfandSpectrum`].
    fn compute(
        &self,
        input: DielectricSpectrumInput<'_>,
    ) -> Result<DielectricSpectrumResult, ComputeError> {
        let Prepared { acf, dt, prefactor } = self.config.prepare(&input)?;
        let (frequencies, spec_re, spec_im) = self.config.windowed_transform(acf, dt)?;
        let sigma_prefactor = prefactor.current_density();
        let eps0_factor = prefactor.inverse_permittivity();
        let eps_inf = self.config.epsilon_inf;
        let n_freq = frequencies.len();
        let mut eps_real = Array1::zeros(n_freq);
        let mut eps_imag = Array1::zeros(n_freq);
        for j in 0..n_freq {
            let omega = frequencies[j];
            if omega == 0.0 {
                eps_real[j] = eps_inf;
                eps_imag[j] = 0.0;
            } else {
                let sigma_re = sigma_prefactor * spec_re[j];
                let sigma_im = sigma_prefactor * spec_im[j];
                eps_real[j] = eps_inf + eps0_factor * sigma_im / omega;
                eps_imag[j] = eps0_factor * sigma_re / omega;
            }
        }
        Ok(DielectricSpectrumResult {
            frequencies,
            eps_real,
            eps_imag,
            epsilon_inf: eps_inf,
        })
    }
}

// ── Dipole-rate × dipole cross route ─────────────────────────────────────────

/// Permittivity from the **dipole-rate × dipole** cross-correlation
/// `C_{ṀM}(t) = Σ_α ⟨δṀ_α(0) δM_α(t)⟩` — a
/// [`Correlator`](crate::compute::Correlator) `Product` cross-correlation with
/// the [`Derivative`] of the dipole series as `a` and the dipole series as
/// `b`.
///
/// ```text
///     χ(ω) = A · Ĉ_{ṀM}(ω),     A = 4π κ / (3 V k_B T)
///     ε*(ω) − ε_∞ = χ(ω)
/// ```
///
/// with the same windowed continuous transform as [`GreenKuboSpectrum`];
/// `ε′ = ε_∞ + Re χ`, `ε″ = −Im χ`. The MD realisation of the ⟨Ṗ(0)·P(t)⟩
/// susceptibility; numerically distinct from the Green–Kubo route.
#[derive(Debug, Clone)]
pub struct DipoleRateCrossSpectrum {
    config: RouteConfig,
}

impl DipoleRateCrossSpectrum {
    /// The route with `window` on the correlation, its prefactor in `units`.
    pub fn new(window: Window, units: UnitPreset) -> Self {
        Self {
            config: RouteConfig::new(window, units),
        }
    }

    route_config_methods!();
}

impl Compute for DipoleRateCrossSpectrum {
    type Input<'a> = DielectricSpectrumInput<'a>;
    type Output = DielectricSpectrumResult;

    /// # Errors
    /// As [`EinsteinHelfandSpectrum`].
    fn compute(
        &self,
        input: DielectricSpectrumInput<'_>,
    ) -> Result<DielectricSpectrumResult, ComputeError> {
        let Prepared { acf, dt, prefactor } = self.config.prepare(&input)?;
        let (frequencies, re, im) = self.config.windowed_transform(acf, dt)?;
        let a = prefactor.fluctuation();
        let eps_inf = self.config.epsilon_inf;
        Ok(DielectricSpectrumResult {
            frequencies,
            eps_real: re.mapv(|re| eps_inf + a * re),
            eps_imag: im.mapv(|im| -a * im),
            epsilon_inf: eps_inf,
        })
    }
}

// ── Dipole autocorrelation (PACF) route ──────────────────────────────────────

/// Permittivity from the **fluctuation dipole autocorrelation**
/// `C(t) = Σ_α ⟨δM_α(0) δM_α(t)⟩` (the
/// [`DipoleAcf`](crate::compute::DipoleAcf) curve) via the Fourier identity
///
/// ```text
///     χ(ω) = A · [ C(0) − iω Ĉ(ω) ],     A = 4π κ / (3 V k_B T)
///     ε*(ω) − ε_∞ = χ(ω)
/// ```
///
/// Distinct from [`EinsteinHelfandSpectrum`], which transforms the tapered
/// `C′(t)` instead.
///
/// # Conductive plateau
///
/// For ionic conductors the PACF plateaus (`C(∞)/C(0) → O(1)`): the cell
/// dipole does not relax, and the bare formula gives a huge false static χ.
/// [`with_plateau`](Self::with_plateau) subtracts the curve's
/// [`Plateau`] value `C(∞)` first, applying the formula to the relaxational
/// `C(t) − C(∞)`. Whether to subtract is the caller's decision; the route
/// never guesses it from the data.
#[derive(Debug, Clone)]
pub struct DipoleAutocorrelationSpectrum {
    config: RouteConfig,
    plateau: Option<Plateau>,
}

impl DipoleAutocorrelationSpectrum {
    /// The route with `window` on the correlation, its prefactor in `units`,
    /// and no plateau subtraction.
    pub fn new(window: Window, units: UnitPreset) -> Self {
        Self {
            config: RouteConfig::new(window, units),
            plateau: None,
        }
    }

    route_config_methods!();

    /// Subtract the correlation's mean over this plateau window (its
    /// non-relaxing tail `C(∞)`) before the transform.
    pub fn with_plateau(mut self, plateau: Plateau) -> Self {
        self.plateau = Some(plateau);
        self
    }

    /// The plateau subtracted, if any.
    pub fn plateau(&self) -> Option<Plateau> {
        self.plateau
    }
}

impl Compute for DipoleAutocorrelationSpectrum {
    type Input<'a> = DielectricSpectrumInput<'a>;
    type Output = DielectricSpectrumResult;

    /// # Errors
    /// As [`EinsteinHelfandSpectrum`], and the [`Plateau`] errors when its
    /// window covers fewer than two lags.
    fn compute(
        &self,
        input: DielectricSpectrumInput<'_>,
    ) -> Result<DielectricSpectrumResult, ComputeError> {
        let Prepared { acf, dt, prefactor } = self.config.prepare(&input)?;
        let c_inf = match self.plateau {
            Some(plateau) => plateau.read(&acf.to_vec())?.value,
            None => 0.0,
        };
        let series = acf.mapv(|v| v - c_inf);
        let c0 = acf[0] - c_inf;
        let (frequencies, re, im) = self.config.windowed_transform(series.view(), dt)?;
        let a = prefactor.fluctuation();
        let eps_inf = self.config.epsilon_inf;
        let n_freq = frequencies.len();
        let mut eps_real = Array1::zeros(n_freq);
        let mut eps_imag = Array1::zeros(n_freq);
        for j in 0..n_freq {
            let omega = frequencies[j];
            // χ = A·(c0 − iω Ĉ), Ĉ = re + i·im:
            // Re χ = A·(c0 + ω·im), Im χ = −A·ω·re; ε′ = ε∞ + Re χ, ε″ = −Im χ.
            if omega == 0.0 {
                eps_real[j] = eps_inf + a * c0;
                eps_imag[j] = 0.0;
            } else {
                eps_real[j] = eps_inf + a * (c0 + omega * im[j]);
                eps_imag[j] = -(a * (-omega * re[j]));
            }
        }
        Ok(DielectricSpectrumResult {
            frequencies,
            eps_real,
            eps_imag,
            epsilon_inf: eps_inf,
        })
    }
}

#[cfg(test)]
mod tests;
