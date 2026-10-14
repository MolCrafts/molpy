//! Consistency checks of a dielectric spectrum: computes whose outputs are
//! [`Verdict`]s.

use molrs::core::UnitPreset;
use molrs::op::F;
use ndarray::{Array1, ArrayView1};

use super::DielectricSpectrumResult;
use crate::compute::dielectric::DielectricPrefactor;
use crate::compute::named_enum::named_enum;
use crate::compute::{Compute, ComputeError, CumulativeTrapezoid, Verdict};

// ── Kramers–Kronig ───────────────────────────────────────────────────────────

/// Outcome of the Kramers–Kronig consistency check.
#[derive(Debug, Clone)]
pub struct KramersKronigCheck {
    /// ε′(ω) recovered from ε″(ω).
    pub recovered: Array1<F>,
    /// Mean absolute error against the spectrum's real part.
    pub mae: F,
    /// The tolerance `mae` was held to.
    pub tolerance: F,
    /// Whether `mae < tolerance`.
    pub passed: bool,
}

impl Verdict for KramersKronigCheck {
    fn passed(&self) -> bool {
        self.passed
    }
}

/// Kramers–Kronig causality of a dielectric spectrum: ε′(ω) is recovered from
/// ε″(ω) by the discrete principal-value integral
///
/// ```text
/// ε′(ω_i) = ε_∞ + (2/π) Σ_{j≠i} ε″(ω_j) ω_j / (ω_j² − ω_i²) Δω_j
/// ```
///
/// and compared with the spectrum's own real part. The check passes when the
/// mean absolute error is below
/// `max(relative_tolerance · (max ε′ − min ε′), absolute_tolerance)`, so
/// small residuals on a spectrum of small dynamic range are forgiven.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KramersKronig {
    relative_tolerance: F,
    absolute_tolerance: F,
}

/// The spectrum a [`KramersKronig`] check reads.
#[derive(Debug, Clone, Copy)]
pub struct KramersKronigInput<'a> {
    /// The ε(ω) of any route; its ε_∞ offsets the recovered real part.
    pub spectrum: &'a DielectricSpectrumResult,
}

impl Default for KramersKronig {
    /// Relative tolerance 0.1 of the dynamic range, absolute floor 0.01.
    fn default() -> Self {
        Self {
            relative_tolerance: 0.1,
            absolute_tolerance: 1e-2,
        }
    }
}

impl KramersKronig {
    /// The check with its default tolerances.
    pub fn new() -> Self {
        Self::default()
    }

    /// Tolerance `max(relative · dynamic range, absolute)`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] unless both are finite and non-negative.
    pub fn with_tolerance(mut self, relative: F, absolute: F) -> Result<Self, ComputeError> {
        require_tolerance("relative_tolerance", relative)?;
        require_tolerance("absolute_tolerance", absolute)?;
        self.relative_tolerance = relative;
        self.absolute_tolerance = absolute;
        Ok(self)
    }

    /// `(relative, absolute)` tolerances.
    pub fn tolerance(&self) -> (F, F) {
        (self.relative_tolerance, self.absolute_tolerance)
    }

    fn recover(omega: ArrayView1<'_, F>, eps_imag: ArrayView1<'_, F>, eps_inf: F) -> Array1<F> {
        let n = omega.len();
        let mut recovered = Array1::<F>::from_elem(n, eps_inf);
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
                // Principal-value quadrature weight of ω_j (half the span of
                // its neighbours; the full spacing at either end).
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
}

impl Compute for KramersKronig {
    type Input<'a> = KramersKronigInput<'a>;
    type Output = KramersKronigCheck;

    /// # Errors
    /// [`ComputeError::DimensionMismatch`] for fewer than three frequencies or
    /// parts of another length than the grid.
    fn compute(&self, input: KramersKronigInput<'_>) -> Result<KramersKronigCheck, ComputeError> {
        let s = input.spectrum;
        require_same_len(
            "KramersKronig eps_real vs frequency",
            s.frequencies.len(),
            s.eps_real.len(),
        )?;
        require_same_len(
            "KramersKronig eps_imag vs frequency",
            s.frequencies.len(),
            s.eps_imag.len(),
        )?;
        if s.frequencies.len() < 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: s.frequencies.len(),
                what: "KramersKronig needs at least 3 frequency points",
            });
        }
        let recovered = Self::recover(s.frequencies.view(), s.eps_imag.view(), s.epsilon_inf);
        let mae = recovered
            .iter()
            .zip(s.eps_real.iter())
            .map(|(r, e)| (r - e).abs())
            .sum::<F>()
            / s.eps_real.len() as F;
        let dynamic_range = s.eps_real.iter().copied().fold(F::NEG_INFINITY, F::max)
            - s.eps_real.iter().copied().fold(F::INFINITY, F::min);
        let tolerance =
            (dynamic_range.abs() * self.relative_tolerance).max(self.absolute_tolerance);
        Ok(KramersKronigCheck {
            recovered,
            mae,
            tolerance,
            passed: mae < tolerance,
        })
    }
}

// ── Conductivity sum rule ────────────────────────────────────────────────────

/// Outcome of the conductivity sum-rule check.
#[derive(Debug, Clone, Copy)]
pub struct SumRuleCheck {
    /// Trapezoidal ∫ σ(ω) dω.
    pub integral: F,
    /// `(π/2)·⟨J²⟩ / (3 V k_B T)`.
    pub expected: F,
    /// `(integral − expected) / |expected|`.
    pub relative_error: F,
    /// Whether `|relative_error| < tolerance`.
    pub passed: bool,
}

impl Verdict for SumRuleCheck {
    fn passed(&self) -> bool {
        self.passed
    }
}

/// The conductivity sum rule `∫₀^∞ σ(ω) dω = (π/2) ⟨J²⟩ / (3 V k_B T)`,
/// with `J` the total current `Ṁ` and `k_B` of the configured [`UnitPreset`].
#[derive(Debug, Clone)]
pub struct ConductivitySumRule {
    units: UnitPreset,
    tolerance: F,
}

/// The conductivity spectrum and state a [`ConductivitySumRule`] reads.
#[derive(Debug, Clone, Copy)]
pub struct SumRuleInput<'a> {
    /// Angular frequencies (ascending).
    pub frequencies: ArrayView1<'a, F>,
    /// σ(ω) on them.
    pub conductivity: ArrayView1<'a, F>,
    /// ⟨J²⟩, the mean squared total current (the current ACF at lag 0).
    pub current_sq_mean: F,
    /// Cell volume.
    pub volume: F,
    /// Temperature.
    pub temperature: F,
}

impl ConductivitySumRule {
    /// The relative error the sum rule tolerates when none is given.
    pub const DEFAULT_TOLERANCE: F = 0.05;

    /// The sum rule in `units`, passing at a relative error below 0.05.
    pub fn new(units: UnitPreset) -> Self {
        Self {
            units,
            tolerance: Self::DEFAULT_TOLERANCE,
        }
    }

    /// Pass when `|relative_error| < tolerance`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] unless `tolerance` is finite and
    /// non-negative.
    pub fn with_tolerance(mut self, tolerance: F) -> Result<Self, ComputeError> {
        require_tolerance("tolerance", tolerance)?;
        self.tolerance = tolerance;
        Ok(self)
    }

    /// The relative tolerance.
    pub fn tolerance(&self) -> F {
        self.tolerance
    }
}

impl Compute for ConductivitySumRule {
    type Input<'a> = SumRuleInput<'a>;
    type Output = SumRuleCheck;

    /// # Errors
    /// * [`ComputeError::DimensionMismatch`] for fewer than two frequencies or
    ///   a conductivity of another length.
    /// * [`ComputeError::OutOfRange`] for a non-positive volume or
    ///   temperature, or a negative or non-finite ⟨J²⟩.
    fn compute(&self, input: SumRuleInput<'_>) -> Result<SumRuleCheck, ComputeError> {
        let (omega, sigma) = (input.frequencies, input.conductivity);
        require_same_len(
            "sum_rule conductivity vs frequency",
            omega.len(),
            sigma.len(),
        )?;
        let prefactor = DielectricPrefactor::new(&self.units, input.volume, input.temperature)?;
        let csm = input.current_sq_mean;
        if !csm.is_finite() || csm < 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "current_sq_mean",
                value: csm.to_string(),
            });
        }
        if omega.len() < 2 {
            return Err(ComputeError::DimensionMismatch {
                expected: 2,
                got: omega.len(),
                what: "conductivity sum rule needs at least 2 frequency points",
            });
        }
        let sigma = sigma.to_vec();
        let integral = CumulativeTrapezoid::running(&sigma, |k| omega[k] - omega[k - 1])
            .last()
            .copied()
            .expect("at least two points");
        let expected = std::f64::consts::PI * 0.5 * csm * prefactor.total_current();
        let relative_error = (integral - expected) / expected.abs().max(1e-30);
        Ok(SumRuleCheck {
            integral,
            expected,
            relative_error,
            passed: relative_error.abs() < self.tolerance,
        })
    }
}

// ── Route agreement ──────────────────────────────────────────────────────────

named_enum! {
    /// Which part of a permittivity spectrum a comparison reads.
    #[derive(Default)]
    pub enum PermittivityPart {
        /// The loss spectrum ε″(ω).
        #[default]
        Imaginary => "imaginary",
        /// The real part ε′(ω).
        Real => "real",
    }
}

/// Outcome of the route-agreement check.
#[derive(Debug, Clone)]
pub struct RouteAgreementCheck {
    /// `(i, j, relative_rms)` for each pair `i < j` of input spectra, in
    /// input order.
    pub pairwise: Vec<(usize, usize, F)>,
    /// Largest pairwise relative RMS.
    pub max_rms: F,
    /// Whether `max_rms < tolerance`.
    pub passed: bool,
}

impl Verdict for RouteAgreementCheck {
    fn passed(&self) -> bool {
        self.passed
    }
}

/// Pairwise relative-RMS agreement of one part of several ε(ω) spectra on one
/// frequency grid — the same permittivity reached by different routes
/// (Einstein–Helfand, Green–Kubo, …). The relative RMS of a pair is
/// `√⟨(a − b)²⟩ / ⟨½(|a| + |b|)⟩`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteAgreement {
    part: PermittivityPart,
    tolerance: F,
}

/// The spectra a [`RouteAgreement`] compares.
#[derive(Debug, Clone, Copy)]
pub struct RouteAgreementInput<'a> {
    /// At least two spectra on the same frequency grid.
    pub spectra: &'a [&'a DielectricSpectrumResult],
}

impl Default for RouteAgreement {
    /// The loss spectrum, passing below a relative RMS of 0.10.
    fn default() -> Self {
        Self {
            part: PermittivityPart::default(),
            tolerance: 0.10,
        }
    }
}

impl RouteAgreement {
    /// The check with its defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compare this part of the spectra.
    pub fn with_part(mut self, part: PermittivityPart) -> Self {
        self.part = part;
        self
    }

    /// Pass when every pairwise relative RMS is below `tolerance`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] unless `tolerance` is finite and
    /// non-negative.
    pub fn with_tolerance(mut self, tolerance: F) -> Result<Self, ComputeError> {
        require_tolerance("tolerance", tolerance)?;
        self.tolerance = tolerance;
        Ok(self)
    }

    /// The part compared.
    pub fn part(&self) -> PermittivityPart {
        self.part
    }

    /// The tolerance.
    pub fn tolerance(&self) -> F {
        self.tolerance
    }

    fn values<'s>(&self, s: &'s DielectricSpectrumResult) -> &'s Array1<F> {
        match self.part {
            PermittivityPart::Imaginary => &s.eps_imag,
            PermittivityPart::Real => &s.eps_real,
        }
    }
}

impl Compute for RouteAgreement {
    type Input<'a> = RouteAgreementInput<'a>;
    type Output = RouteAgreementCheck;

    /// # Errors
    /// * [`ComputeError::DimensionMismatch`] for fewer than two spectra or
    ///   parts of different lengths.
    /// * [`ComputeError::BadShape`] for spectra on different frequency grids.
    fn compute(&self, input: RouteAgreementInput<'_>) -> Result<RouteAgreementCheck, ComputeError> {
        let spectra = input.spectra;
        if spectra.len() < 2 {
            return Err(ComputeError::DimensionMismatch {
                expected: 2,
                got: spectra.len(),
                what: "route agreement needs at least two spectra",
            });
        }
        let grid = &spectra[0].frequencies;
        let n = grid.len();
        for s in spectra {
            require_same_len("RouteAgreement array lengths", n, self.values(s).len())?;
            if s.frequencies != *grid {
                return Err(ComputeError::BadShape {
                    expected: format!("spectra on one frequency grid of {n} points"),
                    got: format!("a grid of {} points that differs", s.frequencies.len()),
                });
            }
        }
        let mut pairwise = Vec::new();
        let mut max_rms = 0.0_f64;
        for i in 0..spectra.len() {
            for j in (i + 1)..spectra.len() {
                let (a, b) = (self.values(spectra[i]), self.values(spectra[j]));
                let mut sum_sq = 0.0;
                let mut norm = 0.0;
                for k in 0..n {
                    let diff = a[k] - b[k];
                    sum_sq += diff * diff;
                    norm += (a[k].abs() + b[k].abs()) / 2.0;
                }
                let rms_abs = (sum_sq / n as F).sqrt();
                let scale = (norm / n as F).max(1e-30);
                let rms_rel = rms_abs / scale;
                max_rms = max_rms.max(rms_rel);
                pairwise.push((i, j, rms_rel));
            }
        }
        Ok(RouteAgreementCheck {
            pairwise,
            max_rms,
            passed: max_rms < self.tolerance,
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

fn require_tolerance(field: &'static str, value: F) -> Result<(), ComputeError> {
    if !(value.is_finite() && value >= 0.0) {
        return Err(ComputeError::OutOfRange {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

crate::compute::spec::compute_spec! {
    KramersKronig as "KramersKronig";
    params |s| {
        relative_tolerance: Float
            = KramersKronig::new().relative_tolerance
            => s.relative_tolerance;
        absolute_tolerance: Float
            = KramersKronig::new().absolute_tolerance
            => s.absolute_tolerance;
    }
    inputs {
        spectrum: Upstream("DielectricSpectrumResult");
    }
}

crate::compute::spec::compute_spec! {
    ConductivitySumRule as "ConductivitySumRule";
    params |s| {
        units: Units;
        tolerance: Float = ConductivitySumRule::DEFAULT_TOLERANCE => s.tolerance;
    }
    inputs {
        frequencies: Scalars;
        conductivity: Scalars;
        current_sq_mean: Scalar;
        volume: Scalar;
        temperature: Scalar;
    }
}

crate::compute::spec::compute_spec! {
    RouteAgreement as "RouteAgreement";
    params |s| {
        part: Choice(PermittivityPart::NAMES) = RouteAgreement::new().part => s.part;
        tolerance: Float = RouteAgreement::new().tolerance => s.tolerance;
    }
    inputs {
        spectra: UpstreamList("DielectricSpectrumResult");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spectrum(
        omega: &[F],
        eps_real: &[F],
        eps_imag: &[F],
        eps_inf: F,
    ) -> DielectricSpectrumResult {
        DielectricSpectrumResult {
            frequencies: Array1::from_vec(omega.to_vec()),
            eps_real: Array1::from_vec(eps_real.to_vec()),
            eps_imag: Array1::from_vec(eps_imag.to_vec()),
            epsilon_inf: eps_inf,
        }
    }

    #[test]
    fn kk_recovers_eps_inf_for_zero_loss() {
        // ε'' ≡ 0 ⇒ the KK integral is zero, so the recovered real part is ε∞
        // everywhere; matching eps_real ⇒ mae 0, passed.
        let s = spectrum(&[1.0, 2.0, 3.0, 4.0], &[2.5; 4], &[0.0; 4], 2.5);
        let out = KramersKronig::new()
            .compute(KramersKronigInput { spectrum: &s })
            .unwrap();
        assert!(out.mae < 1e-12, "mae {}", out.mae);
        assert!(out.passed);
        assert!(out.passed());
    }

    #[test]
    fn kk_requires_three_points() {
        let s = spectrum(&[1.0, 2.0], &[0.0; 2], &[0.0; 2], 1.0);
        assert!(matches!(
            KramersKronig::new().compute(KramersKronigInput { spectrum: &s }),
            Err(ComputeError::DimensionMismatch { .. })
        ));
    }

    /// The tolerances are configuration: a zero tolerance fails even an
    /// exact spectrum, and a negative one is refused.
    #[test]
    fn kk_tolerance_is_configuration() {
        let s = spectrum(&[1.0, 2.0, 3.0, 4.0], &[2.5; 4], &[0.0; 4], 2.5);
        let strict = KramersKronig::new().with_tolerance(0.0, 0.0).unwrap();
        assert!(
            !strict
                .compute(KramersKronigInput { spectrum: &s })
                .unwrap()
                .passed
        );
        assert!(matches!(
            KramersKronig::new().with_tolerance(-0.1, 0.0),
            Err(ComputeError::OutOfRange { .. })
        ));
        assert_eq!(KramersKronig::new().tolerance(), (0.1, 1e-2));
    }

    fn sum_rule(
        rule: &ConductivitySumRule,
        omega: &[F],
        sigma: &[F],
        csm: F,
        volume: F,
        temperature: F,
    ) -> Result<SumRuleCheck, ComputeError> {
        let (omega, sigma) = (
            Array1::from_vec(omega.to_vec()),
            Array1::from_vec(sigma.to_vec()),
        );
        rule.compute(SumRuleInput {
            frequencies: omega.view(),
            conductivity: sigma.view(),
            current_sq_mean: csm,
            volume,
            temperature,
        })
    }

    #[test]
    fn sum_rule_matches_constructed_integral() {
        // Constant σ over [0, 4]: trapezoidal integral = σ·4 = 8. Pick the
        // physical inputs so `expected` equals 8 ⇒ relative_error 0, passed.
        let volume = 100.0;
        let temperature = 300.0;
        let expected = 8.0;
        // expected = π/2 · ⟨J²⟩ / (3 V k_B T)  ⇒  ⟨J²⟩ = expected·3 V k_B T / (π/2)
        let current_sq_mean =
            expected * 3.0 * volume * UnitPreset::real().boltzmann() * temperature
                / (std::f64::consts::PI * 0.5);
        let out = sum_rule(
            &ConductivitySumRule::new(UnitPreset::real()),
            &[0.0, 2.0, 4.0],
            &[2.0; 3],
            current_sq_mean,
            volume,
            temperature,
        )
        .unwrap();
        assert!((out.integral - 8.0).abs() < 1e-12);
        assert!(out.relative_error.abs() < 1e-12, "{}", out.relative_error);
        assert!(out.passed);
    }

    #[test]
    fn sum_rule_rejects_nonpositive_volume() {
        assert!(matches!(
            sum_rule(
                &ConductivitySumRule::new(UnitPreset::real()),
                &[0.0, 1.0],
                &[1.0; 2],
                1.0,
                0.0,
                300.0
            ),
            Err(ComputeError::OutOfRange {
                field: "volume",
                ..
            })
        ));
        assert!(matches!(
            sum_rule(
                &ConductivitySumRule::new(UnitPreset::real()),
                &[0.0, 1.0],
                &[1.0; 2],
                -1.0,
                1.0,
                300.0
            ),
            Err(ComputeError::OutOfRange {
                field: "current_sq_mean",
                ..
            })
        ));
    }

    #[test]
    fn route_agreement_identical_is_zero() {
        let a = spectrum(&[0.0, 1.0, 2.0], &[1.0; 3], &[1.0, 2.0, 3.0], 1.0);
        let b = a.clone();
        let out = RouteAgreement::new()
            .compute(RouteAgreementInput { spectra: &[&a, &b] })
            .unwrap();
        assert_eq!(out.pairwise.len(), 1);
        assert_eq!((out.pairwise[0].0, out.pairwise[0].1), (0, 1));
        assert!(out.max_rms < 1e-12);
        assert!(out.passed);
    }

    #[test]
    fn route_agreement_flags_divergence() {
        let a = spectrum(&[0.0, 1.0, 2.0], &[1.0; 3], &[1.0; 3], 1.0);
        let b = spectrum(&[0.0, 1.0, 2.0], &[1.0; 3], &[2.0; 3], 1.0);
        let out = RouteAgreement::new()
            .compute(RouteAgreementInput { spectra: &[&a, &b] })
            .unwrap();
        assert!(out.max_rms > 0.1);
        assert!(!out.passed);
        // The real parts agree.
        let real = RouteAgreement::new()
            .with_part(PermittivityPart::Real)
            .compute(RouteAgreementInput { spectra: &[&a, &b] })
            .unwrap();
        assert!(real.passed);
    }

    #[test]
    fn route_agreement_needs_one_grid() {
        let a = spectrum(&[0.0, 1.0, 2.0], &[1.0; 3], &[1.0; 3], 1.0);
        let b = spectrum(&[0.0, 2.0, 4.0], &[1.0; 3], &[1.0; 3], 1.0);
        assert!(matches!(
            RouteAgreement::new().compute(RouteAgreementInput { spectra: &[&a, &b] }),
            Err(ComputeError::BadShape { .. })
        ));
        assert!(matches!(
            RouteAgreement::new().compute(RouteAgreementInput { spectra: &[&a] }),
            Err(ComputeError::DimensionMismatch { .. })
        ));
    }
}
