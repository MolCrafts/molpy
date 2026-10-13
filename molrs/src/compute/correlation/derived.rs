//! Derived quantities of a [`TimeCorrelation`]: each method is a one-line
//! delegation to the matching [`Fit`], so the number a caller gets here is
//! the number the fit gives on the same arrays.

use ndarray::Array1;

use super::result::{CorrelationKind, Normalization, TimeCorrelation};
use crate::compute::ComputeError;
use crate::compute::{
    CumulativeTrapezoid, DebyeFit, DebyeFitResult, Fit, LinearFit, LinearFitResult, PowerSpectrum,
    SpectrumResult,
};

impl TimeCorrelation {
    /// `C(t) / C(0)`, rows included.
    ///
    /// # Errors
    ///
    /// [`ComputeError::OutOfRange`] when `C(0)` is zero or not finite — a
    /// displacement correlation, whose `C(0)` is `0` by definition, has no
    /// normalized form.
    pub fn normalized(&self) -> Result<TimeCorrelation, ComputeError> {
        let c0 = self.values[0];
        if !(c0.is_finite() && c0 != 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "TimeCorrelation::values[0]",
                value: format!("{c0} (cannot normalize by it)"),
            });
        }
        let mut out = self.clone();
        out.values.mapv_inplace(|v| v / c0);
        if let Some(rows) = out.per_entity.as_mut() {
            for mut row in rows.rows_mut() {
                let r0 = row[0];
                if r0.is_finite() && r0 != 0.0 {
                    row.mapv_inplace(|v| v / r0);
                }
            }
        }
        Ok(out)
    }

    /// The same curve under the [`Unbiased`](Normalization::Unbiased)
    /// normalization, re-scaled through `counts` and `n_entities`.
    pub fn to_unbiased(&self) -> TimeCorrelation {
        let mut out = self.clone();
        let n = self.n_entities as f64;
        match self.normalization {
            Normalization::Unbiased => {}
            Normalization::Biased => {
                let count0 = self.counts[0] as f64;
                for (v, &count) in out.values.iter_mut().zip(self.counts.iter()) {
                    *v *= count0 / count as f64;
                }
                if let Some(rows) = out.per_entity.as_mut() {
                    for mut row in rows.rows_mut() {
                        for (v, &count) in row.iter_mut().zip(self.counts.iter()) {
                            *v *= count0 / count as f64;
                        }
                    }
                }
            }
            Normalization::Raw => {
                for (v, &count) in out.values.iter_mut().zip(self.counts.iter()) {
                    *v /= n * count as f64;
                }
                if let Some(rows) = out.per_entity.as_mut() {
                    for mut row in rows.rows_mut() {
                        for (v, &count) in row.iter_mut().zip(self.counts.iter()) {
                            *v /= count as f64;
                        }
                    }
                }
            }
        }
        out.normalization = Normalization::Unbiased;
        out
    }

    /// Running trapezoidal integral `∫₀^{t} C(t′) dt′` on the `dt` grid, over
    /// the first `n_lags` samples (`None` = all): [`CumulativeTrapezoid`].
    pub fn integral(&self, n_lags: Option<usize>) -> Result<Array1<f64>, ComputeError> {
        Ok(CumulativeTrapezoid
            .fit((&self.values, self.dt, n_lags))?
            .integral)
    }

    /// The correlation time `τ = ∫₀^{t_max} C(t)/C(0) dt`: the integral of
    /// the normalized curve over its first `n_lags` samples (`None` = all).
    pub fn integrated_time(&self, n_lags: Option<usize>) -> Result<f64, ComputeError> {
        let integral = self.normalized()?.integral(n_lags)?;
        Ok(integral[integral.len() - 1])
    }

    /// Single-exponential fit `C(t)/C(0) = A·exp(−t/τ)`: [`DebyeFit`] on the
    /// normalized curve with this `dt`.
    pub fn exponential_fit(&self) -> Result<DebyeFitResult, ComputeError> {
        let normalized = self.normalized()?;
        DebyeFit.fit((&normalized.values, self.dt))
    }

    /// Ordinary least squares `C(t) = slope · t + intercept` over `fit.window`
    /// (fractions of the last lag) on the `lag_times` axis: [`LinearFit`].
    pub fn linear_fit(&self, fit: LinearFit) -> Result<LinearFitResult, ComputeError> {
        fit.fit((&self.lag_times(), &self.values))
    }

    /// Windowed one-sided power spectrum of the curve, with `dt` read as
    /// femtoseconds: [`PowerSpectrum`].
    pub fn spectrum(&self, transform: &PowerSpectrum) -> Result<SpectrumResult, ComputeError> {
        transform.fit((&self.values, self.dt))
    }

    /// The Einstein diffusion coefficient `D = slope / (2 · n_dims)` of a
    /// [`Displacement`](CorrelationKind::Displacement) curve, from its
    /// [`LinearFit`] over `window`.
    ///
    /// # Errors
    ///
    /// [`ComputeError::OutOfRange`] when the curve is not a displacement
    /// correlation or `n_dims == 0`; otherwise whatever the fit refuses.
    pub fn diffusion_coefficient(
        &self,
        n_dims: usize,
        window: (f64, f64),
    ) -> Result<f64, ComputeError> {
        if self.kind != CorrelationKind::Displacement {
            return Err(ComputeError::OutOfRange {
                field: "TimeCorrelation::kind",
                value: format!(
                    "{:?} (the Einstein relation needs a displacement correlation)",
                    self.kind
                ),
            });
        }
        einstein_diffusion_coefficient(&self.lag_times(), &self.values, n_dims, window)
    }
}

/// `D = slope / (2 · n_dims)` from the [`LinearFit`] of `msd` against
/// `lag_times` over `window`: the one place molrs turns an MSD slope into a
/// diffusion coefficient.
///
/// # Errors
///
/// [`ComputeError::OutOfRange`] for `n_dims == 0`; otherwise whatever the
/// [`LinearFit`] refuses (a bad window, too few points).
pub(crate) fn einstein_diffusion_coefficient(
    lag_times: &Array1<f64>,
    msd: &Array1<f64>,
    n_dims: usize,
    window: (f64, f64),
) -> Result<f64, ComputeError> {
    if n_dims == 0 {
        return Err(ComputeError::OutOfRange {
            field: "n_dims",
            value: n_dims.to_string(),
        });
    }
    let fit = LinearFit { window }.fit((lag_times, msd))?;
    Ok(fit.slope / (2.0 * n_dims as f64))
}

#[cfg(test)]
mod tests {
    use super::super::displacement::displacement_correlate;
    use super::super::engine::tests::series;
    use super::super::engine::{TimeCorrelationSpec, autocorrelate};
    use super::*;
    use crate::signal::WindowType;
    use ndarray::Array3;

    fn decaying() -> TimeCorrelation {
        // A series whose ACF decays: an AR(1)-like cumulative smoothing.
        let mut s = series(400, 2, 1);
        for t in 1..400 {
            for e in 0..2 {
                s[[t, e, 0]] = 0.9 * s[[t - 1, e, 0]] + 0.1 * s[[t, e, 0]];
            }
        }
        autocorrelate(
            s.view(),
            0.5,
            &TimeCorrelationSpec::new().max_lag(60).mean_subtract(true),
        )
        .unwrap()
    }

    #[test]
    fn normalized_starts_at_one_and_rows_follow() {
        let s = series(40, 3, 2);
        let c = autocorrelate(
            s.view(),
            1.0,
            &TimeCorrelationSpec::new().max_lag(8).per_entity(true),
        )
        .unwrap();
        let n = c.normalized().unwrap();
        assert_eq!(n.values[0], 1.0);
        for k in 0..=8 {
            assert_eq!(n.values[k], c.values[k] / c.values[0]);
        }
        let rows = n.per_entity.as_ref().unwrap();
        for e in 0..3 {
            assert_eq!(rows[[e, 0]], 1.0);
        }
        let msd =
            displacement_correlate(s.view(), None, 1.0, &TimeCorrelationSpec::new().max_lag(8))
                .unwrap();
        assert!(msd.normalized().is_err());
    }

    #[test]
    fn to_unbiased_undoes_biased_and_raw() {
        let s = series(40, 3, 2);
        let spec = TimeCorrelationSpec::new().max_lag(10).per_entity(true);
        let unb = autocorrelate(s.view(), 1.0, &spec).unwrap();
        for norm in [Normalization::Biased, Normalization::Raw] {
            let other = autocorrelate(s.view(), 1.0, &spec.normalization(norm))
                .unwrap()
                .to_unbiased();
            assert_eq!(other.normalization, Normalization::Unbiased);
            for k in 0..=10 {
                assert!(
                    (other.values[k] - unb.values[k]).abs() < 1e-12,
                    "{norm:?} k={k}"
                );
            }
            let (a, b) = (other.per_entity.unwrap(), unb.per_entity.as_ref().unwrap());
            for e in 0..3 {
                for k in 0..=10 {
                    assert!(
                        (a[[e, k]] - b[[e, k]]).abs() < 1e-12,
                        "{norm:?} e={e} k={k}"
                    );
                }
            }
        }
        let same = unb.to_unbiased();
        assert_eq!(same.values, unb.values);
    }

    #[test]
    fn integral_and_integrated_time_are_the_cumulative_trapezoid() {
        let c = decaying();
        let integral = c.integral(None).unwrap();
        let direct = CumulativeTrapezoid
            .fit((&c.values, c.dt, None))
            .unwrap()
            .integral;
        assert_eq!(integral, direct);
        let tau = c.integrated_time(Some(30)).unwrap();
        let normalized = c.normalized().unwrap();
        let want = CumulativeTrapezoid
            .fit((&normalized.values, c.dt, Some(30)))
            .unwrap()
            .integral[29];
        assert_eq!(tau, want);
        assert!(tau > 0.0);
    }

    #[test]
    fn exponential_fit_is_the_debye_fit_of_the_normalized_curve() {
        let c = decaying();
        let got = c.exponential_fit().unwrap();
        let normalized = c.normalized().unwrap();
        let want = DebyeFit.fit((&normalized.values, c.dt)).unwrap();
        assert_eq!(got.tau, want.tau);
        assert_eq!(got.amplitude, want.amplitude);
        // An exact exponential: τ recovered.
        let tau = 7.0;
        let exact = TimeCorrelation {
            lags: Array1::from_iter(0..50),
            dt: 0.5,
            values: Array1::from_iter((0..50).map(|k| 3.0 * (-(k as f64) * 0.5 / tau).exp())),
            counts: Array1::from_elem(50, 100),
            n_entities: 1,
            normalization: Normalization::Unbiased,
            kind: CorrelationKind::Auto,
            per_entity: None,
        };
        assert!((exact.exponential_fit().unwrap().tau - tau).abs() < 1e-9);
    }

    #[test]
    fn linear_fit_and_diffusion_coefficient_follow_the_einstein_relation() {
        // Straight-line motion: MSD = |v|² t², not linear; use a synthetic
        // diffusive curve instead: MSD = 6 D t with D = 0.25.
        let lags = 20;
        let dt = 0.5;
        let msd = TimeCorrelation {
            lags: Array1::from_iter(0..=lags),
            dt,
            values: Array1::from_iter((0..=lags).map(|k| 6.0 * 0.25 * k as f64 * dt)),
            counts: Array1::from_iter((0..=lags).map(|k| 100 - k)),
            n_entities: 4,
            normalization: Normalization::Unbiased,
            kind: CorrelationKind::Displacement,
            per_entity: None,
        };
        let fit = msd.linear_fit(LinearFit { window: (0.0, 1.0) }).unwrap();
        assert!((fit.slope - 1.5).abs() < 1e-12);
        let d = msd.diffusion_coefficient(3, (0.0, 1.0)).unwrap();
        assert!((d - 0.25).abs() < 1e-12);
        assert!(msd.diffusion_coefficient(0, (0.0, 1.0)).is_err());
        let mut not_msd = msd.clone();
        not_msd.kind = CorrelationKind::Auto;
        assert!(not_msd.diffusion_coefficient(3, (0.0, 1.0)).is_err());
    }

    #[test]
    fn spectrum_is_the_power_spectrum_fit() {
        let c = decaying();
        let via = c.spectrum(&PowerSpectrum::default()).unwrap();
        let direct = PowerSpectrum::default().fit((&c.values, c.dt)).unwrap();
        assert_eq!(via.intensities, direct.intensities);
        assert_eq!(via.frequencies_cm1, direct.frequencies_cm1);
        let hann = c
            .spectrum(&PowerSpectrum::with_window(WindowType::Hann))
            .unwrap();
        assert_ne!(hann.intensities, via.intensities);
    }

    #[test]
    fn a_three_d_random_walk_has_a_diffusion_coefficient() {
        // Sanity on the full path: engine → displacement → linear fit.
        let mut w = Array3::<f64>::zeros((300, 10, 3));
        let mut x = 12345u64;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 2000) as f64 / 1000.0 - 1.0
        };
        for t in 1..300 {
            for e in 0..10 {
                for c in 0..3 {
                    w[[t, e, c]] = w[[t - 1, e, c]] + rnd();
                }
            }
        }
        let msd =
            displacement_correlate(w.view(), None, 1.0, &TimeCorrelationSpec::new().max_lag(50))
                .unwrap();
        let d = msd.diffusion_coefficient(3, (0.2, 1.0)).unwrap();
        // Step variance per component is 1/3 → MSD ≈ 3·(1/3)·t = t → D ≈ 1/6.
        assert!((d - 1.0 / 6.0).abs() < 0.05, "{d}");
    }
}
