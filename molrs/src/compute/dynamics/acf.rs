//! Generic time-autocorrelation of a vector series, averaged over every time
//! origin.

use ndarray::{Array1, Array3, ArrayView3, s};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::LegacyCompute;
use crate::signal as sig;
use molrs::core::FrameAccess;

/// Autocorrelation curve, one entry per lag.
#[derive(Debug, Clone)]
pub struct AcfResult {
    /// Lags `t = 0, 1, …, max_lag`, in frames.
    pub lags: Array1<usize>,
    /// `C(t)`, in units of the input squared.
    pub acf: Array1<f64>,
}

impl ComputeResult for AcfResult {}

/// Time-autocorrelation of a vector series, averaged over all time origins.
#[derive(Debug, Clone, Copy, Default)]
pub struct Acf;

/// `(series (n_frames, n_entities, n_components), max_lag)` for [`Acf`].
pub type AcfArgs<'a> = (&'a Array3<f64>, usize);

/// The one multiple-time-origin autocorrelation estimator of molrs.
///
/// `C(t) = 1 / (N · (T − t)) · Σ_i Σ_α δv_iα(τ) · δv_iα(τ+t)` over the
/// `(n_frames T, n_entities N, n_components D)` series, where `δv = v − ⟨v⟩_τ`
/// when `mean_subtract` (each `(entity, component)` column loses its own
/// time mean) and `δv = v` otherwise. Every `(entity, component)` column goes
/// through the zero-padded FFT kernel [`crate::signal::acf_fft_accumulate`]
/// with one shared planner and scratch buffer; the normalisation is the
/// fused `(1/N)/(T − t)`.
///
/// A `(T, D)` Cartesian series is the `N = 1` case (`series.insert_axis(Axis(1))`);
/// a `(T, n_dof)` series averaged over degrees of freedom (the VACF) is the
/// `D = 1` case (`insert_axis(Axis(2))`). Both are views, no copy.
///
/// `max_lag` is clamped to `n_frames − 1`: a longer lag has no time origin to
/// average over, and returning a `0/0` entry would look like a measurement.
///
/// # Normalisation
///
/// The inner product runs over the `D` components, the sum over entities, and
/// the average over **all** `T − t` time origins — the *unbiased* normalisation.
/// Dividing by `T` instead (the *biased* form) tapers the tail toward zero; that
/// is a legitimate choice for spectral work, but it is not this one, and the two
/// disagree by `(T−t)/T` at every lag. There is deliberately no `biased()`
/// constructor, for the same reason [`Msd`](crate::compute::Msd) has no per-mode
/// factory: if a second normalisation is ever wanted it belongs in the argument
/// bundle, not in a second constructor.
///
/// # Algorithm and references
///
/// Each `(entity, component)` series is correlated by the **Wiener–Khinchin**
/// route — zero-pad to `≥ 2T`, forward FFT, squared magnitude, inverse FFT —
/// which gives the *linear* (not circular) autocorrelation in `O(T log T)`
/// rather than the `O(T · L)` direct double loop. The padding is what makes it
/// linear; see [`crate::signal::acf_fft`], which this delegates to and which
/// returns exactly the un-normalised numerator above.
///
/// - Wiener (1930), *Generalized harmonic analysis*; Khinchin (1934).
/// - Press et al., *Numerical Recipes*, §13.2 — the zero-padded linear-ACF recipe.
/// - Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed. (2017), §8.4 —
///   the multiple-time-origin correlation estimator for MD, direct and FFT routes.
/// - Kneller et al., *Comput. Phys. Commun.* **91** (1995) 191 (nMoldyn) — the
///   reference implementation of that estimator; the same route
///   [`Msd`](crate::compute::Msd) takes in `Window` mode.
///
/// # The transport correlators
///
/// The transport
/// correlators — [`Vacf`](crate::compute::Vacf) (mean-subtracted, averaged
/// over degrees of freedom), the dipole ACF of
/// [`DebyeRelaxation`](crate::compute::DebyeRelaxation) and the current ACF of
/// [`GreenKuboConductivity`](crate::compute::GreenKuboConductivity) — are this
/// function on a reshaped view, not re-derivations.
///
/// # Errors
///
/// [`ComputeError::EmptyInput`] for fewer than two frames, no entity or no
/// component.
pub fn autocorrelation(
    series: ArrayView3<'_, f64>,
    max_lag: usize,
    mean_subtract: bool,
) -> Result<AcfResult, ComputeError> {
    let (n_frames, n_entities, n_components) = series.dim();
    if n_frames < 2 || n_entities == 0 || n_components == 0 {
        return Err(ComputeError::EmptyInput);
    }
    let max_lag = max_lag.min(n_frames - 1);

    let mut planner = FftPlanner::new();
    let mut acf = Array1::<f64>::zeros(max_lag + 1);
    let mut column = vec![0.0_f64; n_frames];
    let mut scratch: Vec<Complex64> = Vec::new();
    let out = acf.as_slice_mut().expect("zeros array is contiguous");
    let inv_n_frames = 1.0 / n_frames as f64;

    for entity in 0..n_entities {
        for component in 0..n_components {
            let lane = series.slice(s![.., entity, component]);
            let mean = if mean_subtract {
                lane.iter().sum::<f64>() * inv_n_frames
            } else {
                0.0
            };
            for (slot, value) in column.iter_mut().zip(lane.iter()) {
                *slot = value - mean;
            }
            sig::acf_fft_accumulate(&mut planner, &column, max_lag, out, &mut scratch).map_err(
                |e| ComputeError::OutOfRange {
                    field: "acf_fft",
                    value: e.to_string(),
                },
            )?;
        }
    }

    // Unbiased: each lag averages exactly the origins that exist for it.
    let scale = 1.0 / n_entities as f64;
    for (lag, value) in acf.iter_mut().enumerate() {
        *value *= scale / (n_frames - lag) as f64;
    }

    Ok(AcfResult {
        lags: Array1::from_iter(0..=max_lag),
        acf,
    })
}

impl LegacyCompute for Acf {
    /// `(series (n_frames, n_entities, n_components), max_lag)`. The `frames`
    /// slice is unused — an autocorrelation is a property of the series, and
    /// the series need not be positions.
    type Args<'a> = AcfArgs<'a>;
    type Output = AcfResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        (series, max_lag): Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        autocorrelation(series.view(), max_lag, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    /// The definition, written out. Any FFT bug shows up as a mismatch here.
    fn direct(series: &Array3<f64>, max_lag: usize) -> Vec<f64> {
        let (n_frames, n_entities, n_components) = series.dim();
        (0..=max_lag)
            .map(|lag| {
                let mut sum = 0.0;
                for tau in 0..(n_frames - lag) {
                    for e in 0..n_entities {
                        for c in 0..n_components {
                            sum += series[[tau, e, c]] * series[[tau + lag, e, c]];
                        }
                    }
                }
                sum / (n_entities * (n_frames - lag)) as f64
            })
            .collect()
    }

    fn series(n_frames: usize, n_entities: usize, n_components: usize) -> Array3<f64> {
        // Deterministic, non-periodic, mean-nonzero: a periodic or zero-mean
        // input would hide a wrap-around or a normalisation slip.
        Array3::from_shape_fn((n_frames, n_entities, n_components), |(t, e, c)| {
            ((t * 7 + e * 13 + c * 3) as f64 * 0.37).sin() + 0.25 * (e as f64 + 1.0)
        })
    }

    #[test]
    fn fft_route_matches_the_direct_double_loop() {
        let s = series(64, 5, 3);
        let got = autocorrelation(s.view(), 20, false).unwrap();
        let want = direct(&s, 20);
        for (lag, (g, w)) in got.acf.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-10, "lag {lag}: {g} vs {w}");
        }
    }

    #[test]
    fn lag_zero_is_the_mean_square() {
        let s = series(32, 4, 3);
        let got = autocorrelation(s.view(), 5, false).unwrap();
        let n_entities = s.dim().1;
        let expected: f64 = s.iter().map(|v| v * v).sum::<f64>() / (n_entities * s.dim().0) as f64;
        assert!((got.acf[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn is_unbiased_not_biased() {
        // The two normalisations differ by (T-t)/T; pin which one this is.
        let s = series(40, 3, 2);
        let got = autocorrelation(s.view(), 10, false).unwrap();
        let want = direct(&s, 10);
        let biased_at_10 = want[10] * (40.0 - 10.0) / 40.0;
        assert!((got.acf[10] - want[10]).abs() < 1e-10);
        assert!(
            (got.acf[10] - biased_at_10).abs() > 1e-12,
            "unbiased and biased coincide; the test cannot tell them apart"
        );
    }

    #[test]
    fn max_lag_is_clamped_to_the_available_origins() {
        let s = series(8, 2, 3);
        let got = autocorrelation(s.view(), 999, false).unwrap();
        assert_eq!(got.acf.len(), 8);
        assert_eq!(got.lags[7], 7);
    }

    #[test]
    fn a_constant_series_autocorrelates_flat() {
        let s = Array3::from_elem((16, 3, 2), 2.0);
        let got = autocorrelation(s.view(), 6, false).unwrap();
        // every lag: (1/(N(T-t))) * N(T-t) * D * 4 = D * 4
        for value in got.acf.iter() {
            assert!((value - 8.0).abs() < 1e-10, "{value}");
        }
    }

    #[test]
    fn too_few_frames_is_an_error() {
        let s = series(1, 2, 3);
        assert!(autocorrelation(s.view(), 0, false).is_err());
    }
}
